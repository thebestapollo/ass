#!/usr/bin/env python3
"""
Audio track sync tool or, audio super sync (ASS)

Given 2+ almost-identical audio files, estimate relative delays via cross-correlation
(on a bandlimited, envelope-enhanced mono reference), then optionally write
shifted/aligned outputs and print offsets.

Usage:
    python sync_audio.py ref.wav other1.wav [other2.wav ...] \
            --out aligned/ --write --method hilbert --max-shift-ms 2000 \
            --corr-channels auto|lfe|surrounds|center|fronts|all|indices [--corr-indices 4,5]

Features:
- Robust delay estimation using FFT cross-correlation on RMS or Hilbert envelope
- Handles different sample rates via resampling to a common rate
- Channel selection for correlation (e.g., LFE or surrounds) to avoid dialogue leakage
- Optional band-pass to emphasize shared spectral content
- Limit maximum search window (useful when drift not expected)
- Write time-shifted versions padded with silence to align to reference start
- Input formats: uses soundfile/libsndfile where possible; falls back to ffmpeg
    decoding for formats like AC-3/E-AC-3 or audio inside MP4/MKV

Outputs:
- Prints per-file estimated offset in ms (positive => file starts later than ref)
- If --write, saves aligned files into --out directory

Note: This estimates a single global offset (no time-warp). If recordings have drift,
use DAW time-stretch or dynamic time warping tools.
"""
from __future__ import annotations
import argparse
import os
import io
import shutil
import subprocess
from dataclasses import dataclass
import json
from typing import List, Tuple, Optional, cast

import numpy as np
import soundfile as sf
from scipy import signal
from numpy.typing import NDArray


DEFAULT_TARGET_SR = 16000  # Hz for correlation (keeps it fast, robust)

# Type aliases for clarity (float32 throughout the audio path)
FloatArray = NDArray[np.float32]
MonoArray = FloatArray
MultiChArray = FloatArray  # shape: (n, ch)


FFMPEG_TRIGGER_EXTS = {'.ac3', '.eac3', '.dts', '.dtshd', '.mka', '.mkv', '.mp4', '.m4a', '.m4v', '.ts', '.vob'}


def _ffmpeg_exists() -> bool:
    return shutil.which('ffmpeg') is not None


def _ffprobe_exists() -> bool:
    return shutil.which('ffprobe') is not None


def _ffprobe_audio_info(path: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """Return (sample_rate, channels, duration_sec) using ffprobe if available.

    Any field may be None if probing fails.
    """
    if not _ffprobe_exists():
        return None, None, None
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=sample_rate,channels',
        '-show_entries', 'format=duration',
        '-of', 'json',
        path
    ]
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data = json.loads(proc.stdout.decode('utf-8'))
        sr: Optional[int] = None
        ch: Optional[int] = None
        dur: Optional[float] = None
        if 'streams' in data and data['streams']:
            s0 = data['streams'][0]
            if 'sample_rate' in s0 and s0['sample_rate'] not in (None, 'N/A', ''):
                try:
                    sr = int(s0['sample_rate'])
                except Exception:
                    sr = None
            if 'channels' in s0 and s0['channels'] not in (None, 'N/A', ''):
                try:
                    ch = int(s0['channels'])
                except Exception:
                    ch = None
        if 'format' in data and 'duration' in data['format'] and data['format']['duration'] not in (None, 'N/A', ''):
            try:
                dur = float(data['format']['duration'])
            except Exception:
                dur = None
        return sr, ch, dur
    except Exception:
        return None, None, None


def _read_audio_via_ffmpeg(path: str) -> Tuple[MultiChArray, int]:
    """Decode any input to WAV (float32) via ffmpeg and read with soundfile.

    Preserves original channel count and sample rate.
    """
    if not _ffmpeg_exists():
        raise RuntimeError('ffmpeg not found on PATH; install it to decode this format')

    # Build ffmpeg command to emit float32 WAV to stdout
    cmd = [
        'ffmpeg', '-v', 'error', '-nostdin',
        '-i', path,
        '-map', 'a:0',  # pick first audio stream
        '-c:a', 'pcm_f32le',
        '-f', 'wav',
        'pipe:1'
    ]
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'ffmpeg failed to decode {os.path.basename(path)}: {e.stderr.decode("utf-8", "ignore")}')

    bio = io.BytesIO(proc.stdout)
    data, sr = sf.read(bio, always_2d=True, dtype='float32')
    return cast(MultiChArray, data), int(sr)


def _build_pan_filter_for_mode(channels: int, mode: str, indices: Optional[List[int]]) -> Optional[str]:
    """Return an ffmpeg pan filter string to produce mono from selected channels.

    If None is returned, caller may use '-ac 1' for generic downmix.
    """
    def clamp(ids: List[int]) -> List[int]:
        return [i for i in ids if 0 <= i < channels]

    sel: List[int]
    if mode == 'indices' and indices:
        sel = clamp(indices)
    elif mode == 'all' or mode == 'auto':
        # Let ffmpeg handle downmix with -ac 1
        return None
    elif mode == 'fronts':
        sel = clamp([0, 1]) if channels >= 2 else list(range(channels))
    elif mode == 'center':
        sel = clamp([2]) if channels >= 3 else list(range(channels))
    elif mode == 'lfe':
        sel = clamp([3]) if channels >= 4 else list(range(channels))
    elif mode == 'surrounds':
        if channels >= 6:
            sel = clamp([4, 5] + ([6, 7] if channels >= 8 else []))
        elif channels >= 4:
            sel = clamp([channels-2, channels-1])
        else:
            sel = list(range(channels))
    else:
        return None

    if not sel:
        return None
    # Average selected channels equally into mono c0
    coeff = 1.0 / len(sel)
    terms = [f"{coeff:.6f}*c{idx}" for idx in sel]
    return f"pan=mono|c0={'+' .join(terms)}"


def read_corr_mono_segment(
    path: str,
    mode: str,
    indices: Optional[List[int]],
    target_sr: int,
    start_sec: Optional[float],
    max_seconds: Optional[float]
) -> Tuple[MonoArray, int]:
    """Decode a mono correlation segment efficiently.

    Prefers ffmpeg to downmix to mono and resample before handing bytes to Python.
    Falls back to soundfile with partial reads if necessary.
    """
    # Try ffmpeg fast path
    if _ffmpeg_exists():
        ch: Optional[int] = None
        _sr_probe, ch_probe, _dur = _ffprobe_audio_info(path)
        if ch_probe is not None:
            ch = ch_probe
        pan: Optional[str] = None
        if ch is not None:
            pan = _build_pan_filter_for_mode(ch, mode, indices)
        cmd = ['ffmpeg', '-v', 'error', '-nostdin']
        if start_sec is not None and start_sec > 0:
            cmd += ['-ss', f"{start_sec:.3f}"]
        cmd += ['-i', path]
        if max_seconds is not None and max_seconds > 0:
            cmd += ['-t', f"{max_seconds:.3f}"]
        if pan is not None:
            cmd += ['-filter:a', pan]
        else:
            cmd += ['-ac', '1']  # generic mono downmix
        cmd += [
            '-ar', str(target_sr),
            '-c:a', 'pcm_f32le',
            '-f', 'wav',
            'pipe:1'
        ]
        try:
            proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            bio = io.BytesIO(proc.stdout)
            data, sr = sf.read(bio, always_2d=False, dtype='float32')
            # data may be (n,) mono already
            if data.ndim == 2:
                data = data.mean(axis=1)
            return cast(MonoArray, np.asarray(data, dtype=np.float32)), int(sr)
        except Exception:
            # fall back to soundfile path below
            pass

    # Fallback: soundfile partial read (may fail for some codecs)
    # Use start/frames to avoid loading entire file
    info_sr: Optional[int] = None
    try:
        # Try reading 1 frame to get sample rate cheaply
        _probe, probe_sr = sf.read(path, frames=1, dtype='float32', always_2d=True)
        info_sr = int(probe_sr)
    except Exception:
        pass
    x: MultiChArray
    if info_sr is None:
        # As a last resort, read whole file (may fail for codecs not supported by libsndfile)
        data_full, sr_all = sf.read(path, always_2d=True, dtype='float32')
        x = cast(MultiChArray, data_full)
        sr = int(sr_all)
    else:
        sr = info_sr
        start_frame: int = int(round(max(0.0, (start_sec or 0.0)) * sr))
        frames: Optional[int] = None
        if max_seconds is not None and max_seconds > 0:
            frames = int(round(max_seconds * sr))
        data_arr, sr2 = sf.read(path, start=start_frame, frames=(frames if frames is not None else -1), dtype='float32', always_2d=True)
        x = cast(MultiChArray, data_arr)
        sr = int(sr2)
    # Downmix to mono according to mode
    mono = select_channels_for_correlation(x, mode, indices)
    mono_ds, ds_sr = resample_if_needed(mono, sr, target_sr)
    return mono_ds, ds_sr


def read_audio(path: str) -> Tuple[MultiChArray, int]:
    """Read audio using soundfile; fall back to ffmpeg for unsupported formats.

    This adds support for AC-3/E-AC-3 and common container formats when ffmpeg is installed.
    """
    ext = os.path.splitext(path)[1].lower()
    try_sf_first = ext not in FFMPEG_TRIGGER_EXTS

    if try_sf_first:
        try:
            data, sr = sf.read(path, always_2d=True, dtype='float32')
            return cast(MultiChArray, data), int(sr)
        except Exception:
            # fall back
            pass

    # Try ffmpeg fallback
    try:
        data, sr = _read_audio_via_ffmpeg(path)
        return data, sr
    except Exception:
        # If fallback fails and we didn't try SF yet (because ext triggered ffmpeg), try SF as last resort
        if not try_sf_first:
            try:
                data2, sr2 = sf.read(path, always_2d=True, dtype='float32')
                return cast(MultiChArray, data2), int(sr2)
            except Exception:
                pass
        raise


def to_mono(x: MultiChArray) -> MonoArray:
    m = x.mean(axis=1)
    return cast(MonoArray, m.astype(np.float32, copy=False))


def select_channels_for_correlation(x: MultiChArray, mode: str, indices: Optional[List[int]] = None) -> MonoArray:
    """Select a subset of channels for correlation and downmix to mono.

    Heuristics assume common layouts:
    - 5.1 (6 ch): [L, R, C, LFE, Ls, Rs]
    - 7.1 (8 ch): [L, R, C, LFE, Ls, Rs, Lb, Rb] (or S/Back variations)
    If the exact channel isn't available, falls back gracefully.
    """
    ch = x.shape[1]
    sel: Optional[List[int]] = None
    def clamp(ids: List[int]) -> List[int]:
        return [i for i in ids if 0 <= i < ch]

    if mode == 'indices' and indices:
        sel = clamp(indices)
    elif mode == 'all':
        sel = list(range(ch))
    elif mode == 'fronts':
        sel = clamp([0, 1]) if ch >= 2 else list(range(ch))
    elif mode == 'center':
        sel = clamp([2]) if ch >= 3 else list(range(ch))
    elif mode == 'lfe':
        # Commonly index 3
        if ch >= 4:
            sel = clamp([3])
        else:
            sel = list(range(ch))
    elif mode == 'surrounds':
        if ch >= 6:
            sel = clamp([4, 5])
            # If 7.1 or greater, include backs too
            if ch >= 8:
                sel.extend(clamp([6, 7]))
        elif ch >= 4:
            # Use last two as a weak guess
            sel = clamp([ch-2, ch-1])
        else:
            sel = list(range(ch))
    else:  # 'auto'
        if ch >= 4:
            sel = clamp([3])  # prefer LFE
        elif ch >= 6:
            sel = clamp([4, 5])
        else:
            sel = list(range(ch))

    if not sel:
        sel = list(range(ch))

    y = x[:, sel]
    if y.ndim == 1 or y.shape[1] == 1:
        return cast(MonoArray, y.reshape(-1).astype(np.float32, copy=False))
    return cast(MonoArray, y.mean(axis=1).astype(np.float32, copy=False))


def resample_if_needed(x: MonoArray, sr: int, target_sr: int) -> Tuple[MonoArray, int]:
    if sr == target_sr:
        return x, sr
    # Use polyphase resampling for speed and quality
    g = np.gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    y = signal.resample_poly(x, up, down)
    y = cast(MonoArray, np.asarray(y, dtype=np.float32))
    return y, target_sr


def bandpass(x: MonoArray, sr: int, lo_hz: Optional[float], hi_hz: Optional[float]) -> MonoArray:
    if lo_hz is None and hi_hz is None:
        return x
    nyq = sr / 2.0
    if lo_hz is None:
        assert hi_hz is not None
        btype = 'lowpass'
        Wn: float | List[float] = float(hi_hz / nyq)
    elif hi_hz is None:
        assert lo_hz is not None
        btype = 'highpass'
        Wn = float(lo_hz / nyq)
    else:
        btype = 'bandpass'
        Wn = [float(lo_hz / nyq), float(hi_hz / nyq)]
    b_a = signal.butter(4, Wn, btype=btype)
    # butter may return (b, a) or SOS depending on args; with our usage, it's (b, a)
    assert isinstance(b_a, tuple) and len(b_a) == 2
    b = np.asarray(b_a[0])
    a = np.asarray(b_a[1])
    y = signal.filtfilt(b, a, x)
    return cast(MonoArray, np.asarray(y, dtype=np.float32))


def envelope(x: MonoArray, method: str, sr: int) -> MonoArray:
    x = x.astype(np.float32, copy=False)
    if method == 'rms':
        # Fast RMS over 10 ms window
        win = max(1, int(0.01 * sr))
        w = np.ones(win) / win
        e = np.sqrt(signal.convolve(x**2, w, mode='same'))
        return cast(MonoArray, e.astype(np.float32, copy=False))
    elif method == 'hilbert':
        # hilbert returns complex array; take magnitude as envelope
        analytic = np.asarray(signal.hilbert(x))
        env = np.abs(analytic)
        return cast(MonoArray, env.astype(np.float32, copy=False))
    else:
        return x


def fft_xcorr(a: MonoArray, b: MonoArray, max_lag: Optional[int]) -> int:
    n = int(2 ** np.ceil(np.log2(len(a) + len(b) - 1)))
    A = np.fft.rfft(a, n)
    B = np.fft.rfft(b, n)
    xcorr = np.fft.irfft(A * np.conj(B), n)
    # Re-center correlation to support negative lags
    xcorr = np.concatenate((xcorr[-(len(b)-1):], xcorr[:len(a)]))
    if max_lag is not None:
        # zero-lag index in the trimmed/centered xcorr is len(b)-1
        center = len(b) - 1
        start = center - max_lag
        end = center + max_lag + 1
        window = xcorr[start:end]
        lag = np.argmax(window) + start - center
    else:
        lag = int(np.argmax(xcorr) - (len(b) - 1))
    return int(lag)


@dataclass
class OffsetResult:
    file: str
    samples: int
    ms: float


def estimate_offset(ref_mono: MonoArray, x_mono: MonoArray, sr: int, method: str, max_shift_ms: Optional[int]) -> int:
    # Normalize to reduce gain differences
    def norm(y: MonoArray) -> MonoArray:
        s = float(np.std(y)) + 1e-9
        z = (y - float(np.mean(y))) / s
        return cast(MonoArray, z.astype(np.float32, copy=False))

    a = norm(ref_mono)
    b = norm(x_mono)

    # Optional envelope extraction
    a_env = envelope(a, method, sr)
    b_env = envelope(b, method, sr)

    # Downsample envelopes for speed in correlation
    decim = max(1, int(sr // 4000))
    if decim > 1:
        a_env = cast(MonoArray, np.asarray(signal.decimate(a_env, decim, ftype='fir', zero_phase=True), dtype=np.float32))
        b_env = cast(MonoArray, np.asarray(signal.decimate(b_env, decim, ftype='fir', zero_phase=True), dtype=np.float32))
        corr_sr = sr // decim
    else:
        corr_sr = sr

    max_lag = None
    if max_shift_ms is not None:
        max_lag = int((max_shift_ms / 1000.0) * corr_sr)

    lag_decim = fft_xcorr(a_env, b_env, max_lag)
    lag_samples = lag_decim * decim
    return lag_samples


def apply_shift_multichannel(x: MultiChArray, shift: int) -> MultiChArray:
    """Shift by inserting/removing samples at the start; positive shift delays x."""
    n, ch = x.shape
    if shift == 0:
        return x
    if shift > 0:
        pad = np.zeros((shift, ch), dtype=x.dtype)
        return cast(MultiChArray, np.vstack([pad, x]))
    else:
        s = -shift
        if s >= n:
            return cast(MultiChArray, np.zeros((0, ch), dtype=x.dtype))
        return cast(MultiChArray, x[s:])


def trim_to_min_length(arrays: List[MultiChArray]) -> List[MultiChArray]:
    if not arrays:
        return arrays
    m = min(a.shape[0] for a in arrays)
    return [a[:m] for a in arrays]


def main():
    ap = argparse.ArgumentParser(description='Sync nearly-identical audio tracks using cross-correlation.')
    ap.add_argument('files', nargs='+', help='Audio files. First is the reference.')
    ap.add_argument('--method', choices=['raw', 'rms', 'hilbert'], default='hilbert', help='Signal used for correlation.')
    ap.add_argument('--target-sr', type=int, default=DEFAULT_TARGET_SR, help='Resample rate for correlation.')
    ap.add_argument('--bandpass', type=float, nargs=2, metavar=('LOW_HZ','HIGH_HZ'), default=None, help='Optional band-pass for correlation path.')
    ap.add_argument('--max-shift-ms', type=int, default=2000, help='Maximum expected absolute shift in milliseconds.')
    ap.add_argument('--write', action='store_true', help='Write aligned files to --out directory.')
    ap.add_argument('--out', type=str, default='aligned', help='Output directory when using --write.')
    ap.add_argument('--suffix', type=str, default='_aligned', help='Suffix for written filenames.')
    ap.add_argument('--no-trim', action='store_true', help='Do not trim aligned outputs to common length.')
    ap.add_argument('--corr-channels', type=str, default='auto', choices=['auto','lfe','surrounds','center','fronts','all','indices'], help='Channels to use for correlation downmix.')
    ap.add_argument('--corr-indices', type=str, default=None, help='Comma-separated 0-based channel indices for --corr-channels indices.')
    ap.add_argument('--corr-seconds', type=float, default=None, help='Use at most this many seconds of audio for correlation (e.g., 300 for 5 minutes). Reduces memory/CPU.')
    ap.add_argument('--corr-start-seconds', type=float, default=0.0, help='Start at this position (seconds) when extracting correlation window.')
    ap.add_argument('--write-mode', choices=['ffmpeg', 'memory'], default='ffmpeg', help='How to write aligned outputs. ffmpeg mode avoids loading entire files into RAM.')

    args = ap.parse_args()
    if len(args.files) < 2:
        raise SystemExit('Provide at least two files: reference and one or more targets.')

    ref_path, *others = args.files

    # Parse correlation channel indices if provided
    sel_idx: Optional[List[int]] = None
    if args.corr_channels == 'indices' and args.corr_indices:
        try:
            sel_idx = [int(s.strip()) for s in args.corr_indices.split(',') if s.strip()]
        except Exception:
            raise SystemExit('Invalid --corr-indices; expected comma-separated integers')

    # Prepare correlation reference path (load only the needed window, as mono)
    ref_corr, corr_sr = read_corr_mono_segment(
        ref_path, args.corr_channels, sel_idx, args.target_sr,
        args.corr_start_seconds, args.corr_seconds
    )
    if args.bandpass is not None:
        ref_corr = bandpass(ref_corr, corr_sr, args.bandpass[0], args.bandpass[1])

    results: List[OffsetResult] = []
    shifted_arrays: List[Tuple[MultiChArray, int]] = []
    written_paths: List[str] = []

    for p in others:
        # Load only the correlation window for this file
        x_corr, _corr_sr2 = read_corr_mono_segment(
            p, args.corr_channels, sel_idx, corr_sr,
            args.corr_start_seconds, args.corr_seconds
        )
        if args.bandpass is not None:
            x_corr = bandpass(x_corr, corr_sr, args.bandpass[0], args.bandpass[1])

        shift_samples = estimate_offset(ref_corr, x_corr, corr_sr, args.method, args.max_shift_ms)
        shift_ms = 1000.0 * shift_samples / corr_sr
        results.append(OffsetResult(file=p, samples=shift_samples, ms=shift_ms))
        # We'll write later using streaming or memory depending on args

    # Prepare outputs
    for r in results:
        sign = '+' if r.samples >= 0 else ''
        print(f'{os.path.basename(r.file)}: shift {sign}{r.ms:.1f} ms (corr domain samples={r.samples})')

    if args.write:
        os.makedirs(args.out, exist_ok=True)

        # Compute pad for reference (in milliseconds)
        pad_ref_ms = max([0.0] + [max(0.0, r.ms) for r in results])

        if args.write_mode == 'ffmpeg' and _ffmpeg_exists():
            # Probe durations and sample rates
            ref_sr, ref_ch, ref_dur = _ffprobe_audio_info(ref_path)
            if ref_sr is None or ref_ch is None or ref_dur is None:
                # best-effort fallback to soundfile minimal read (duration unknown)
                try:
                    d, srr = sf.read(ref_path, frames=1, dtype='float32', always_2d=True)
                    ref_sr = int(srr)
                    ref_ch = int(d.shape[1] if d.ndim == 2 else 1)
                except Exception:
                    ref_sr = ref_sr or DEFAULT_TARGET_SR
                    ref_ch = ref_ch or 2
                # duration might remain None

            # Durations for others
            others_info: List[Tuple[str, int, int, Optional[float]]] = []
            for p in others:
                sr, ch, dur = _ffprobe_audio_info(p)
                if sr is None or ch is None:
                    try:
                        d, srr = sf.read(p, frames=1, dtype='float32', always_2d=True)
                        sr = int(srr)
                        ch = int(d.shape[1] if d.ndim == 2 else 1)
                    except Exception:
                        sr = sr or DEFAULT_TARGET_SR
                        ch = ch or 2
                # duration may remain None
                others_info.append((p, sr, ch, dur))

            # Compute minimal common length after shifts (in seconds)
            def length_after_shift_sec(dur_s: float, shift_ms: float) -> float:
                if shift_ms >= 0:
                    return dur_s + (shift_ms / 1000.0)
                else:
                    adv = (-shift_ms) / 1000.0
                    return max(0.0, dur_s - adv)

            min_out_len: Optional[float] = None
            have_all_durations = (ref_dur is not None) and all(dur is not None for (_p, _sr, _ch, dur) in others_info)
            if not args.no_trim and have_all_durations:
                ref_out_len = (ref_dur or 0.0) + (pad_ref_ms / 1000.0)
                x_out_lens = [length_after_shift_sec(d or 0.0, r.ms) for (_p, _sr, _ch, d), r in zip(others_info, results)]
                min_out_len = min([ref_out_len] + x_out_lens)

            # Write reference with padding via ffmpeg
            base_ref = os.path.splitext(os.path.basename(ref_path))[0]
            out_ref_path = os.path.join(args.out, f'{base_ref}{args.suffix}.wav')
            ref_filters: List[str] = []
            if pad_ref_ms > 0:
                ref_filters.append(f"adelay=delays={int(round(pad_ref_ms))}:all=1")
            if min_out_len is not None:
                ref_filters.append(f"atrim=end={min_out_len:.6f}")
                ref_filters.append("asetpts=PTS-STARTPTS")
            cmd = ['ffmpeg', '-y', '-v', 'error', '-nostdin', '-i', ref_path]
            if ref_filters:
                cmd += ['-filter:a', ','.join(ref_filters)]
            cmd += ['-c:a', 'pcm_s16le', out_ref_path]
            try:
                subprocess.run(cmd, check=True)
                written_paths.append(out_ref_path)
            except subprocess.CalledProcessError as e:
                raise SystemExit(f'ffmpeg failed writing reference: {e}')

            # Write others with their shifts
            for (p, sr, ch, dur), r in zip(others_info, results):
                base = os.path.splitext(os.path.basename(p))[0]
                out_path = os.path.join(args.out, f'{base}{args.suffix}.wav')
                filters: List[str] = []
                if r.ms > 0:
                    filters.append(f"adelay=delays={int(round(r.ms))}:all=1")
                elif r.ms < 0:
                    adv_sec = (-r.ms) / 1000.0
                    filters.append(f"atrim=start={adv_sec:.6f}")
                    filters.append("asetpts=PTS-STARTPTS")
                if min_out_len is not None:
                    filters.append(f"atrim=end={min_out_len:.6f}")
                    filters.append("asetpts=PTS-STARTPTS")
                cmd = ['ffmpeg', '-y', '-v', 'error', '-nostdin', '-i', p]
                if filters:
                    cmd += ['-filter:a', ','.join(filters)]
                cmd += ['-c:a', 'pcm_s16le', out_path]
                try:
                    subprocess.run(cmd, check=True)
                    written_paths.append(out_path)
                except subprocess.CalledProcessError as e:
                    raise SystemExit(f'ffmpeg failed writing {os.path.basename(p)}: {e}')

            print(f'Wrote {len(written_paths)} files to {args.out}')

        else:
            # Memory mode: load, shift, and write (may require lots of RAM)
            # Load reference fully
            ref, ref_sr = read_audio(ref_path)
            # Determine max positive shift in original rates to pad ref for alignment at common start
            pos_shifts: List[int] = []
            # We need each other's sample rate to convert shift samples to their original domain
            others_full: List[Tuple[str, MultiChArray, int]] = []
            for p in others:
                x, x_sr = read_audio(p)
                others_full.append((p, x, x_sr))
            for (p, _x, x_sr), r in zip(others_full, results):
                pos_shifts.append(max(0, int(round(r.ms / 1000.0 * x_sr))))
            pad_ref = max([0] + pos_shifts)
            ref_out = ref
            if pad_ref > 0:
                ref_out = np.vstack([np.zeros((pad_ref, ref.shape[1]), dtype=ref.dtype), ref])

            outputs: List[MultiChArray] = [ref_out]
            for (p, x, x_sr), r in zip(others_full, results):
                shift_orig = int(round(r.ms / 1000.0 * x_sr))
                x_shifted = apply_shift_multichannel(x, shift_orig)
                shifted_arrays.append((x_shifted, x_sr))
                outputs.append(x_shifted)
            if not args.no_trim:
                outputs = trim_to_min_length(outputs)

            # Write files: reference plus each other
            base_ref = os.path.splitext(os.path.basename(ref_path))[0]
            out_ref_path = os.path.join(args.out, f'{base_ref}{args.suffix}.wav')
            sf.write(out_ref_path, outputs[0], ref_sr)
            written_paths.append(out_ref_path)

            for (p, _x, x_sr), out_data in zip(others_full, outputs[1:]):
                base = os.path.splitext(os.path.basename(p))[0]
                out_path = os.path.join(args.out, f'{base}{args.suffix}.wav')
                sf.write(out_path, out_data, x_sr)
                written_paths.append(out_path)

            print(f'Wrote {len(written_paths)} files to {args.out}')


if __name__ == '__main__':
    main()
