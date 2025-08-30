# Audio Super Sync (ASS)

CLI to estimate and correct offsets between nearly-identical audio tracks using robust cross-correlation on smoothed envelopes.

## Install

Create a virtual environment and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: install ffmpeg for broader codec support (AC-3/E-AC-3, audio in MP4/MKV) and faster, low‑memory output writing.

## Usage

Basic: print offsets vs the first (reference) file:

```bash
python sync_audio.py ref.wav take2.wav take3.wav
```

Write aligned files into `aligned/`:

```bash
python sync_audio.py ref.wav take2.wav --write --out aligned
```

## Options

Positional:
- files: one or more audio files. The first is the reference; all others are aligned to it.

General:
- --method [raw|rms|hilbert] (default: hilbert) — envelope used for correlation.
- --target-sr INT (default: 16000) — resample rate used only in the correlation path.
- --bandpass LOW HIGH — optional band-pass in Hz for correlation only, e.g. `--bandpass 200 4000`.
- --max-shift-ms INT (default: 2000) — maximum absolute shift to search in ms.

Correlation window and channels:
- --corr-seconds FLOAT — use at most this many seconds for correlation (speeds up/lowers memory).
- --corr-start-seconds FLOAT (default: 0.0) — start of the correlation window (in seconds).
- --corr-channels [auto|lfe|surrounds|center|fronts|all|indices] (default: auto) — which channels to downmix for correlation.
- --corr-indices CSV — comma-separated 0-based indices, used when `--corr-channels indices`.

Writing outputs:
- --write — actually write aligned audio files.
- --out PATH (default: aligned) — output directory when using `--write`.
- --suffix STR (default: _aligned) — filename suffix for written files.
- --no-trim — do not trim outputs to a common length. By default, outputs are trimmed when possible.
- --write-mode [ffmpeg|memory] (default: ffmpeg) — writing strategy. ffmpeg mode is streaming/low‑RAM and preferred when ffmpeg is installed; memory mode loads full files into RAM.

## Output semantics

- Positive shift means the target starts later than the reference; we insert leading silence on the target to align.
- A single global offset is estimated (no drift correction/time-stretch).

## Examples

- Compare an AAC/M4A and an AC-3 file (requires ffmpeg for decoding), print offsets only:

```bash
python sync_audio.py source1.m4a source2.ac3
```

- Write aligned WAVs next to the reference with a custom suffix:

```bash
python sync_audio.py ref.wav take2.wav take3.wav --write --out aligned --suffix _sync
```

- Focus correlation on dialogue band and reduce search window to ±1.5 s:

```bash
python sync_audio.py ref.wav take2.wav --bandpass 200 4000 --max-shift-ms 1500
```

- Use only the surrounds for correlation to avoid center/dialogue leakage:

```bash
python sync_audio.py ref_5.1.wav take2_5.1.wav --corr-channels surrounds
```

- Select custom channel indices (e.g., channels 4 and 5 in a 7.1 track):

```bash
python sync_audio.py ref_7.1.wav take2_7.1.wav --corr-channels indices --corr-indices 4,5
```

- Correlate using a 5-minute window starting at 60 s to save time/memory:

```bash
python sync_audio.py ref.wav take2.wav --corr-start-seconds 60 --corr-seconds 300
```

- Switch envelope method and correlation sample rate:

```bash
python sync_audio.py ref.wav take2.wav --method rms --target-sr 24000
```

- Write without ffmpeg by using in-memory mode (uses more RAM):

```bash
python sync_audio.py ref.wav take2.wav --write --write-mode memory
```

## Notes

- Works with WAV/FLAC/OGG and other formats supported by libsndfile via `soundfile`.
- With ffmpeg installed, also handles AC-3/E-AC-3, DTS, and audio in common containers (MP4/MKV/M4A, etc.).
- Uses resampling and mono downmix only for correlation; written outputs preserve original sample rate and channels.
