#!/usr/bin/env python3
import numpy as np
from sync_audio import estimate_offset, resample_if_needed
from scipy import signal

# Generate synthetic test: 2 kHz tone burst + noise, with known delay
sr = 16000
sec = 3.0
n = int(sr*sec)

t = np.arange(n)/sr
sig = (np.sin(2*np.pi*2000*t) * (signal.windows.tukey(n,0.1))).astype(np.float32)
# Add some pink-ish noise
np.random.seed(0)
noise = signal.lfilter([1],[1,-0.97], np.random.randn(n).astype(np.float32))*0.05
ref = sig + noise

# Create delayed version by 750 ms
shift_ms = 750
shift_samp = int(sr*shift_ms/1000)
delayed = np.concatenate([np.zeros(shift_samp, dtype=np.float32), ref])

# Correlation domain
ref_corr, sr_corr = resample_if_needed(ref.copy(), sr, 16000)
del_corr, _ = resample_if_needed(delayed.copy(), sr, 16000)

# Estimate
lag = estimate_offset(ref_corr, del_corr, sr_corr, method='hilbert', max_shift_ms=2000)
measured_ms = 1000*lag/sr_corr
print('expected_delay_ms=+', shift_ms, 'measured_ms=', measured_ms)

# Our convention: delayed target => negative lag
assert abs(measured_ms + shift_ms) < 10, 'Offset estimation error too large'
print('Self-test passed')
