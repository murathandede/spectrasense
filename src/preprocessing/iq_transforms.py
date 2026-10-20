import numpy as np
from scipy import signal

def normalize_iq(iq_data: np.ndarray) -> np.ndarray:
    """Scale both quadratures by a shared RMS, preserving constellation geometry."""
    iq_data = np.asarray(iq_data, dtype=np.float32)
    if iq_data.shape[-2] != 2:
        raise ValueError("Expected (..., 2, time) I/Q input")
    rms = np.sqrt(np.mean(np.sum(iq_data ** 2, axis=-2, keepdims=True), axis=-1, keepdims=True))
    return iq_data / np.maximum(rms, 1e-8)


def enrich_iq(iq_data: np.ndarray) -> np.ndarray:
    """Append instantaneous amplitude and phase in radians to I/Q."""
    i, q = iq_data[..., 0:1, :], iq_data[..., 1:2, :]
    return np.concatenate((iq_data, np.hypot(i, q), np.arctan2(q, i)), axis=-2).astype(np.float32)


def iq_to_spectrogram(iq_sample: np.ndarray, nperseg: int = 32, noverlap: int = 16) -> np.ndarray:
    complex_sig = iq_sample[0] + 1j * iq_sample[1]
    _, _, Zxx = signal.stft(complex_sig, nperseg=nperseg, noverlap=noverlap, return_onesided=False)
    spec = np.log10(np.abs(Zxx) ** 2 + 1e-8)
    return spec.astype(np.float32)