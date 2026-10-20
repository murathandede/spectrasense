import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.preprocessing.iq_transforms import normalize_iq, enrich_iq

MODULATIONS = sorted(["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "CPFSK", "GFSK", "PAM4", "WBFM", "AM-DSB", "AM-SSB"])

class RadioMLDataset(Dataset):
    def __init__(self, samples, labels, snrs, normalize=True, in_channels=2):
        if in_channels not in (2, 4):
            raise ValueError("in_channels must be 2 or 4")
        if normalize:
            samples = normalize_iq(samples)
        if in_channels == 4:
            samples = enrich_iq(samples)
        self.samples = torch.from_numpy(np.asarray(samples, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.snrs = torch.from_numpy(np.asarray(snrs, dtype=np.int32))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx], self.snrs[idx]


def load_radioml_data(pkl_path, train_ratio=0.8, seed=42, in_channels=2):
    path = Path(pkl_path)
    if not path.is_file():
        raise FileNotFoundError(f"Extract the real RadioML dataset first: {path}")
    if path.stat().st_size < 50_000_000:
        raise ValueError("RadioML pickle is smaller than 50 MB; real data required")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must lie between 0 and 1")
    # Only load the trusted, user-provided RadioML archive (pickle executes code).
    with path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")
    mods, snrs = MODULATIONS, list(range(-20, 20, 2))
    expected = {(mod, snr) for mod in mods for snr in snrs}
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError("Expected all 220 RadioML2016.10a modulation/SNR groups")
    rng = np.random.default_rng(seed)
    partitions = [([], [], []), ([], [], [])]
    for mod_idx, mod in enumerate(mods):
        for snr in snrs:
            arr = np.asarray(data[(mod, snr)], dtype=np.float32)
            if arr.shape != (1000, 2, 128) or not np.isfinite(arr).all():
                raise ValueError(f"Invalid RadioML group {(mod, snr)}: {arr.shape}")
            indices = rng.permutation(len(arr))
            split = int(len(arr) * train_ratio)
            for parts, idx in zip(partitions, (indices[:split], indices[split:])):
                parts[0].append(arr[idx])
                parts[1].append(np.full(len(idx), mod_idx, dtype=np.int64))
                parts[2].append(np.full(len(idx), snr, dtype=np.int32))
    datasets = [RadioMLDataset(*(np.concatenate(v) for v in parts), in_channels=in_channels)
                for parts in partitions]
    return *datasets, mods, snrs
