import json
import random
from pathlib import Path
import time
import numpy as np
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.dataset import load_radioml_data
from src.models.conv1d_amc import Conv1DAMCNet


def train():
    cfg = yaml.safe_load(Path("configs/radioml.yaml").read_text())
    seed = cfg["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg["training"].get("cpu_threads", 6))
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    train_ds, val_ds, mods, snrs = load_radioml_data(
        cfg["data"]["raw_path"], cfg["data"]["train_split"], seed, cfg["model"]["in_channels"])
    print(f"Real RadioML2016.10a: train={len(train_ds)}, validation={len(val_ds)}", flush=True)
    loader_args = dict(batch_size=cfg["data"]["batch_size"], num_workers=cfg["data"]["num_workers"])
    train_loader = DataLoader(train_ds, shuffle=True, generator=torch.Generator().manual_seed(seed), **loader_args)
    val_loader = DataLoader(val_ds, **loader_args)
    model = Conv1DAMCNet(cfg["model"]["in_channels"], len(mods), cfg["model"]["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    Path("results").mkdir(exist_ok=True)
    history, best_acc = [], -1.0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        start = time.perf_counter()
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        lr = optimizer.param_groups[0]["lr"]
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        model.eval()
        val_correct, val_total, high_correct, high_total = 0, 0, 0, 0
        with torch.inference_mode():
            for x, y, s in val_loader:
                hits = model(x.to(device)).argmax(1).cpu() == y
                high = s >= 0
                val_correct += hits.sum().item()
                val_total += len(y)
                high_correct += hits[high].sum().item()
                high_total += high.sum().item()
        val_acc, high_acc = val_correct / val_total, high_correct / high_total
        scheduler.step()
        row = dict(epoch=epoch, loss=loss_sum/total, train_accuracy=correct/total,
                   overall_val_accuracy=val_acc, high_snr_val_accuracy=high_acc,
                   lr=lr, seconds=time.perf_counter()-start)
        history.append(row)
        print(f"Epoch {epoch:02d}/{cfg['training']['epochs']} | loss={row['loss']:.4f} | "
              f"Overall Val Acc={val_acc:.4%} | High-SNR Val Acc (SNR >= 0 dB)={high_acc:.4%} | "
              f"{row['seconds']:.1f}s", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(dict(model_state=model.state_dict(), config=cfg, mods=mods, snrs=snrs,
                            epoch=epoch, overall_val_accuracy=val_acc, high_snr_val_accuracy=high_acc),
                       checkpoint_dir / "best_model.pth")
        Path("results/training_history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    train()
