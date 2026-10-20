import json
import os
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from src.dataset import load_radioml_data
from src.models.conv1d_amc import Conv1DAMCNet

def evaluate():
    with open("configs/radioml.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    torch.set_num_threads(cfg["training"].get("cpu_threads", 6))
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    _, test_ds, mods, snrs = load_radioml_data(cfg["data"]["raw_path"], train_ratio=cfg["data"]["train_split"], seed=cfg["training"]["seed"], in_channels=cfg["model"]["in_channels"])
    test_loader = DataLoader(test_ds, batch_size=cfg["data"]["batch_size"], shuffle=False)
    model = Conv1DAMCNet(in_channels=cfg["model"]["in_channels"], num_classes=len(mods), dropout=cfg["model"]["dropout"]).to(device)
    ckpt_path = os.path.join(cfg["training"]["checkpoint_dir"], "best_model.pth")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    if checkpoint["config"] != cfg or checkpoint["mods"] != mods:
        raise ValueError("Checkpoint/config mismatch; evaluate with the training configuration")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    preds_all, labels_all, snrs_all = [], [], []
    with torch.no_grad():
        for x, y, s in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(y.numpy())
            snrs_all.extend(s.numpy())
    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)
    snrs_all = np.array(snrs_all)
    os.makedirs("results", exist_ok=True)
    snr_acc = [np.mean(preds_all[snrs_all == snr] == labels_all[snrs_all == snr]) for snr in snrs]
    metrics = {
        "dataset": "RadioML2016.10a", "checkpoint_epoch": checkpoint["epoch"],
        "split_seed": cfg["training"]["seed"], "evaluation_samples": len(labels_all),
        "evaluation_role": "validation (also used for checkpoint selection; not independent test)",
        "overall_accuracy": float(np.mean(preds_all == labels_all)),
        "high_snr_accuracy": float(np.mean(preds_all[snrs_all >= 0] == labels_all[snrs_all >= 0])),
        "accuracy_by_snr": {str(snr): float(acc) for snr, acc in zip(snrs, snr_acc)},
    }
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed("results/predictions.npz", predictions=preds_all, labels=labels_all, snrs=snrs_all, mods=np.array(mods))
    print(json.dumps(metrics, indent=2))
    plt.figure(figsize=(9, 5))
    plt.plot(snrs, snr_acc, marker="o", linewidth=2)
    plt.title("RadioML2016.10a - Validation Accuracy by SNR")
    plt.xlabel("Signal-to-Noise Ratio (SNR dB)")
    plt.ylabel("Classification Accuracy")
    plt.ylim(0, 1)
    plt.axhline(1 / len(mods), color="gray", linestyle=":", label="Chance (1/11)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/accuracy_vs_snr.png", dpi=300)
    plt.close()
    high_snr = max(snrs)
    mask = (snrs_all == high_snr)
    cm = confusion_matrix(labels_all[mask], preds_all[mask], labels=np.arange(len(mods)), normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=mods, yticklabels=mods, cmap="Blues")
    plt.title(f"Validation Confusion Matrix @ +{high_snr} dB SNR")
    plt.xlabel("Predicted Modulation")
    plt.ylabel("True Modulation")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=300)
    plt.close()
    print("[TAMAMLANDI] Grafikler results/ klasorune kaydedildi!")

if __name__ == "__main__":
    evaluate()