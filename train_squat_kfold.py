"""
train_squat_kfold.py — K-FOLD training for squat classifier (binary: golden vs other)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from feedback6class.data import (
    SquatSequenceDataset,
    load_from_folder,
    SequenceSample,
)
from feedback6class.train_utils import set_seed, train_model


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--folder", type=Path,
                   default=Path("model_ready_reps-20260429T143559Z-3-001/model_ready_reps/"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/squat_kfold"))
    p.add_argument("--k", type=int, default=5)

    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)

    p.add_argument("--num-joints", type=int, default=7)
    p.add_argument("--dims", type=int, default=3)

    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ─────────────────────────────────────────────
# Label mapping (BINARY)
# ─────────────────────────────────────────────

def build_labels(samples):
    new_samples = [
        SequenceSample(
            path=s.path,
            label=0 if s.label == 0 else 1,
            split=s.split
        )
        for s in samples
    ]

    idx_to_label = {
        0: "good_form",
        1: "other",
    }

    return new_samples, idx_to_label


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading: {args.folder}")

    samples = load_from_folder(args.folder)

    if len(samples) == 0:
        raise ValueError("No data found. Check your .npy/.csv folder path.")

    samples, idx_to_label = build_labels(samples)

    print(f"Total samples: {len(samples)}")
    print(f"Classes: {idx_to_label}")

    kf = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    all_acc = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(samples)):
        print(f"\n===== FOLD {fold+1}/{args.k} =====")

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        train_ds = SquatSequenceDataset(
            train_samples,
            num_joints=args.num_joints,
            dims=args.dims,
            augment=True,
        )

        val_ds = SquatSequenceDataset(
            val_samples,
            num_joints=args.num_joints,
            dims=args.dims,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fold_dir = args.output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        best_model, history = train_model(
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=fold_dir,
            num_classes=len(idx_to_label),
            idx_to_label=idx_to_label,
            input_dims=args.dims,
            num_joints=args.num_joints,
            num_angles=5,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

        best_acc = max(h.val_acc for h in history)
        all_acc.append(best_acc)

        print(f"Fold {fold+1} best acc: {best_acc:.4f}")

        # ── Plot curves for this fold ─────────────────────
        epochs = [h.epoch for h in history]
        train_loss = [h.train_loss for h in history]
        val_loss = [h.val_loss for h in history]
        train_acc = [h.train_acc for h in history]
        val_acc = [h.val_acc for h in history]

        # Loss plot
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve (Fold {fold+1})")
        plt.legend()
        plt.grid(True)
        plt.savefig(fold_dir / "loss_curve.png")
        plt.close()

        # Accuracy plot
        plt.figure()
        plt.plot(epochs, train_acc, label="Train Accuracy")
        plt.plot(epochs, val_acc, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Curve (Fold {fold+1})")
        plt.legend()
        plt.grid(True)
        plt.savefig(fold_dir / "accuracy_curve.png")
        plt.close()

    print("\n====================")
    print(f"Mean K-Fold Accuracy: {np.mean(all_acc):.4f}")
    print(f"Std Dev: {np.std(all_acc):.4f}")


if __name__ == "__main__":
    main()