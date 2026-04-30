"""
train_squat_classifier.py
─────────────────────────
Train a multi-class squat form classifier.

Label schema (inferred from filename):
  label_0 → good_form
  label_1 → knee_cave
  label_2 → forward_lean
  label_3 → shallow_depth
  label_4 → heel_rise
  label_5 → asymmetric_stance

Usage examples
──────────────
# From a folder of .npy (and optional companion .csv) files:
python train_squat_classifier.py --folder path/to/reps/

# With explicit settings:
python train_squat_classifier.py \\
    --folder model_ready_reps/model_ready_reps/ \\
    --epochs 300 \\
    --batch-size 8 \\
    --lr 5e-4 \\
    --num-joints 7 \\
    --output-dir runs/squat_v2

Notes
─────
- If the folder only contains label_0 and label_1 files the model will
  automatically fall back to 2-class (good / poor) mode.
- Weighted CrossEntropyLoss handles class imbalance automatically.
- Early stopping is enabled by default (--patience 30).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from squat_classifier.data import (
    LABEL_SCHEMA,
    SquatSequenceDataset,
    discover_label_set,
    load_from_folder,
)
from squat_classifier.train_utils import set_seed, train_model


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a multi-class squat form classifier (CNN + Bi-GRU + Attention)."
    )
    p.add_argument(
        "--folder", type=Path,
        default=Path("model_ready_reps/model_ready_reps/"),
        help="Folder containing .npy (and optional companion .csv) files.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("runs/squat_classifier"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--num-joints", type=int, default=7)
    p.add_argument("--dims", type=int, default=3, help="Spatial dims per joint (3 = x,y,z).")
    p.add_argument("--max-frames", type=int, default=0, help="0 = no downsampling.")
    p.add_argument("--spatial-channels", type=int, default=64)
    p.add_argument("--angle-channels", type=int, default=32)
    p.add_argument("--gru-hidden", type=int, default=128)
    p.add_argument("--gru-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=30, help="Early stopping patience (epochs).")
    p.add_argument("--augment", action="store_true", help="Enable left/right mirror augmentation.")
    return p.parse_args()


# ── Train/val split ────────────────────────────────────────────────────────────

def split_samples(samples, seed: int, val_ratio: float):
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    # Stratified split: preserve per-class ratios
    from collections import defaultdict
    by_class = defaultdict(list)
    for s in shuffled:
        by_class[s.label].append(s)

    train_samples, val_samples = [], []
    for cls_samples in by_class.values():
        n_val = max(1, int(len(cls_samples) * val_ratio))
        val_samples.extend(cls_samples[:n_val])
        train_samples.extend(cls_samples[n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    folder = args.folder
    print(f"Loading data from: {folder.resolve()}")

    all_samples = load_from_folder(folder)
    print(f"Found {len(all_samples)} total samples.")

    # Discover which label indices actually exist in the data
    found_labels = sorted(discover_label_set(folder))
    print(f"Label indices found in data: {found_labels}")

    # Build label maps — use the LABEL_SCHEMA names when available, else fallback
    if set(found_labels) <= {0, 1}:
        # Binary fallback
        idx_to_name = {0: "good_form", 1: "poor_form"}
        print("Only binary labels found — training in 2-class mode.")
    else:
        idx_to_name = {
            i: LABEL_SCHEMA[i]["name"] if i in LABEL_SCHEMA else f"class_{i}"
            for i in found_labels
        }

    # Remap sample labels so they are contiguous (0, 1, 2, …)
    # This is a no-op when labels are already {0,1,...,N-1} but safe otherwise.
    sorted_label_indices = sorted(idx_to_name.keys())
    remap = {old: new for new, old in enumerate(sorted_label_indices)}
    idx_to_label = {new: idx_to_name[old] for old, new in remap.items()}
    label_to_idx = {v: k for k, v in idx_to_label.items()}
    num_classes = len(idx_to_label)

    from squat_classifier.data import SequenceSample
    remapped_samples = [
        SequenceSample(path=s.path, label=remap[s.label], split=s.split)
        for s in all_samples
        if s.label in remap
    ]

    print(f"Classes ({num_classes}): {idx_to_label}")

    train_samples, val_samples = split_samples(remapped_samples, args.seed, args.val_ratio)
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    # Class distribution
    from collections import Counter
    train_dist = Counter(s.label for s in train_samples)
    val_dist = Counter(s.label for s in val_samples)
    for i in range(num_classes):
        print(f"  {idx_to_label[i]:<25} train={train_dist[i]}  val={val_dist[i]}")

    max_frames = args.max_frames if args.max_frames > 0 else None

    train_dataset = SquatSequenceDataset(
        train_samples,
        num_joints=args.num_joints,
        dims=args.dims,
        max_frames=max_frames,
        augment=args.augment,
    )
    val_dataset = SquatSequenceDataset(
        val_samples,
        num_joints=args.num_joints,
        dims=args.dims,
        max_frames=max_frames,
        augment=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_model_path, history = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_classes=num_classes,
        idx_to_label=idx_to_label,
        input_dims=args.dims,
        num_joints=args.num_joints,
        num_angles=5,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gru_hidden_size=args.gru_hidden,
        spatial_channels=args.spatial_channels,
        angle_channels=args.angle_channels,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        num_workers=args.num_workers,
        device=device,
        patience=args.patience,
    )

    best_val_acc = max((s.val_acc for s in history), default=None)

    metadata = {
        "num_classes": num_classes,
        "label_to_idx": label_to_idx,
        "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
        "label_schema": {
            str(k): {
                "name": LABEL_SCHEMA[sorted_label_indices[k]]["name"] if sorted_label_indices[k] in LABEL_SCHEMA else idx_to_label[k],
                "display": LABEL_SCHEMA[sorted_label_indices[k]]["display"] if sorted_label_indices[k] in LABEL_SCHEMA else idx_to_label[k],
                "feedback": LABEL_SCHEMA[sorted_label_indices[k]]["feedback"] if sorted_label_indices[k] in LABEL_SCHEMA else "",
                "cues": LABEL_SCHEMA[sorted_label_indices[k]]["cues"] if sorted_label_indices[k] in LABEL_SCHEMA else [],
            }
            for k in range(num_classes)
        },
        "num_joints": args.num_joints,
        "dims": args.dims,
        "num_angles": 5,
        "spatial_channels": args.spatial_channels,
        "angle_channels": args.angle_channels,
        "gru_hidden": args.gru_hidden,
        "gru_layers": args.gru_layers,
        "dropout": args.dropout,
        "best_model_path": str(best_model_path),
        "best_val_acc": best_val_acc,
    }
    meta_path = args.output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved best model : {best_model_path}")
    print(f"Saved metadata   : {meta_path}")
    print(f"Best val acc     : {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
