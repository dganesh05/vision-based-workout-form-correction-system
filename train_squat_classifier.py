from __future__ import annotations


import argparse
import json
import random
from pathlib import Path

import torch

from squat_classifier.data import SquatSequenceDataset, load_manifest, load_from_folder
from squat_classifier.train_utils import set_seed, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a squat classifier from 3D joint sequences using "
            "CNN + stacked Bi-GRU + Luong dot attention."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path, help="CSV with path,label[,split]")
    group.add_argument("--folder", type=Path, help="Folder containing .npy files (correct__/incorrect__ prefix)")
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Base folder for relative paths")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/squat_classifier"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-joints", type=int, default=17)
    parser.add_argument("--dims", type=int, default=3, help="Input dims per joint (use 3 for x,y,z)")
    parser.add_argument("--person-id", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no downsampling")
    parser.add_argument("--spatial-channels", type=int, default=64)
    parser.add_argument("--gru-hidden", type=int, default=128)
    parser.add_argument("--gru-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    return parser.parse_args()


def split_samples(samples: list, seed: int, val_ratio: float) -> tuple[list, list]:
    explicit_train = [s for s in samples if s.split == "train"]
    explicit_val = [s for s in samples if s.split == "val"]
    explicit_test = [s for s in samples if s.split == "test"]

    if explicit_train and explicit_val:
        print(f"Using explicit split from manifest: train={len(explicit_train)} val={len(explicit_val)}")
        if explicit_test:
            print(f"Ignoring test rows during training: {len(explicit_test)}")
        return explicit_train, explicit_val

    shuffled = samples[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    val_samples = shuffled[:val_size]
    train_samples = shuffled[val_size:]

    if not train_samples:
        raise ValueError("Not enough samples after split. Add more rows or reduce val_ratio.")

    print(f"Using random split: train={len(train_samples)} val={len(val_samples)}")
    return train_samples, val_samples



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.folder is not None:
        samples = load_from_folder(args.folder)
    else:
        samples = load_manifest(manifest_path=args.manifest, data_root=args.data_root)

    labels = sorted({sample.label for sample in samples})
    if len(labels) < 2:
        raise ValueError("Need at least two classes in manifest labels (e.g., good and bad).")
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    train_samples, val_samples = split_samples(samples=samples, seed=args.seed, val_ratio=args.val_ratio)

    max_frames = args.max_frames if args.max_frames > 0 else None
    train_dataset = SquatSequenceDataset(
        train_samples,
        label_to_idx=label_to_idx,
        num_joints=args.num_joints,
        dims=args.dims,
        person_id=args.person_id,
        min_confidence=args.min_confidence,
        max_frames=max_frames,
    )
    val_dataset = SquatSequenceDataset(
        val_samples,
        label_to_idx=label_to_idx,
        num_joints=args.num_joints,
        dims=args.dims,
        person_id=args.person_id,
        min_confidence=args.min_confidence,
        max_frames=max_frames,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    print(f"Classes: {label_to_idx}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path, history = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_classes=len(labels),
        input_dims=args.dims,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gru_hidden_size=args.gru_hidden,
        spatial_channels=args.spatial_channels,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        num_workers=args.num_workers,
        device=device,
    )

    metadata = {
        "label_to_idx": label_to_idx,
        "idx_to_label": {idx: label for label, idx in label_to_idx.items()},
        "num_joints": args.num_joints,
        "dims": args.dims,
        "person_id": args.person_id,
        "min_confidence": args.min_confidence,
        "spatial_channels": args.spatial_channels,
        "gru_hidden": args.gru_hidden,
        "gru_layers": args.gru_layers,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "lr": args.lr,
        "best_model_path": str(best_model_path),
        "best_val_acc": max((row.val_acc for row in history), default=None),
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved best model: {best_model_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
