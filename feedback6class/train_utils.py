"""
train_utils.py — Training loop, evaluation, and feedback utilities (FIXED)
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import SquatSequenceDataset, collate_padded_batch
from .model import BiCGRUClassifier


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Epoch stats ────────────────────────────────────────────────────────────────

@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    per_class_val_acc: dict[str, float]


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for joints, angles, lengths, labels, mask in dataloader:
            joints, angles = joints.to(device), angles.to(device)
            lengths, labels = lengths.to(device), labels.to(device)
            mask = mask.to(device)

            logits, _ = model(joints=joints, angles=angles, lengths=lengths, mask=mask)
            loss = criterion(logits, labels)

            loss_sum += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for p, t in zip(preds.tolist(), labels.tolist()):
                class_total[t] += 1
                if p == t:
                    class_correct[t] += 1

    avg_loss = loss_sum / max(total, 1)
    avg_acc = correct / max(total, 1)

    per_class = {
        i: class_correct[i] / max(class_total[i], 1)
        for i in range(num_classes)
    }

    return avg_loss, avg_acc, per_class


# ── Class weights ──────────────────────────────────────────────────────────────

def _class_weights(samples, num_classes, device):
    counts = torch.zeros(num_classes)

    for s in samples:
        counts[s.label] += 1

    counts = counts.clamp(min=1)
    weights = counts.sum() / (num_classes * counts)

    return weights.to(device)


# ── MAIN TRAIN LOOP ────────────────────────────────────────────────────────────

def train_model(
    *,
    train_dataset: SquatSequenceDataset,
    val_dataset: SquatSequenceDataset,
    output_dir: Path,

    num_classes: int,
    idx_to_label: dict[int, str],

    input_dims: int = 3,
    num_joints: int = 7,
    num_angles: int = 5,

    epochs: int = 100,
    batch_size: int = 8,

    lr: float = 5e-4,
    weight_decay: float = 1e-4,

    gru_hidden_size: int = 16,
    spatial_channels: int = 32,
    angle_channels: int = 16,
    gru_layers: int = 1,
    dropout: float = 0.3,

    num_workers: int = 0,
    device: torch.device | None = None,

    patience: int = 6,
) -> tuple[Path, list[EpochStats]]:

    output_dir.mkdir(parents=True, exist_ok=True)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── loaders ───────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_padded_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_padded_batch,
    )

    # ── MODEL (FIXED SIGNATURE) ───────────────────────────
    model = BiCGRUClassifier(
        input_dims=input_dims,
        num_joints=num_joints,
        num_angles=num_angles,
        spatial_channels=spatial_channels,
        angle_channels=angle_channels,
        gru_hidden_size=gru_hidden_size,
        gru_layers=gru_layers,
        dropout=dropout,
        num_classes=num_classes,
    ).to(device)

    # ── LOSS ──────────────────────────────────────────────
    weights = _class_weights(train_dataset.samples, num_classes, device)

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.1
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    # ── TRAINING STATE ────────────────────────────────────
    best_val_acc = 0.0
    best_path = output_dir / "best_model.pt"
    history: list[EpochStats] = []
    no_improve = 0

    # ── TRAIN LOOP ────────────────────────────────────────
    for epoch in range(1, epochs + 1):

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for joints, angles, lengths, labels, mask in train_loader:
            joints, angles = joints.to(device), angles.to(device)
            lengths, labels = lengths.to(device), labels.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            logits, _ = model(joints, angles, lengths, mask)
            loss = criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        val_loss, val_acc, per_class = evaluate(
            model, val_loader, criterion, device, num_classes
        )

        scheduler.step(val_acc)

        # ── logging ───────────────────────────────────────
        per_class_named = {
            idx_to_label[i]: round(v, 4)
            for i, v in per_class.items()
        }

        stats = EpochStats(
            epoch=epoch,
            train_loss=round(train_loss, 5),
            train_acc=round(train_acc, 4),
            val_loss=round(val_loss, 5),
            val_acc=round(val_acc, 4),
            per_class_val_acc=per_class_named,
        )

        history.append(stats)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f}"
        )

        # ── SAVE BEST MODEL ──────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "model_state": model.state_dict(),
                "config": {
                    "input_dims": input_dims,
                    "num_joints": num_joints,
                    "num_angles": num_angles,
                    "spatial_channels": spatial_channels,
                    "angle_channels": angle_channels,
                    "gru_hidden_size": gru_hidden_size,
                    "gru_layers": gru_layers,
                    "dropout": dropout,
                    "num_classes": num_classes,
                },
                "idx_to_label": idx_to_label
            }

            torch.save(checkpoint, best_path)
            no_improve = 0
        else:
            no_improve += 1

        # ── EARLY STOP ───────────────────────────────────
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # ── SAVE HISTORY ──────────────────────────────────────
    (output_dir / "history.json").write_text(
        json.dumps([asdict(h) for h in history], indent=2)
    )

    print(f"\nBest val acc: {best_val_acc:.4f}")
    print(f"Saved model: {best_path}")

    return best_path, history