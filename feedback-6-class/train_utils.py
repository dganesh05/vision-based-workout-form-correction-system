"""
train_utils.py — Training loop, evaluation, and feedback utilities.
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

from .data import LABEL_SCHEMA, SquatSequenceDataset, collate_padded_batch
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


# ── Feedback generation ────────────────────────────────────────────────────────

def get_feedback(
    logits: torch.Tensor,
    idx_to_label: dict[int, str],
    top_k: int = 2,
) -> list[dict]:
    """
    Generate structured form feedback from classifier logits.

    Returns a list (sorted by confidence) of dicts:
        {
          "rank"       : int,
          "label_idx"  : int,
          "label_name" : str,
          "display"    : str,
          "confidence" : float (0-1),
          "feedback"   : str,
          "cues"       : list[str],
        }
    """
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    top_probs, top_idxs = probs.topk(min(top_k, len(probs)))

    results = []
    for rank, (prob, idx) in enumerate(zip(top_probs.tolist(), top_idxs.tolist()), start=1):
        schema = LABEL_SCHEMA.get(idx, {
            "name": idx_to_label.get(idx, f"class_{idx}"),
            "display": idx_to_label.get(idx, f"Class {idx}"),
            "feedback": "No feedback available.",
            "cues": [],
        })
        results.append({
            "rank": rank,
            "label_idx": idx,
            "label_name": schema["name"],
            "display": schema["display"],
            "confidence": round(prob, 4),
            "feedback": schema["feedback"],
            "cues": schema["cues"],
        })
    return results


def format_feedback_text(feedback_list: list[dict]) -> str:
    """Pretty-print feedback for CLI or logging."""
    lines = ["─" * 55, "  SQUAT FORM ANALYSIS", "─" * 55]
    for item in feedback_list:
        pct = f"{item['confidence'] * 100:.1f}%"
        lines.append(f"\n#{item['rank']}  {item['display']}  ({pct} confidence)")
        lines.append(f"   {item['feedback']}")
        if item["cues"]:
            for cue in item["cues"]:
                lines.append(f"   • {cue}")
    lines.append("─" * 55)
    return "\n".join(lines)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(
    model: BiCGRUClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, dict[int, float]]:
    """
    Returns (avg_loss, avg_acc, per_class_acc).
    per_class_acc maps label_idx → accuracy on that class.
    """
    model.eval()
    loss_sum = 0.0
    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for joints, angles, lengths, labels, mask in dataloader:
            joints = joints.to(device)
            angles = angles.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            logits, _ = model(joints=joints, angles=angles, lengths=lengths, mask=mask)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)
            for p, t in zip(preds.tolist(), labels.tolist()):
                class_total[t] = class_total.get(t, 0) + 1
                if p == t:
                    class_correct[t] = class_correct.get(t, 0) + 1

    total = sum(class_total.values())
    correct = sum(class_correct.values())
    avg_loss = loss_sum / max(total, 1)
    avg_acc = correct / max(total, 1)
    per_class = {
        i: class_correct[i] / max(class_total[i], 1)
        for i in range(num_classes)
    }
    return avg_loss, avg_acc, per_class


# ── Training ───────────────────────────────────────────────────────────────────

def _class_weights(samples, num_classes: int, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights to handle imbalanced datasets."""
    counts = torch.zeros(num_classes)
    for s in samples:
        counts[s.label] += 1
    counts = counts.clamp(min=1)
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


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
    epochs: int = 200,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    gru_hidden_size: int = 128,
    spatial_channels: int = 64,
    angle_channels: int = 32,
    gru_layers: int = 3,
    dropout: float = 0.3,
    num_workers: int = 0,
    device: torch.device | None = None,
    patience: int = 30,
) -> tuple[Path, list[EpochStats]]:
    """
    Train the BiCGRUClassifier with early stopping.

    Uses:
      - Inverse-frequency class weighting (CrossEntropyLoss)
      - ReduceLROnPlateau scheduler
      - Early stopping on val_acc with `patience` epochs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Weighted loss for class imbalance
    weights = _class_weights(train_dataset.samples, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, verbose=True
    )

    best_val_acc = -1.0
    best_path = output_dir / "best_model.pt"
    history: list[EpochStats] = []
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for joints, angles, lengths, labels, mask in train_loader:
            joints = joints.to(device)
            angles = angles.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            logits, _ = model(joints=joints, angles=angles, lengths=lengths, mask=mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(1) == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        val_loss, val_acc, per_class_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )
        scheduler.step(val_acc)

        per_class_named = {
            idx_to_label.get(i, str(i)): round(acc, 4)
            for i, acc in per_class_acc.items()
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
            f"Epoch {epoch:04d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  |  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )
        if epoch % 10 == 0:
            for cls, acc in per_class_named.items():
                print(f"   {cls:<22}: {acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
            break

    history_path = output_dir / "history.json"
    history_path.write_text(
        json.dumps([asdict(s) for s in history], indent=2), encoding="utf-8"
    )
    print(f"\nBest val_acc: {best_val_acc:.4f} — saved to {best_path}")
    return best_path, history
