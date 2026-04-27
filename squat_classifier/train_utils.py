from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SquatSequenceDataset, collate_padded_batch
from .model import BiCGRUClassifier


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def evaluate(
    model: BiCGRUClassifier,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, lengths, labels, mask in dataloader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            logits, _ = model(sequences=sequences, lengths=lengths, mask=mask)
            loss = criterion(logits, labels)
            loss_sum += float(loss.item()) * labels.size(0)

            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

    avg_loss = loss_sum / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


def train_model(
    *,
    train_dataset: SquatSequenceDataset,
    val_dataset: SquatSequenceDataset,
    output_dir: Path,
    num_classes: int,
    input_dims: int,
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    gru_hidden_size: int = 128,
    spatial_channels: int = 64,
    gru_layers: int = 3,
    dropout: float = 0.3,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> tuple[Path, list[EpochStats]]:
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
        spatial_channels=spatial_channels,
        gru_hidden_size=gru_hidden_size,
        gru_layers=gru_layers,
        dropout=dropout,
        num_classes=num_classes,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_path = output_dir / "best_model.pt"
    history: list[EpochStats] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for sequences, lengths, labels, mask in train_loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            logits, _ = model(sequences=sequences, lengths=lengths, mask=mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            running_correct += int((logits.argmax(dim=1) == labels).sum().item())
            running_total += int(labels.size(0))

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc = evaluate(model=model, dataloader=val_loader, criterion=criterion, device=device)

        stats = EpochStats(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )
        history.append(stats)
        print(
            f"Epoch {epoch:04d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps([asdict(row) for row in history], indent=2), encoding="utf-8")
    return best_path, history

