"""
data.py — Dataset and loading utilities for BINARY squat quality classification.

Task:
    0 → golden_form (correct squat)
    1 → not_golden (ANY form deviation)

All original fault types are collapsed into "not_golden".
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

# ─────────────────────────────────────────────
# Binary label schema (for UI / explanation only)
# ─────────────────────────────────────────────

BINARY_LABEL_SCHEMA: dict[int, dict] = {
    0: {
        "name": "golden_form",
        "display": "Golden Form",
        "feedback": "Excellent squat form.",
        "cues": [],
    },
    1: {
        "name": "not_golden",
        "display": "Form Deviation",
        "feedback": "Some movement errors detected. Review squat mechanics.",
        "cues": [
            "Check knee alignment",
            "Maintain upright chest",
            "Ensure full depth",
            "Keep heels grounded",
            "Maintain symmetry",
        ],
    },
}

# Angle column order expected in CSV files
CSV_ANGLE_COLUMNS = [
    "Right_Knee_Angle",
    "Left_Knee_Angle",
    "Right_Hip_Angle",
    "Left_Hip_Angle",
    "Spine_Lean_Angle",
]

# ─────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class SequenceSample:
    path: Path
    label: int  # 0 = golden, 1 = not_golden
    split: str | None = None


# ─────────────────────────────────────────────
# Label parsing (CRITICAL: binary collapse here)
# ─────────────────────────────────────────────

_LABEL_RE = re.compile(r"label_(\d+)")

def label_from_filename(fname: str) -> int:
    """
    Convert multi-class labels → binary gate:

        0 → golden_form
        1–5 → not_golden
    """
    m = _LABEL_RE.search(fname)
    if not m:
        return 1  # unknown = unsafe → treat as not golden

    raw = int(m.group(1))

    return 0 if raw == 0 else 1


# ─────────────────────────────────────────────
# File loading
# ─────────────────────────────────────────────

def _load_npy(path: Path, num_joints: int, dims: int) -> np.ndarray:
    arr = np.load(path).astype(np.float32, copy=False)

    if arr.ndim == 2:
        arr = arr[:, :, None]

    if arr.shape[1] != num_joints:
        raise ValueError(f"{path}: expected {num_joints} joints, got {arr.shape[1]}")

    if arr.shape[2] < dims:
        pad = ((0, 0), (0, 0), (0, dims - arr.shape[2]))
        arr = np.pad(arr, pad, mode="constant")

    return arr[:, :, :dims]


def _load_csv_angles(path: Path) -> np.ndarray:
    rows = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        missing = [c for c in CSV_ANGLE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        for row in reader:
            rows.append([float(row[c]) for c in CSV_ANGLE_COLUMNS])

    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    return np.array(rows, dtype=np.float32)


def load_sequence(
    path: Path,
    num_joints: int = 7,
    dims: int = 3,
) -> tuple[np.ndarray, np.ndarray | None]:

    if path.suffix.lower() == ".npy":
        joints = _load_npy(path, num_joints, dims)

        csv_path = path.with_suffix(".csv")
        angles = _load_csv_angles(csv_path) if csv_path.exists() else None

        return joints, angles

    if path.suffix.lower() == ".csv":
        angles = _load_csv_angles(path)
        T = angles.shape[0]
        joints = np.zeros((T, num_joints, dims), dtype=np.float32)
        return joints, angles

    raise ValueError(f"Unsupported file: {path}")


# ─────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────

def normalize_joints(seq: np.ndarray) -> np.ndarray:
    if seq.shape[0] == 0:
        return seq

    J = seq.shape[1]

    if J > 4:
        pelvis = (seq[:, 3] + seq[:, 4]) * 0.5
    else:
        pelvis = seq.mean(axis=1)

    centered = seq - pelvis[:, None, :]

    if J > 5:
        shoulder = seq[:, 5]
        scale = np.linalg.norm(shoulder - pelvis, axis=1)
    else:
        scale = np.linalg.norm(centered.reshape(centered.shape[0], -1), axis=1)

    scale = float(np.clip(np.median(scale), 1e-4, None))
    return centered / scale


def normalize_angles(angles: np.ndarray) -> np.ndarray:
    mean = angles.mean(axis=0, keepdims=True)
    std = angles.std(axis=0, keepdims=True).clip(min=1e-4)
    return (angles - mean) / std


# ─────────────────────────────────────────────
# Folder loading
# ─────────────────────────────────────────────

def load_from_folder(folder: Path) -> list[SequenceSample]:
    samples = []

    for p in sorted(folder.glob("*.npy")):
        samples.append(SequenceSample(p, label_from_filename(p.name)))

    for p in sorted(folder.glob("*.csv")):
        if p.with_suffix(".npy").exists():
            continue
        samples.append(SequenceSample(p, label_from_filename(p.name)))

    if not samples:
        raise ValueError(f"No data in {folder}")

    return samples


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class SquatSequenceDataset(Dataset):
    def __init__(
        self,
        samples: Iterable[SequenceSample],
        *,
        num_joints: int = 7,
        dims: int = 3,
        max_frames: int | None = None,
        augment: bool = False,
    ):
        self.samples = list(samples)
        self.num_joints = num_joints
        self.dims = dims
        self.max_frames = max_frames
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        joints, angles = load_sequence(
            s.path,
            num_joints=self.num_joints,
            dims=self.dims,
        )

        joints = normalize_joints(joints)

        if angles is None:
            angles = np.zeros((joints.shape[0], len(CSV_ANGLE_COLUMNS)), dtype=np.float32)
        else:
            angles = normalize_angles(angles)
            T = min(len(joints), len(angles))
            joints, angles = joints[:T], angles[:T]

        if self.max_frames and len(joints) > self.max_frames:
            idxs = np.linspace(0, len(joints) - 1, self.max_frames).astype(int)
            joints, angles = joints[idxs], angles[idxs]

        return (
            torch.from_numpy(joints).float(),
            torch.from_numpy(angles).float(),
            s.label,
        )


# ─────────────────────────────────────────────
# Collation
# ─────────────────────────────────────────────

def collate_padded_batch(batch):
    joints, angles, labels = zip(*batch)

    lengths = torch.tensor([x.shape[0] for x in joints])
    labels = torch.tensor(labels)

    T_max = max(lengths).item()
    J, C = joints[0].shape[1], joints[0].shape[2]
    A = angles[0].shape[1]

    B = len(batch)

    j_pad = torch.zeros(B, T_max, J, C)
    a_pad = torch.zeros(B, T_max, A)
    mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i, (j, a) in enumerate(zip(joints, angles)):
        t = j.shape[0]
        j_pad[i, :t] = j
        a_pad[i, :t] = a
        mask[i, :t] = True

    return j_pad, a_pad, lengths, labels, mask