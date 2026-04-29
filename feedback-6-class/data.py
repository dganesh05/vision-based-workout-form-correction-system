"""
data.py — Dataset and loading utilities for multi-class squat form classification.

Supports two input modalities:
  - .npy files : (T, J, 3) arrays of normalised 3-D joint positions
  - .csv files : (T, 5) arrays of pre-computed joint angles
    columns expected: Right_Knee, Left_Knee, Right_Hip, Left_Hip, Spine_Lean

Form-fault labels (LABEL_SCHEMA):
  0  good_form          — correct squat
  1  knee_cave          — knees collapse inward (valgus)
  2  forward_lean       — excessive torso lean
  3  shallow_depth      — insufficient squat depth
  4  heel_rise          — heels lifting off the floor
  5  asymmetric_stance  — left/right imbalance

The label is embedded in the filename:
  label_0  → good_form
  label_1  → knee_cave
  label_2  → forward_lean
  label_3  → shallow_depth
  label_4  → heel_rise
  label_5  → asymmetric_stance

If only binary labels (label_0 / label_1) are present the schema collapses to
  {good_form: 0, poor_form: 1}
and the model outputs two classes instead of six.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ── Label schema ──────────────────────────────────────────────────────────────

LABEL_SCHEMA: dict[int, dict] = {
    0: {
        "name": "good_form",
        "display": "Good Form",
        "feedback": "Great squat! Maintain this technique.",
        "cues": [],
    },
    1: {
        "name": "knee_cave",
        "display": "Knee Cave (Valgus)",
        "feedback": "Your knees are collapsing inward. Push your knees out in line with your toes.",
        "cues": [
            "Drive knees outward over pinky toes",
            "Engage glutes throughout the movement",
            "Try a slightly wider stance",
        ],
    },
    2: {
        "name": "forward_lean",
        "display": "Excessive Forward Lean",
        "feedback": "Your torso is leaning too far forward. Keep your chest up and spine neutral.",
        "cues": [
            "Keep chest up, eyes forward",
            "Improve ankle mobility",
            "Widen stance slightly and turn toes out",
        ],
    },
    3: {
        "name": "shallow_depth",
        "display": "Shallow Depth",
        "feedback": "You are not reaching full depth. Aim for hips below parallel.",
        "cues": [
            "Sit deeper into the squat",
            "Work on hip flexor and ankle mobility",
            "Slow the descent to control depth",
        ],
    },
    4: {
        "name": "heel_rise",
        "display": "Heel Rise",
        "feedback": "Your heels are lifting. Keep your full foot flat on the floor.",
        "cues": [
            "Improve ankle dorsiflexion",
            "Try heel-elevated squats as a mobility drill",
            "Drive through the whole foot evenly",
        ],
    },
    5: {
        "name": "asymmetric_stance",
        "display": "Asymmetric Stance",
        "feedback": "You have a significant left/right imbalance. Focus on symmetrical loading.",
        "cues": [
            "Check foot placement for symmetry",
            "Strengthen the weaker side with single-leg work",
            "Film from the front to monitor alignment",
        ],
    },
}

# Angle column order expected in CSV files
CSV_ANGLE_COLUMNS = ["Right_Knee", "Left_Knee", "Right_Hip", "Left_Hip", "Spine_Lean"]


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SequenceSample:
    path: Path
    label: int          # integer class index
    split: str | None = None


# ── File loading ───────────────────────────────────────────────────────────────

def _load_npy(path: Path, num_joints: int, dims: int) -> np.ndarray:
    """Load a (T, J, C) .npy file, padding/trimming channels as needed."""
    arr = np.load(path).astype(np.float32, copy=False)
    if arr.ndim == 2:                    # (T, J) → (T, J, 1)
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Expected (T,J,C) in {path}, got {arr.shape}")
    if arr.shape[1] != num_joints:
        raise ValueError(f"Expected {num_joints} joints in {path}, got {arr.shape[1]}")
    if arr.shape[2] < dims:
        pad = ((0, 0), (0, 0), (0, dims - arr.shape[2]))
        arr = np.pad(arr, pad, mode="constant")
    return arr[:, :, :dims]


def _load_csv_angles(path: Path) -> np.ndarray:
    """
    Load a CSV of per-frame joint angles.
    Returns (T, 5) float32 array in column order: Right_Knee, Left_Knee,
    Right_Hip, Left_Hip, Spine_Lean.
    """
    df_rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        missing = [c for c in CSV_ANGLE_COLUMNS if c not in cols]
        if missing:
            raise ValueError(f"CSV {path} missing columns: {missing}")
        for row in reader:
            df_rows.append([float(row[c]) for c in CSV_ANGLE_COLUMNS])
    if not df_rows:
        raise ValueError(f"Empty CSV: {path}")
    return np.array(df_rows, dtype=np.float32)


def load_sequence(
    path: Path,
    num_joints: int = 7,
    dims: int = 3,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Returns:
        joints  : (T, J, dims) float32  — 3-D joint positions
        angles  : (T, 5) float32 or None — pre-computed angles if a matching
                  CSV exists alongside the .npy, otherwise None.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        joints = _load_npy(path, num_joints=num_joints, dims=dims)
        # Look for a companion CSV with the same stem
        csv_path = path.with_suffix(".csv")
        angles = _load_csv_angles(csv_path) if csv_path.exists() else None
        return joints, angles
    if suffix == ".csv":
        angles = _load_csv_angles(path)
        # No joint data — return zero joints so the model can still run
        T = angles.shape[0]
        joints = np.zeros((T, num_joints, dims), dtype=np.float32)
        return joints, angles
    raise ValueError(f"Unsupported extension: {path}")


def normalize_joints(sequence: np.ndarray) -> np.ndarray:
    """
    Centre around pelvis and scale by torso length.
    Works for any (T, J, C) array regardless of J.
    """
    if sequence.shape[0] == 0:
        return sequence
    J = sequence.shape[1]
    # Pelvis proxy: mean of joints 3 & 4 (indices 3, 4) if available
    if J > 4:
        pelvis = (sequence[:, 3] + sequence[:, 4]) * 0.5
    else:
        pelvis = sequence.mean(axis=1)
    centered = sequence - pelvis[:, None, :]
    # Scale: torso length proxy (joint 5 or 6 toward top vs pelvis)
    if J > 5:
        shoulder_proxy = (sequence[:, 5] + sequence[:, 6]) * 0.5 if J > 6 else sequence[:, 5]
        scale_vals = np.linalg.norm(shoulder_proxy - pelvis, axis=1)
    else:
        flat = centered.reshape(centered.shape[0], -1)
        scale_vals = np.linalg.norm(flat, axis=1)
    scale = float(np.clip(np.median(scale_vals), 1e-4, None))
    return centered / scale


def normalize_angles(angles: np.ndarray) -> np.ndarray:
    """
    Z-score normalise each angle column independently.
    Angles in degrees typically range 60-180; standardising helps the GRU.
    """
    mean = angles.mean(axis=0, keepdims=True)
    std = angles.std(axis=0, keepdims=True).clip(min=1e-4)
    return (angles - mean) / std


# ── Label inference from filename ──────────────────────────────────────────────

_LABEL_RE = re.compile(r"label_(\d+)")


def label_from_filename(fname: str, label_to_idx: dict[str, int] | None = None) -> int:
    """
    Extract the integer label from a filename like:
      golden_reference__G001_side_angles__hole_69__label_1_aug_34.csv
      original_data__Copy_of_IMG_6991__hole_40__label_0.npy
    Falls back to 0 if no match.
    """
    m = _LABEL_RE.search(fname)
    return int(m.group(1)) if m else 0


def discover_label_set(folder: Path) -> set[int]:
    """Return all unique label indices found in filenames under folder."""
    labels: set[int] = set()
    for p in folder.glob("*.npy"):
        labels.add(label_from_filename(p.name))
    for p in folder.glob("*.csv"):
        labels.add(label_from_filename(p.name))
    return labels


# ── Sample loading from folder ─────────────────────────────────────────────────

def load_from_folder(folder: Path) -> list[SequenceSample]:
    """
    Scan folder for .npy (primary) and standalone .csv files and return
    SequenceSample objects with labels inferred from filenames.
    CSV files that have a matching .npy are skipped (they are treated as
    companion angle files and loaded automatically by load_sequence).
    """
    samples: list[SequenceSample] = []
    for npy in sorted(folder.glob("*.npy")):
        label = label_from_filename(npy.name)
        samples.append(SequenceSample(path=npy, label=label))
    # Standalone CSVs (no companion .npy)
    for csv_path in sorted(folder.glob("*.csv")):
        if csv_path.with_suffix(".npy").exists():
            continue  # already loaded as companion
        label = label_from_filename(csv_path.name)
        samples.append(SequenceSample(path=csv_path, label=label))
    if not samples:
        raise ValueError(f"No .npy/.csv samples found in {folder}")
    return samples


# ── Dataset ────────────────────────────────────────────────────────────────────

class SquatSequenceDataset(Dataset):
    """
    Yields (joints_tensor, angles_tensor, label_int) per sample.

    joints_tensor : (T, J, dims)  float32
    angles_tensor : (T, 5)        float32  — zeros when no CSV companion
    label         : int
    """

    def __init__(
        self,
        samples: Iterable[SequenceSample],
        *,
        num_joints: int = 7,
        dims: int = 3,
        max_frames: int | None = None,
        augment: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.num_joints = num_joints
        self.dims = dims
        self.max_frames = max_frames
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        joints, angles = load_sequence(sample.path, num_joints=self.num_joints, dims=self.dims)

        if joints.shape[0] == 0:
            raise ValueError(f"Empty sequence: {sample.path}")

        joints = normalize_joints(joints)

        if angles is None:
            angles = np.zeros((joints.shape[0], len(CSV_ANGLE_COLUMNS)), dtype=np.float32)
        else:
            angles = normalize_angles(angles)
            # Align length
            T = min(joints.shape[0], angles.shape[0])
            joints = joints[:T]
            angles = angles[:T]

        if self.max_frames and joints.shape[0] > self.max_frames:
            idx_sel = np.linspace(0, joints.shape[0] - 1, self.max_frames).astype(np.int64)
            joints = joints[idx_sel]
            angles = angles[idx_sel]

        if self.augment:
            joints, angles = _augment(joints, angles)

        return (
            torch.from_numpy(joints).float(),
            torch.from_numpy(angles).float(),
            sample.label,
        )


def _augment(
    joints: np.ndarray, angles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Lightweight augmentation: mirror left/right and add small Gaussian noise."""
    # Mirror left/right (flip x-axis)
    if np.random.rand() < 0.5:
        joints = joints.copy()
        joints[:, :, 0] *= -1.0
        # Also swap Left/Right angle columns: [RK, LK, RH, LH, Spine]
        angles = angles.copy()
        angles[:, [0, 1]] = angles[:, [1, 0]]   # swap knee angles
        angles[:, [2, 3]] = angles[:, [3, 2]]   # swap hip angles
    # Gaussian noise on joints
    joints = joints + np.random.normal(0, 0.005, joints.shape).astype(np.float32)
    return joints, angles


# ── Collation ──────────────────────────────────────────────────────────────────

def collate_padded_batch(batch):
    """
    Collate variable-length sequences with zero-padding.

    Returns:
        joints  : (B, T_max, J, C)
        angles  : (B, T_max, 5)
        lengths : (B,)
        labels  : (B,)
        mask    : (B, T_max) bool — True where valid
    """
    joints_list, angles_list, labels_list = zip(*batch)
    lengths = torch.tensor([j.shape[0] for j in joints_list], dtype=torch.long)
    labels = torch.tensor(labels_list, dtype=torch.long)
    T_max = int(lengths.max().item())
    J = joints_list[0].shape[1]
    C = joints_list[0].shape[2]
    A = angles_list[0].shape[1]
    B = len(batch)

    joints_padded = torch.zeros(B, T_max, J, C)
    angles_padded = torch.zeros(B, T_max, A)
    mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i, (j, a) in enumerate(zip(joints_list, angles_list)):
        T = j.shape[0]
        joints_padded[i, :T] = j
        angles_padded[i, :T] = a
        mask[i, :T] = True

    return joints_padded, angles_padded, lengths, labels, mask
