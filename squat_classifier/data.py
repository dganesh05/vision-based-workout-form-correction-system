from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SequenceSample:
    path: Path
    label: str
    split: str | None = None
    start_frame: int | None = None
    end_frame: int | None = None


def _extract_person(people: list[dict], person_id: int) -> dict | None:
    for person in people:
        if int(person.get("person_id", -1)) == person_id:
            return person
    return None


def _frame_keypoints_to_array(
    frame_payload: dict,
    person_id: int,
    num_joints: int,
    dims: int,
    min_confidence: float,
) -> np.ndarray:
    frame_array = np.zeros((num_joints, dims), dtype=np.float32)

    people = frame_payload.get("people", [])
    if not isinstance(people, list):
        return frame_array

    person = _extract_person(people, person_id=person_id)
    if person is None:
        return frame_array

    keypoints = person.get("keypoints", [])
    if not isinstance(keypoints, list):
        return frame_array

    for joint in keypoints:
        joint_id = int(joint.get("joint_id", -1))
        if joint_id < 0 or joint_id >= num_joints:
            continue

        conf = float(joint.get("confidence", 1.0))
        if conf < min_confidence:
            continue

        x = float(joint.get("x", 0.0))
        y = float(joint.get("y", 0.0))
        if dims >= 3:
            z = float(joint.get("z", 0.0))
            frame_array[joint_id] = np.array([x, y, z], dtype=np.float32)
        else:
            frame_array[joint_id] = np.array([x, y], dtype=np.float32)
    return frame_array


def load_sequence(
    sequence_path: Path,
    *,
    person_id: int = 0,
    num_joints: int = 17,
    dims: int = 3,
    min_confidence: float = 0.0,
) -> np.ndarray:
    suffix = sequence_path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(sequence_path).astype(np.float32, copy=False)
        # Accept both (T, J, C) and (T, J) shapes
        if arr.ndim == 2:
            # (T, J) -> (T, J, 1)
            arr = arr[:, :, None]
        if arr.ndim != 3:
            raise ValueError(f"Expected (T, J, C) or (T, J) in {sequence_path}, found {arr.shape}")
        if arr.shape[1] != num_joints:
            raise ValueError(
                f"Expected {num_joints} joints in {sequence_path}, found {arr.shape[1]}"
            )
        if arr.shape[2] < dims:
            # Pad with zeros to reach dims channels
            pad_width = ((0, 0), (0, 0), (0, dims - arr.shape[2]))
            arr = np.pad(arr, pad_width, mode="constant")
        return arr[:, :, :dims]

    if suffix == ".json":
        with sequence_path.open("r", encoding="utf-8") as f:
            frames = json.load(f)

        if not isinstance(frames, list):
            raise ValueError(f"Expected top-level frame list in {sequence_path}")

        sequence = [
            _frame_keypoints_to_array(
                frame_payload=frame,
                person_id=person_id,
                num_joints=num_joints,
                dims=dims,
                min_confidence=min_confidence,
            )
            for frame in frames
        ]
        if not sequence:
            return np.empty((0, num_joints, dims), dtype=np.float32)
        return np.stack(sequence, axis=0)

    raise ValueError(f"Unsupported file type for {sequence_path}. Use .json or .npy")


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    if sequence.shape[0] == 0:
        return sequence

    # Center around pelvis (mean of left/right hip when available).
    if sequence.shape[1] > 12:
        pelvis = (sequence[:, 11] + sequence[:, 12]) * 0.5
    else:
        pelvis = sequence.mean(axis=1)

    centered = sequence - pelvis[:, None, :]

    # Scale by median distance from pelvis to shoulder center for body-size invariance.
    if sequence.shape[1] > 6:
        shoulder = (sequence[:, 5] + sequence[:, 6]) * 0.5
        scale = np.linalg.norm(shoulder - pelvis, axis=1)
    else:
        flat = centered.reshape(centered.shape[0], -1)
        scale = np.linalg.norm(flat, axis=1)

    scale = np.clip(np.median(scale), 1e-4, None)
    return centered / scale


def load_manifest(manifest_path: Path, data_root: Path) -> list[SequenceSample]:
    samples: list[SequenceSample] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"path", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest must contain columns: {sorted(required)}; missing={missing}")

        for row in reader:
            raw_path = row["path"].strip()
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = data_root / path

            split = row.get("split", "").strip().lower() or None
            start_frame = row.get("start_frame", "").strip()
            end_frame = row.get("end_frame", "").strip()

            samples.append(
                SequenceSample(
                    path=path,
                    label=row["label"].strip(),
                    split=split,
                    start_frame=int(start_frame) if start_frame else None,
                    end_frame=int(end_frame) if end_frame else None,
                )
            )
    if not samples:
        raise ValueError(f"No samples loaded from manifest: {manifest_path}")
    return samples


# New: Load samples directly from a folder of .npy files
def load_from_folder(folder_path: Path, split: str = "train") -> list[SequenceSample]:
    """
    Scan a folder for .npy files and return SequenceSample objects with labels based on filename prefix.
    correct__*.npy -> label 'good', incorrect__*.npy -> label 'bad'.
    """
    samples: list[SequenceSample] = []
    for npy_file in folder_path.glob("*.npy"):
        fname = npy_file.name
        if fname.startswith("correct__"):
            label = "good"
        elif fname.startswith("incorrect__"):
            label = "bad"
        else:
            continue  # skip files that don't match
        samples.append(
            SequenceSample(
                path=npy_file,
                label=label,
                split=split,
            )
        )
    if not samples:
        raise ValueError(f"No .npy samples found in {folder_path}")
    return samples


class SquatSequenceDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        samples: Iterable[SequenceSample],
        label_to_idx: dict[str, int],
        *,
        num_joints: int = 17,
        dims: int = 3,
        person_id: int = 0,
        min_confidence: float = 0.0,
        max_frames: int | None = None,
    ) -> None:
        self.samples = list(samples)
        self.label_to_idx = label_to_idx
        self.num_joints = num_joints
        self.dims = dims
        self.person_id = person_id
        self.min_confidence = min_confidence
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        sequence = load_sequence(
            sample.path,
            person_id=self.person_id,
            num_joints=self.num_joints,
            dims=self.dims,
            min_confidence=self.min_confidence,
        )

        if sample.start_frame is not None or sample.end_frame is not None:
            start = sample.start_frame or 0
            end = sample.end_frame if sample.end_frame is not None else sequence.shape[0]
            sequence = sequence[start:end]

        if sequence.shape[0] == 0:
            raise ValueError(f"Empty sequence for sample: {sample.path}")

        sequence = normalize_sequence(sequence)

        if self.max_frames is not None and sequence.shape[0] > self.max_frames:
            index = np.linspace(0, sequence.shape[0] - 1, num=self.max_frames).astype(np.int64)
            sequence = sequence[index]

        label_idx = self.label_to_idx[sample.label]
        tensor = torch.from_numpy(sequence).float()
        return tensor, label_idx


def collate_padded_batch(
    batch: list[tuple[torch.Tensor, int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not batch:
        raise ValueError("Empty batch in collate_padded_batch.")

    lengths = torch.tensor([item[0].shape[0] for item in batch], dtype=torch.long)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    joint_count = int(batch[0][0].shape[1])
    dims = int(batch[0][0].shape[2])

    padded = torch.zeros((len(batch), max_len, joint_count, dims), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i, (sequence, _) in enumerate(batch):
        seq_len = sequence.shape[0]
        padded[i, :seq_len] = sequence
        mask[i, :seq_len] = True

    return padded, lengths, labels, mask


# Utility to load all golden reference .npy files from model_ready_reps/model_ready_reps/
from torch.utils.data import DataLoader
def get_golden_reference_dataloader(
    folder_path: str = "model_ready_reps/model_ready_reps/",
    batch_size: int = 8,
    shuffle: bool = False,
    num_joints: int = 17,
    dims: int = 3,
    max_frames: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Loads all .npy files from the golden reference folder and returns a DataLoader ready for model input.
    Labels are inferred from the filename: label_1 -> good, label_0 -> bad.
    """
    folder = Path(folder_path)
    samples = []
    for npy_file in folder.glob("*.npy"):
        fname = npy_file.name
        if "label_1" in fname:
            label = "good"
        elif "label_0" in fname:
            label = "bad"
        else:
            continue
        samples.append(SequenceSample(path=npy_file, label=label))
    if not samples:
        raise ValueError(f"No .npy samples found in {folder_path}")

    label_to_idx = {"good": 1, "bad": 0}
    dataset = SquatSequenceDataset(
        samples,
        label_to_idx,
        num_joints=num_joints,
        dims=dims,
        max_frames=max_frames,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_padded_batch,
        num_workers=num_workers,
    )
