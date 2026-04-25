import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_NUM_JOINTS = 17


def _find_person(people: list[dict], person_id: int) -> dict | None:
    for person in people:
        if int(person.get("person_id", -1)) == person_id:
            return person
    return None


def _frame_to_joint_array(
    frame_payload: dict,
    person_id: int,
    num_joints: int,
    fill_missing: float,
) -> np.ndarray:
    frame_array = np.full((num_joints, 3), fill_missing, dtype=np.float32)

    people = frame_payload.get("people", [])
    if not isinstance(people, list):
        return frame_array

    person = _find_person(people, person_id=person_id)
    if person is None:
        return frame_array

    keypoints = person.get("keypoints", [])
    if not isinstance(keypoints, list):
        return frame_array

    for joint in keypoints:
        joint_id = int(joint.get("joint_id", -1))
        if joint_id < 0 or joint_id >= num_joints:
            continue

        x = float(joint.get("x", fill_missing))
        y = float(joint.get("y", fill_missing))
        conf = float(joint.get("confidence", fill_missing))
        frame_array[joint_id] = np.array([x, y, conf], dtype=np.float32)

    return frame_array


def convert_json_to_tvc(
    json_path: Path,
    person_id: int = 0,
    num_joints: int = DEFAULT_NUM_JOINTS,
    fill_missing: float = 0.0,
) -> np.ndarray:
    with json_path.open("r", encoding="utf-8") as f:
        frames = json.load(f)

    if not isinstance(frames, list):
        raise ValueError(f"Expected top-level list in {json_path}")

    sequence = [
        _frame_to_joint_array(
            frame_payload=frame_payload,
            person_id=person_id,
            num_joints=num_joints,
            fill_missing=fill_missing,
        )
        for frame_payload in frames
    ]

    if not sequence:
        return np.empty((0, num_joints, 3), dtype=np.float32)

    return np.stack(sequence, axis=0).astype(np.float32, copy=False)


def convert_many_json_files(
    json_paths: Iterable[Path],
    output_dir: Path,
    person_id: int,
    num_joints: int,
    fill_missing: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for json_path in json_paths:
        sequence = convert_json_to_tvc(
            json_path=json_path,
            person_id=person_id,
            num_joints=num_joints,
            fill_missing=fill_missing,
        )
        out_path = output_dir / f"{json_path.stem}.npy"
        np.save(out_path, sequence)
        print(f"Saved {out_path} with shape {sequence.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert keypoint JSON from process_videos_hard_attention.py into "
            "a standardized NumPy tensor with shape (T, 17, 3)."
        )
    )
    parser.add_argument("--input-json", type=Path, help="Single keypoint JSON file")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing keypoint JSON files (recursively searched)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/standardized_2d"),
        help="Directory where .npy outputs are written",
    )
    parser.add_argument("--person-id", type=int, default=0, help="Target person_id to extract")
    parser.add_argument("--num-joints", type=int, default=DEFAULT_NUM_JOINTS, help="Joint count")
    parser.add_argument(
        "--fill-missing",
        type=float,
        default=0.0,
        help="Value used when person/joint is missing in a frame",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_json is None and args.input_dir is None:
        raise ValueError("Provide either --input-json or --input-dir")

    json_paths: list[Path]
    if args.input_json is not None:
        json_paths = [args.input_json]
    else:
        if not args.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
        json_paths = sorted(args.input_dir.rglob("*_keypoints.json"))

    if not json_paths:
        print("No keypoint JSON files found.")
        return

    convert_many_json_files(
        json_paths=json_paths,
        output_dir=args.output_dir,
        person_id=args.person_id,
        num_joints=args.num_joints,
        fill_missing=args.fill_missing,
    )


if __name__ == "__main__":
    main()
