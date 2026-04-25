import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

import numpy as np

from pose_json_bridge import DEFAULT_NUM_JOINTS, convert_json_to_tvc


def _load_module_from_file(module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import adapter module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path.stem] = module
    spec.loader.exec_module(module)
    return module


def _load_adapter(module_path: Path, function_name: str) -> Callable:
    module = _load_module_from_file(module_path)
    fn = getattr(module, function_name, None)
    if fn is None:
        raise AttributeError(
            f"Adapter function '{function_name}' was not found in {module_path}."
        )
    return fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GLA-GCN 3D lifting from standardized 2D keypoints. "
            "Input can be YOLO JSON or a prebuilt .npy tensor with shape (T, 17, 3)."
        )
    )
    parser.add_argument("--input-json", type=Path, help="Path to YOLO keypoint JSON")
    parser.add_argument("--input-npy", type=Path, help="Path to prebuilt (T,17,3) numpy file")
    parser.add_argument(
        "--input-json-dir",
        type=Path,
        help="Directory containing YOLO keypoint JSON files (recursively searched)",
    )
    parser.add_argument(
        "--input-npy-dir",
        type=Path,
        help="Directory containing prebuilt (T,17,3) .npy files (recursively searched)",
    )
    parser.add_argument(
        "--adapter-file",
        type=Path,
        required=True,
        help=(
            "Python file that exposes a function run_glagcn_inference(sequence_2d, checkpoint, device)."
        ),
    )
    parser.add_argument(
        "--adapter-fn",
        default="run_glagcn_inference",
        help="Adapter function name inside --adapter-file",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to pretrained GLA-GCN checkpoint",
    )
    parser.add_argument("--device", default="cpu", help="Torch device string, e.g. cpu or cuda:0")
    parser.add_argument("--person-id", type=int, default=0, help="Target person_id in JSON")
    parser.add_argument("--num-joints", type=int, default=DEFAULT_NUM_JOINTS, help="Joint count")
    parser.add_argument(
        "--fill-missing",
        type=float,
        default=0.0,
        help="Value used when person/joint data is missing",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/poses_3d/glagcn_output.npy"),
        help="Output .npy path for 3D coordinates",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/poses_3d/glagcn"),
        help="Output directory used for batch directory mode",
    )
    return parser.parse_args()


def _validate_source_args(args: argparse.Namespace) -> str:
    provided = [
        ("input_json", args.input_json),
        ("input_npy", args.input_npy),
        ("input_json_dir", args.input_json_dir),
        ("input_npy_dir", args.input_npy_dir),
    ]
    active = [name for name, value in provided if value is not None]
    if len(active) != 1:
        raise ValueError(
            "Provide exactly one input source: --input-json, --input-npy, "
            "--input-json-dir, or --input-npy-dir"
        )
    return active[0]


def _collect_inputs(args: argparse.Namespace, source: str) -> list[tuple[str, Path]]:
    if source == "input_json":
        if not args.input_json.exists():
            raise FileNotFoundError(f"Input JSON not found: {args.input_json}")
        return [("json", args.input_json)]

    if source == "input_npy":
        if not args.input_npy.exists():
            raise FileNotFoundError(f"Input NPY not found: {args.input_npy}")
        return [("npy", args.input_npy)]

    if source == "input_json_dir":
        if not args.input_json_dir.exists() or not args.input_json_dir.is_dir():
            raise FileNotFoundError(f"Input JSON directory not found: {args.input_json_dir}")
        files = sorted(args.input_json_dir.rglob("*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON files found in {args.input_json_dir}")
        return [("json", path) for path in files]

    if not args.input_npy_dir.exists() or not args.input_npy_dir.is_dir():
        raise FileNotFoundError(f"Input NPY directory not found: {args.input_npy_dir}")
    files = sorted(args.input_npy_dir.rglob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No NPY files found in {args.input_npy_dir}")
    return [("npy", path) for path in files]


def _load_sequence(args: argparse.Namespace, source_type: str, source_path: Path) -> np.ndarray:
    if source_type == "npy":
        sequence = np.load(source_path)
    else:
        sequence = convert_json_to_tvc(
            json_path=source_path,
            person_id=args.person_id,
            num_joints=args.num_joints,
            fill_missing=args.fill_missing,
        )

    if sequence.ndim != 3:
        raise ValueError(f"Expected 3D tensor (T,V,C). Got shape: {sequence.shape}")
    if sequence.shape[1] != args.num_joints:
        raise ValueError(
            f"Expected V={args.num_joints} joints. Got V={sequence.shape[1]} from {sequence.shape}."
        )
    if sequence.shape[2] != 3:
        raise ValueError(
            f"Expected C=3 channels (x,y,confidence). Got C={sequence.shape[2]}"
        )
    return sequence.astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    if not args.adapter_file.exists():
        raise FileNotFoundError(f"Adapter file not found: {args.adapter_file}")

    source = _validate_source_args(args)
    input_items = _collect_inputs(args, source)

    inference_fn = _load_adapter(module_path=args.adapter_file, function_name=args.adapter_fn)

    batch_mode = source in {"input_json_dir", "input_npy_dir"}
    if batch_mode:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for source_type, source_path in input_items:
        sequence_2d = _load_sequence(args, source_type=source_type, source_path=source_path)

        sequence_3d = inference_fn(
            sequence_2d=sequence_2d,
            checkpoint=str(args.checkpoint) if args.checkpoint else None,
            device=args.device,
        )
        sequence_3d = np.asarray(sequence_3d, dtype=np.float32)

        if sequence_3d.shape[:2] != sequence_2d.shape[:2]:
            raise ValueError(
                "GLA-GCN adapter returned mismatched T/V dimensions: "
                f"input={sequence_2d.shape}, output={sequence_3d.shape}"
            )
        if sequence_3d.shape[2] != 3:
            raise ValueError(
                "GLA-GCN adapter must return channels (x,y,z), so output C must equal 3. "
                f"Got shape {sequence_3d.shape}"
            )

        if batch_mode:
            out_path = args.output_dir / f"{source_path.stem}_glagcn.npy"
        else:
            out_path = args.output

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, sequence_3d)
        print(f"Saved GLA-GCN output to {out_path} with shape {sequence_3d.shape}")


if __name__ == "__main__":
    main()
