"""
predict.py — Run inference on a single squat rep and display form feedback.

Usage
─────
# From a .npy file (with optional companion .csv for angle features):
python predict.py --input path/to/rep.npy --model-dir runs/squat_classifier/

# From a standalone .csv of angles:
python predict.py --input path/to/angles.csv --model-dir runs/squat_classifier/

Output
──────
Prints the top-K form predictions with confidence scores and correction cues.
Optionally saves a JSON file with full results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from squat_classifier.data import (
    CSV_ANGLE_COLUMNS,
    load_sequence,
    normalize_joints,
    normalize_angles,
)
from squat_classifier.model import BiCGRUClassifier
from squat_classifier.train_utils import format_feedback_text, get_feedback


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict squat form from a single rep file.")
    p.add_argument("--input", type=Path, required=True, help=".npy or .csv rep file.")
    p.add_argument(
        "--model-dir", type=Path, default=Path("runs/squat_classifier"),
        help="Directory containing best_model.pt and metadata.json.",
    )
    p.add_argument("--top-k", type=int, default=3, help="Number of top predictions to show.")
    p.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def load_metadata(model_dir: Path) -> dict:
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {model_dir}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_dir: Path, meta: dict, device: torch.device) -> BiCGRUClassifier:
    model = BiCGRUClassifier(
        input_dims=meta["dims"],
        num_joints=meta["num_joints"],
        num_angles=meta.get("num_angles", 5),
        spatial_channels=meta["spatial_channels"],
        angle_channels=meta.get("angle_channels", 32),
        gru_hidden_size=meta["gru_hidden"],
        gru_layers=meta["gru_layers"],
        dropout=meta.get("dropout", 0.3),
        num_classes=meta["num_classes"],
    ).to(device)
    weights_path = model_dir / "best_model.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"best_model.pt not found in {model_dir}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def prepare_input(
    input_path: Path,
    meta: dict,
    device: torch.device,
):
    num_joints = meta["num_joints"]
    dims = meta["dims"]
    num_angles = meta.get("num_angles", 5)

    joints, angles = load_sequence(input_path, num_joints=num_joints, dims=dims)
    joints = normalize_joints(joints)

    if angles is None:
        import numpy as np
        angles = np.zeros((joints.shape[0], num_angles), dtype="float32")
    else:
        angles = normalize_angles(angles)
        T = min(joints.shape[0], angles.shape[0])
        joints = joints[:T]
        angles = angles[:T]

    import torch
    joints_t = torch.from_numpy(joints).float().unsqueeze(0).to(device)   # (1, T, J, C)
    angles_t = torch.from_numpy(angles).float().unsqueeze(0).to(device)   # (1, T, 5)
    lengths = torch.tensor([joints.shape[0]], dtype=torch.long).to(device) # (1,)
    mask = torch.ones(1, joints.shape[0], dtype=torch.bool).to(device)    # (1, T)

    return joints_t, angles_t, lengths, mask


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    meta = load_metadata(args.model_dir)

    # Rebuild idx_to_label with int keys
    idx_to_label = {int(k): v for k, v in meta["idx_to_label"].items()}
    label_schema = {int(k): v for k, v in meta.get("label_schema", {}).items()}

    model = load_model(args.model_dir, meta, device)

    joints_t, angles_t, lengths, mask = prepare_input(args.input, meta, device)

    with torch.no_grad():
        logits, attn_weights = model(
            joints=joints_t, angles=angles_t, lengths=lengths, mask=mask
        )

    import torch.nn.functional as F
    probs = F.softmax(logits, dim=-1).squeeze(0)
    top_probs, top_idxs = probs.topk(min(args.top_k, len(probs)))

    feedback_list = []
    for rank, (prob, idx) in enumerate(
        zip(top_probs.tolist(), top_idxs.tolist()), start=1
    ):
        schema = label_schema.get(idx, {})
        feedback_list.append({
            "rank": rank,
            "label_idx": idx,
            "label_name": idx_to_label.get(idx, f"class_{idx}"),
            "display": schema.get("display", idx_to_label.get(idx, f"class_{idx}")),
            "confidence": round(prob, 4),
            "feedback": schema.get("feedback", ""),
            "cues": schema.get("cues", []),
        })

    print(format_feedback_text(feedback_list))
    print(f"\nInput file : {args.input}")
    print(f"Model dir  : {args.model_dir}")
    print(f"Device     : {device}")

    # Attention summary (most-attended frames)
    attn = attn_weights.squeeze(0).cpu().tolist()
    T = len(attn)
    top_frames = sorted(range(T), key=lambda i: attn[i], reverse=True)[:5]
    print(f"\nTop attended frames (of {T}): {[f+1 for f in top_frames]}")

    if args.output_json:
        output = {
            "input": str(args.input),
            "predictions": feedback_list,
            "attention_weights": attn,
        }
        args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
