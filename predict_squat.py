from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from squat_classifier.data import collate_padded_batch, load_sequence, normalize_sequence
from squat_classifier.model import BiCGRUClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict squat quality from a joint sequence file.")
    parser.add_argument("--sequence", type=Path, required=True, help="Path to sequence (.json or .npy)")
    parser.add_argument("--model", type=Path, required=True, help="Path to best_model.pt")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata.json")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    idx_to_label = {int(k): v for k, v in metadata["idx_to_label"].items()}

    model = BiCGRUClassifier(
        input_dims=int(metadata["dims"]),
        spatial_channels=int(metadata["spatial_channels"]),
        gru_hidden_size=int(metadata["gru_hidden"]),
        gru_layers=int(metadata["gru_layers"]),
        dropout=float(metadata["dropout"]),
        num_classes=len(idx_to_label),
    )
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    sequence = load_sequence(
        args.sequence,
        person_id=int(metadata["person_id"]),
        num_joints=int(metadata["num_joints"]),
        dims=int(metadata["dims"]),
        min_confidence=float(metadata["min_confidence"]),
    )
    sequence = normalize_sequence(sequence)
    sequence_tensor = torch.from_numpy(sequence).float()
    padded, lengths, _, mask = collate_padded_batch([(sequence_tensor, 0)])

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = model.to(device)
    padded = padded.to(device)
    lengths = lengths.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        logits, _ = model(sequences=padded, lengths=lengths, mask=mask)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = int(probs.argmax().item())
    pred_label = idx_to_label[pred_idx]
    confidence = float(probs[pred_idx].item())

    print(f"Predicted class: {pred_label}")
    print(f"Confidence: {confidence:.4f}")
    print("Class probabilities:")
    for idx, prob in enumerate(probs.tolist()):
        print(f"  {idx_to_label[idx]}: {prob:.4f}")


if __name__ == "__main__":
    main()

