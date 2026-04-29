from __future__ import annotations

import torch
import numpy as np
from pathlib import Path

from feedback6class.model import BiCGRUClassifier


# ─────────────────────────────────────────────
# ENSEMBLE INFERENCE ENGINE
# ─────────────────────────────────────────────

class SquatInferenceEngine:
    def __init__(self, model_paths, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.models = [self._load_model(p) for p in model_paths]

        print(f"[INFO] Loaded {len(self.models)} models")

    # ─────────────────────────────────────────────
    def _load_model(self, path: Path):

        ckpt = torch.load(path, map_location=self.device)

        # IMPORTANT: we assume same architecture used in training
        # (you MUST ensure all folds used same config)

        model = BiCGRUClassifier(
            input_dims=3,
            num_joints=7,
            num_angles=5,
            spatial_channels=32,
            angle_channels=16,
            gru_hidden_size=16,
            gru_layers=1,
            dropout=0.3,
            num_classes=2
        )

        model.load_state_dict(ckpt["model_state"])
        model.to(self.device)
        model.eval()

        return model

    # ─────────────────────────────────────────────
    def predict_ensemble(self, joints, angles, lengths, mask):

        joints = joints.to(self.device)
        angles = angles.to(self.device)
        lengths = lengths.to(self.device)
        mask = mask.to(self.device)

        probs_list = []
        attn_list = []

        with torch.no_grad():
            for model in self.models:
                logits, attn = model(joints, angles, lengths, mask)
                probs_list.append(torch.softmax(logits, dim=-1))
                attn_list.append(attn)

        avg_probs = torch.mean(torch.stack(probs_list), dim=0)
        avg_attn = torch.mean(torch.stack(attn_list), dim=0)

        return avg_probs, avg_attn

    # ─────────────────────────────────────────────
    def decode(self, probs):
        probs = probs[0]
        cls = torch.argmax(probs).item()
        conf = probs[cls].item()
        return cls, conf, probs.cpu().numpy()

    # ─────────────────────────────────────────────
    def feedback(self, cls):
        return {
            0: {
                "name": "good_form",
                "feedback": "Great squat form!",
                "cues": []
            },
            1: {
                "name": "non_golden",
                "feedback": "Form issue detected",
                "cues": ["Check knee alignment", "Keep chest up"]
            }
        }.get(cls, {
            "name": "unknown",
            "feedback": "Unknown pattern",
            "cues": []
        })

    # ─────────────────────────────────────────────
    def analyze(self, inputs):
        joints, angles, lengths, mask = inputs

        probs, attn = self.predict_ensemble(joints, angles, lengths, mask)

        cls, conf, prob_vec = self.decode(probs)

        label_info = self.feedback(cls)

        return {
            "prediction": {
                "class": cls,
                "name": label_info["name"],
                "confidence": conf,
                "probabilities": prob_vec.tolist(),
            },
            "feedback": {
                "primary_feedback": label_info["feedback"],
                "cues": label_info.get("cues", []),
            },
            "attention": attn[0].cpu().numpy(),
        }