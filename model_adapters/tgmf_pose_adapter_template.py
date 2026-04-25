from __future__ import annotations

import numpy as np


def run_tgmf_pose_inference(
    sequence_2d: np.ndarray,
    checkpoint: str | None,
    device: str,
    prompt: str,
) -> np.ndarray:
    """
    Replace this template with direct calls into the official TGMF-Pose repository.

    Required behavior:
    - Input shape: (T, 17, 3) channels are (x, y, confidence)
    - Output shape: (T, 17, 3) channels must be (x, y, z)
    - prompt carries the text-guided prior (e.g., "A person performing a barbell squat.")
    """
    raise NotImplementedError(
        "Implement TGMF-Pose inference here.\n"
        "Tip: clone the official repo and import its model/inference utilities from this file."
    )
