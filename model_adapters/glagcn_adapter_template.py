from __future__ import annotations

import numpy as np


def run_glagcn_inference(
    sequence_2d: np.ndarray,
    checkpoint: str | None,
    device: str,
) -> np.ndarray:
    """
    Replace this template with direct calls into the official GLA-GCN repository.

    Required behavior:
    - Input shape: (T, 17, 3) channels are (x, y, confidence)
    - Output shape: (T, 17, 3) channels must be (x, y, z)
    """
    raise NotImplementedError(
        "Implement GLA-GCN inference here.\n"
        "Tip: clone the official repo and import its model/inference utilities from this file."
    )
