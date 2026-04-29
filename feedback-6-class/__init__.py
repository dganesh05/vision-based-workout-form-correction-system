"""squat_classifier — multi-class squat form analysis package."""
from .data import (
    LABEL_SCHEMA,
    CSV_ANGLE_COLUMNS,
    SequenceSample,
    SquatSequenceDataset,
    load_from_folder,
    collate_padded_batch,
    discover_label_set,
)
from .model import BiCGRUClassifier
from .train_utils import set_seed, train_model, get_feedback, format_feedback_text

__all__ = [
    "LABEL_SCHEMA",
    "CSV_ANGLE_COLUMNS",
    "SequenceSample",
    "SquatSequenceDataset",
    "load_from_folder",
    "collate_padded_batch",
    "discover_label_set",
    "BiCGRUClassifier",
    "set_seed",
    "train_model",
    "get_feedback",
    "format_feedback_text",
]
