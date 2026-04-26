# Feedback Transformer Module

This module is an initial prototype for classifying squat form using pose-derived temporal features.

Input shape:
(batch_size, sequence_length, num_features)

Example:
(32, 60, 8)

Output:
Class probabilities for squat form labels:
- correct
- shallow_squat
- knee_valgus
- trunk_lean
- heel_lift

This module currently uses dummy inputs and will later be connected to extracted 3D pose features.