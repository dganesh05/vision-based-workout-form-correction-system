# feedback/build_our_sequence_dataset.py

import os
import numpy as np
import pandas as pd

# =====================================
# CONFIG
# =====================================

DATA_FOLDER = "data/angles_csv"

OUTPUT_X = "feedback/X_our_sequences.npy"
OUTPUT_Y = "feedback/y_our_labels.npy"

SEQUENCE_LENGTH = 30

FEATURE_COLUMNS = [
    "Right_Knee",
    "Left_Knee",
    "Right_Hip",
    "Left_Hip",
    "Spine_Lean"
]

# =====================================
# LOAD ALL CSV FILES
# =====================================

all_sequences = []
all_labels = []

print("Reading all angle CSV files...\n")

for file_name in os.listdir(DATA_FOLDER):

    if not file_name.endswith(".csv"):
        continue

    file_path = os.path.join(DATA_FOLDER, file_name)

    print(f"Processing: {file_name}")

    df = pd.read_csv(file_path)

    # Basic cleaning
    df = df.dropna()

    if len(df) < SEQUENCE_LENGTH:
        print("Skipped (too short)\n")
        continue

    # Features only
    features = df[FEATURE_COLUMNS].values

    # Temporary training label:
    # golden reference = 0
    # original data = 1

    if "golden reference" in file_name.lower():
        label = 0
    else:
        label = 1

    # Sliding window
    for i in range(len(df) - SEQUENCE_LENGTH + 1):
        seq = features[i:i + SEQUENCE_LENGTH]

        all_sequences.append(seq)
        all_labels.append(label)

print("\nSequence creation complete ✅")

# =====================================
# CONVERT TO NUMPY
# =====================================

X = np.array(all_sequences)
y = np.array(all_labels)

print("\nFinal Shapes:")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nLabel Counts:")
unique, counts = np.unique(y, return_counts=True)

for u, c in zip(unique, counts):
    print(f"Label {u}: {c}")

# =====================================
# SAVE FILES
# =====================================

np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y)

print("\n====================================")
print("Sequence dataset created successfully ✅")
print("Saved:")
print(OUTPUT_X)
print(OUTPUT_Y)
print("====================================")