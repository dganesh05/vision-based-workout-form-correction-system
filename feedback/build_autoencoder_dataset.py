# feedback/build_autoencoder_dataset.py
# UPDATED FOR PROPER 3D NORMALIZATION
# IMPORTANT:
# Fit scaler ONLY on Golden Reference
# Then apply same scaler to User Data

import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

# =====================================
# SETTINGS
# =====================================

DATA_PATH = "data/model_ready_reps"

print("Loading all squat files...")

all_files = sorted([
    f for f in os.listdir(DATA_PATH)
    if f.endswith(".npy")
])

print(f"Total files found: {len(all_files)}")

golden_data = []
user_data = []

# =====================================
# LOAD + RESHAPE
# (41, 7, 3) → (41, 21)
# =====================================

for file in all_files:
    file_path = os.path.join(DATA_PATH, file)

    arr = np.load(file_path)

    if arr.shape == (41, 7, 3):
        arr = arr.reshape(41, 21)

        if "golden reference" in file.lower():
            golden_data.append(arr)
        else:
            user_data.append(arr)

# Convert to numpy arrays
X_golden_train = np.array(golden_data)
X_user_test = np.array(user_data)

print("\nBefore Normalization:")
print("Golden Shape:", X_golden_train.shape)
print("User Shape:", X_user_test.shape)

# =====================================
# NORMALIZATION (VERY IMPORTANT)
# FIT ONLY ON GOLDEN DATA
# =====================================

print("\nApplying Proper MinMax Normalization...")

samples_g, timesteps, features = X_golden_train.shape
samples_u = X_user_test.shape[0]

# reshape for scaler
golden_2d = X_golden_train.reshape(-1, features)
user_2d = X_user_test.reshape(-1, features)

# FIT ONLY ON GOLDEN REFERENCE
scaler = MinMaxScaler()
scaler.fit(golden_2d)

# transform both using SAME scaler
golden_scaled = scaler.transform(golden_2d)
user_scaled = scaler.transform(user_2d)

# reshape back
X_golden_train = golden_scaled.reshape(
    samples_g,
    timesteps,
    features
)

X_user_test = user_scaled.reshape(
    samples_u,
    timesteps,
    features
)

print("\nAfter Normalization:")
print("Golden Shape:", X_golden_train.shape)
print("User Shape:", X_user_test.shape)

# =====================================
# SAVE SCALER
# Needed for prediction consistency
# =====================================

joblib.dump(
    scaler,
    "feedback/golden_scaler.pkl"
)

# =====================================
# SAVE DATA
# =====================================

np.save(
    "feedback/X_golden_train.npy",
    X_golden_train
)

np.save(
    "feedback/X_user_test.npy",
    X_user_test
)

print("\n===================================")
print("PROPER 3D DATASET CREATION COMPLETE")
print("===================================")

print("\nSaved files:")
print("- feedback/X_golden_train.npy")
print("- feedback/X_user_test.npy")
print("- feedback/golden_scaler.pkl")

print("\nDone 🚀")