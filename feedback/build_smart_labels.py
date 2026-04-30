import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# =====================================
# SETTINGS
# =====================================

MODEL_PATH = "feedback/final_bilstm_autoencoder.keras"
DATA_PATH = "data/model_ready_reps"

print("Loading trained autoencoder...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully")

# =====================================
# LOAD ALL FILES
# =====================================

all_files = sorted([
    f for f in os.listdir(DATA_PATH)
    if f.endswith(".npy")
])

print(f"\nTotal files found: {len(all_files)}")

X = []
filenames = []

for file in all_files:
    file_path = os.path.join(DATA_PATH, file)

    arr = np.load(file_path)

    if arr.shape == (41, 5):
        X.append(arr)
        filenames.append(file)

X = np.array(X)

print("Original Shape:", X.shape)

# =====================================
# NORMALIZATION (IMPORTANT)
# =====================================

print("\nApplying MinMax Normalization...")

samples, timesteps, features = X.shape

scaler = MinMaxScaler()

X_reshaped = X.reshape(-1, features)
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(samples, timesteps, features)

print("Normalized Shape:", X.shape)

# Save scaler-ready normalized data
np.save("feedback/X_normalized.npy", X)

# =====================================
# RECONSTRUCTION ERROR
# =====================================

print("\nRunning reconstruction...")

reconstructed = model.predict(X)

errors = np.mean(
    np.square(X - reconstructed),
    axis=(1, 2)
)

print("\nSample Errors:")
print(errors[:10])

# =====================================
# BETTER THRESHOLD
# =====================================

# old = 60 percentile
# better = 50~55 percentile

threshold = np.percentile(errors, 55)

print(f"\nImproved Dynamic Threshold: {threshold:.4f}")

# label:
# 1 = good squat
# 0 = needs improvement

labels = np.where(
    errors <= threshold,
    1,
    0
)

print("\nLabel Distribution:")

unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")

# =====================================
# SAVE
# =====================================

np.save("feedback/X_smart_classifier.npy", X)
np.save("feedback/y_smart_classifier.npy", labels)

print("\n===================================")
print("UPDATED SMART LABEL DATASET CREATED")
print("===================================")

print("Saved:")
print("- feedback/X_smart_classifier.npy")
print("- feedback/y_smart_classifier.npy")
print("- feedback/X_normalized.npy")

print("\nDone 🚀")