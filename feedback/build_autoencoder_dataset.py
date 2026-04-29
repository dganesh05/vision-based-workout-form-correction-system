import os
import numpy as np

DATA_DIR = "data/model_ready_reps"
SAVE_DIR = "feedback"

golden_data = []
user_data = []

files = [f for f in os.listdir(DATA_DIR) if f.endswith(".npy")]

print(f"Total files found: {len(files)}")

for file in files:
    file_path = os.path.join(DATA_DIR, file)

    try:
        x = np.load(file_path, allow_pickle=True)

        # make sure shape is correct
        if x.shape != (41, 5):
            print(f"Skipping {file} -> unexpected shape {x.shape}")
            continue

        # golden reference = good squat training data
        if "golden reference" in file.lower():
            golden_data.append(x)

        # original data = test/user squat data
        elif "original data" in file.lower():
            user_data.append(x)

    except Exception as e:
        print(f"Error reading {file}: {e}")

golden_data = np.array(golden_data)
user_data = np.array(user_data)

print("\n===================================")
print("Dataset Creation Complete")
print("===================================")

print(f"Golden Reference Shape: {golden_data.shape}")
print(f"User Test Shape: {user_data.shape}")

np.save(os.path.join(SAVE_DIR, "X_golden_train.npy"), golden_data)
np.save(os.path.join(SAVE_DIR, "X_user_test.npy"), user_data)

print("\nSaved files:")
print("- feedback/X_golden_train.npy")
print("- feedback/X_user_test.npy")
print("\nDone 🚀")