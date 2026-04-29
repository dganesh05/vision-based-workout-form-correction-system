import numpy as np
import os

DATA_DIR = "data/model_ready_reps"

files = [f for f in os.listdir(DATA_DIR) if f.endswith(".npy")]

print("Total npy files:", len(files))

sample_file = os.path.join(DATA_DIR, files[0])
x = np.load(sample_file, allow_pickle=True)

print("Sample file:", files[0])
print("Shape:", x.shape)
print("Dtype:", x.dtype)
print("Preview:")
print(x[:2])