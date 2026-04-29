import numpy as np
from tensorflow.keras.models import load_model

print("Loading trained model...")

model = load_model(
    "feedback/final_bilstm_autoencoder.keras",
    compile=False
)

# =====================================
# LOAD BOTH DATASETS
# =====================================

X_golden = np.load("feedback/X_golden_train.npy")
X_user = np.load("feedback/X_user_test.npy")

print("\nGolden Shape:", X_golden.shape)
print("User Shape:", X_user.shape)

# =====================================
# PREDICT RECONSTRUCTION
# =====================================

print("\nRunning predictions...")

golden_pred = model.predict(X_golden)
user_pred = model.predict(X_user)

# =====================================
# RECONSTRUCTION ERROR
# =====================================

golden_errors = np.mean(
    np.square(X_golden - golden_pred),
    axis=(1, 2)
)

user_errors = np.mean(
    np.square(X_user - user_pred),
    axis=(1, 2)
)

# =====================================
# FINAL RESULTS
# =====================================

print("\n====================================")
print("GENERALIZATION CHECK")
print("====================================")

print(f"\nGolden Mean Error: {golden_errors.mean():.4f}")
print(f"Golden Min Error : {golden_errors.min():.4f}")
print(f"Golden Max Error : {golden_errors.max():.4f}")

print("\n-----------------------------")

print(f"User Mean Error: {user_errors.mean():.4f}")
print(f"User Min Error : {user_errors.min():.4f}")
print(f"User Max Error : {user_errors.max():.4f}")

print("\n====================================")

if user_errors.mean() > golden_errors.mean():
    print("GOOD → Model is learning patterns and detecting deviations ✅")
else:
    print("WARNING → Model may be memorizing too much ⚠️")

print("\nDone 🚀")