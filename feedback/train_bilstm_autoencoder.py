# feedback/train_bilstm_autoencoder.py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
    Dense,
    Bidirectional,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

# =====================================
# LOAD DATA
# =====================================

print("Loading Golden Reference Training Data...")

X = np.load("feedback/X_golden_train.npy")

print("Full Dataset Shape:", X.shape)

# =====================================
# TRAIN / VALIDATION SPLIT
# =====================================

X_train, X_val = train_test_split(
    X,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("\nDataset Split Complete")
print(f"Training Shape: {X_train.shape}")
print(f"Validation Shape: {X_val.shape}")

timesteps = X_train.shape[1]   # 41
features = X_train.shape[2]    # 5

print(f"\nTimesteps: {timesteps}")
print(f"Features: {features}")

# =====================================
# BUILD IMPROVED BiLSTM AUTOENCODER
# =====================================

print("\nBuilding Improved BiLSTM Autoencoder Model...")

inputs = Input(shape=(timesteps, features))

# =====================================
# ENCODER
# Added dropout to prevent memorization
# =====================================

encoded = Bidirectional(
    LSTM(
        64,
        activation="tanh",
        return_sequences=False,
        dropout=0.30,
        recurrent_dropout=0.20
    )
)(inputs)

# Additional regularization
encoded = Dropout(0.30)(encoded)

# =====================================
# SMALLER BOTTLENECK
# Reduced from 32 → 16
# Forces real learning
# =====================================

bottleneck = Dense(
    16,
    activation="relu"
)(encoded)

# =====================================
# DECODER
# =====================================

decoded = RepeatVector(timesteps)(bottleneck)

decoded = Bidirectional(
    LSTM(
        64,
        activation="tanh",
        return_sequences=True,
        dropout=0.30,
        recurrent_dropout=0.20
    )
)(decoded)

decoded = Dropout(0.30)(decoded)

outputs = TimeDistributed(
    Dense(features)
)(decoded)

model = Model(inputs, outputs)

# =====================================
# COMPILE
# =====================================

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# =====================================
# CALLBACKS
# =====================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=7,
    min_lr=0.00001,
    verbose=1
)

# =====================================
# TRAIN MODEL
# =====================================

print("\nStarting Improved Training...\n")

history = model.fit(
    X_train,
    X_train,
    validation_data=(X_val, X_val),
    epochs=100,
    batch_size=8,
    shuffle=True,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# =====================================
# FINAL VALIDATION EVALUATION
# =====================================

print("\nEvaluating Final Validation Loss...")

val_loss = model.evaluate(
    X_val,
    X_val,
    verbose=0
)

print(f"\nFinal Validation Loss: {val_loss:.4f}")

# =====================================
# SAVE MODEL
# Use .keras instead of old .h5
# =====================================

model.save("feedback/final_bilstm_autoencoder.keras")

print("\n===================================")
print("FINAL IMPROVED BiLSTM TRAINING COMPLETE")
print("===================================")
print("Saved:")
print("- feedback/final_bilstm_autoencoder.keras")

# =====================================
# PLOT LOSS CURVE
# =====================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Improved BiLSTM Autoencoder Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()

plt.savefig("feedback/loss_curve.png")
plt.show()

print("\nSaved:")
print("- feedback/loss_curve.png")

print("\nSystem Ready 🚀")