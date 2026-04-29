# feedback/train_bilstm_autoencoder.py

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense,
    RepeatVector,
    TimeDistributed,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

# =====================================
# LOAD DATA
# =====================================

print("Loading Autoencoder Dataset...")

X_train = np.load("feedback/X_golden_train.npy")

print("Training Shape:", X_train.shape)

timesteps = X_train.shape[1]   # 41
features = X_train.shape[2]    # 21

# =====================================
# BUILD BiLSTM AUTOENCODER
# New Input: (41, 21)
# Old Input: (41, 5)
# =====================================

print("\nBuilding 3D BiLSTM Autoencoder...")

inputs = Input(shape=(timesteps, features))

# =====================================
# ENCODER
# =====================================

x = Bidirectional(
    LSTM(
        64,
        return_sequences=True,
        dropout=0.20,
        recurrent_dropout=0.10
    )
)(inputs)

x = Bidirectional(
    LSTM(
        32,
        return_sequences=False,
        dropout=0.20,
        recurrent_dropout=0.10
    )
)(x)

x = BatchNormalization()(x)

encoded = Dense(
    32,
    activation="relu",
    name="bottleneck_features"
)(x)

# =====================================
# DECODER
# =====================================

x = RepeatVector(timesteps)(encoded)

x = Bidirectional(
    LSTM(
        32,
        return_sequences=True,
        dropout=0.20,
        recurrent_dropout=0.10
    )
)(x)

x = Bidirectional(
    LSTM(
        64,
        return_sequences=True,
        dropout=0.20,
        recurrent_dropout=0.10
    )
)(x)

decoded = TimeDistributed(
    Dense(features)
)(x)

# =====================================
# FINAL MODEL
# =====================================

model = Model(inputs, decoded)

# =====================================
# COMPILE
# =====================================

model.compile(
    optimizer=Adam(
        learning_rate=0.0005
    ),
    loss="mse"
)

model.summary()

# =====================================
# CALLBACKS
# =====================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=0.00005,
    verbose=1
)

# =====================================
# TRAIN
# Autoencoder = X → X
# =====================================

print("\nStarting 3D Autoencoder Training...\n")

history = model.fit(
    X_train,
    X_train,
    validation_split=0.20,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# =====================================
# SAVE MODEL
# =====================================

model.save(
    "feedback/final_bilstm_autoencoder_3d.keras"
)

print("\nSaved:")
print("- feedback/final_bilstm_autoencoder_3d.keras")

# =====================================
# LOSS CURVE
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

plt.title("3D BiLSTM Autoencoder Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Reconstruction Loss (MSE)")
plt.legend()

plt.savefig(
    "feedback/autoencoder_loss_curve_3d.png"
)

plt.show()

print("Saved:")
print("- feedback/autoencoder_loss_curve_3d.png")

print("\nDone 🚀")