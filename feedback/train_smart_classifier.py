import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

# =====================================
# LOAD SMART DATASET
# =====================================

print("Loading Updated Smart Classifier Dataset...")

X = np.load("feedback/X_smart_classifier.npy")
y = np.load("feedback/y_smart_classifier.npy")

print("X Shape:", X.shape)
print("y Shape:", y.shape)

print("\nLabel Distribution:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")

timesteps = X.shape[1]
features = X.shape[2]

# =====================================
# TRAIN / VALIDATION / TEST SPLIT
# =====================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset Split Complete")
print("Train Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)
print("Test Shape:", X_test.shape)

# =====================================
# CLASS WEIGHTS
# =====================================

print("\nCalculating Class Weights...")

class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {
    0: class_weights_values[0],
    1: class_weights_values[1]
}

print("Class Weights:", class_weight_dict)

# =====================================
# SMOOTHER + STRONGER MODEL
# =====================================

print("\nBuilding Improved Smart BiLSTM Classifier...")

model = Sequential()

model.add(
    Bidirectional(
        LSTM(
            32,
            return_sequences=True,
            dropout=0.20,
            recurrent_dropout=0.10
        ),
        input_shape=(timesteps, features)
    )
)

model.add(
    Bidirectional(
        LSTM(
            16,
            return_sequences=False,
            dropout=0.20,
            recurrent_dropout=0.10
        )
    )
)

model.add(BatchNormalization())

model.add(
    Dropout(0.20)
)

model.add(
    Dense(
        16,
        activation="relu"
    )
)

model.add(
    Dropout(0.15)
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)

# =====================================
# COMPILE
# =====================================

model.compile(
    optimizer=Adam(
        learning_rate=0.0005
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
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
# =====================================

print("\nStarting Improved Smart Training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# =====================================
# TEST EVALUATION
# =====================================

print("\nEvaluating Final Test Accuracy...")

test_loss, test_acc = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nFINAL TEST ACCURACY: {test_acc:.4f}")
print(f"FINAL TEST LOSS: {test_loss:.4f}")

# =====================================
# PREDICTIONS
# =====================================

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"\nROC-AUC Score: {roc_auc:.4f}")

print("\n===================================")
print("FINAL CLASSIFICATION REPORT")
print("===================================")

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

# =====================================
# SAVE MODEL
# =====================================

model.save(
    "feedback/final_smart_classifier.keras"
)

print("\nSaved:")
print("- feedback/final_smart_classifier.keras")

# =====================================
# ACCURACY CURVE
# =====================================

plt.figure(figsize=(10, 6))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Improved Smart Classifier Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("feedback/final_smart_accuracy_curve.png")
plt.show()

# =====================================
# LOSS CURVE
# =====================================

plt.figure(figsize=(10, 6))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Improved Smart Classifier Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Binary Crossentropy Loss")
plt.legend()

plt.savefig("feedback/final_smart_loss_curve.png")
plt.show()

print("\nDone 🚀")