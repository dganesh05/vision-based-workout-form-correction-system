import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =====================================
# LOAD DATA
# =====================================

print("Loading sequence dataset...")

X = np.load("feedback/X_our_sequences.npy")
y = np.load("feedback/y_our_labels.npy")

print("\nDataset Loaded Successfully ✅")
print("X shape:", X.shape)
print("y shape:", y.shape)

# =====================================
# ONE HOT ENCODING
# =====================================

NUM_CLASSES = 2

y_cat = to_categorical(y, num_classes=NUM_CLASSES)

print("\nOne-hot encoding complete ✅")

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain/Test Split Complete ✅")
print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# =====================================
# BUILD MODEL
# =====================================

print("\nBuilding BiLSTM model...")

model = Sequential()

model.add(
    Bidirectional(
        LSTM(
            64,
            return_sequences=False
        ),
        input_shape=(X.shape[1], X.shape[2])
    )
)

model.add(Dropout(0.3))

model.add(Dense(32, activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(NUM_CLASSES, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Built Successfully ✅")
model.summary()

# =====================================
# TRAIN MODEL
# =====================================

print("\nTraining started...\n")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=25,
    batch_size=32,
    verbose=1
)

print("\nTraining Complete ✅")

# =====================================
# EVALUATE MODEL
# =====================================

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("====================================")
print("FINAL RESULTS")
print("====================================")
print(f"Test Accuracy: {accuracy:.4f}")
print("====================================")

# Predictions

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))

# =====================================
# SAVE MODEL
# =====================================

model.save("feedback/final_bilstm_model.h5")

print("\n====================================")
print("Final BiLSTM Model Saved Successfully ✅")
print("Saved to: feedback/final_bilstm_model.h5")
print("====================================")