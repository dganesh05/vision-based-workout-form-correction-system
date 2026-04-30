import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =====================================
# LOAD TRAINING DATA
# =====================================

file_path = "feedback/training_data.csv"
df = pd.read_csv(file_path)

print("Dataset Loaded ✅")
print(df.head())
print("\nLabel Counts:")
print(df["label"].value_counts())


# =====================================
# SELECT FEATURES
# =====================================

features = [
    "avg_knee",
    "avg_hip",
    "spine_lean",
    "knee_symmetry"
]

X = df[features]
y = df["label"]


# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain/Test Split Complete ✅")
print("Train size:", len(X_train))
print("Test size:", len(X_test))


# =====================================
# TRAIN MODEL
# =====================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Complete ✅")


# =====================================
# PREDICTIONS
# =====================================

y_pred = model.predict(X_test)


# =====================================
# EVALUATION
# =====================================

accuracy = accuracy_score(y_test, y_pred)

print("\n====================================")
print("MODEL RESULTS")
print("====================================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n====================================")
print("Random Forest Baseline Complete ✅")
print("====================================")