# feedback/train_model_v2.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib


# =====================================
# LOAD FINAL COMBINED DATASET
# =====================================

file_path = "feedback/training_data_v2.csv"
df = pd.read_csv(file_path)

print("Dataset Loaded ✅")
print(df.head())

print("\n====================================")
print("LABEL COUNTS")
print("====================================")
print(df["label"].value_counts())
print("====================================")


# =====================================
# IMPORTANT:
# NO DATA LEAKAGE
#
# DO NOT USE:
# quality_score
# status
#
# because they help create label
#
# USE ONLY REAL BIOMECHANICS FEATURES
# =====================================

features = [
    "avg_knee_angle",
    "avg_hip_angle",
    "spine_lean",
    "knee_symmetry"
]

X = df[features]
y = df["label"]


# =====================================
# TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain/Test Split Complete ✅")
print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")


# =====================================
# RANDOM FOREST MODEL
# =====================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
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
print("FINAL MODEL RESULTS")
print("====================================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =====================================
# FEATURE IMPORTANCE
# VERY STRONG FOR PRESENTATION
# =====================================

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\n====================================")
print("FEATURE IMPORTANCE")
print("====================================")
print(importance_df)
print("====================================")


# =====================================
# SAVE FINAL MODEL
# =====================================

model_path = "feedback/squat_model.pkl"
joblib.dump(model, model_path)

print("\n====================================")
print("FINAL MODEL SAVED SUCCESSFULLY ✅")
print("Saved to:", model_path)
print("====================================")