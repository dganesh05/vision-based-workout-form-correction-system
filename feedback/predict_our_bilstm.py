# feedback/predict_our_bilstm.py

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# =====================================
# CONFIG
# =====================================

MODEL_PATH = "feedback/final_bilstm_model.h5"
DATA_FOLDER = "data/angles_csv"

SEQUENCE_LENGTH = 30

FEATURE_COLUMNS = [
    "Right_Knee",
    "Left_Knee",
    "Right_Hip",
    "Left_Hip",
    "Spine_Lean"
]

# =====================================
# LOAD MODEL
# =====================================

print("Loading Final BiLSTM Model...")

model = load_model(MODEL_PATH)

print("Model Loaded Successfully ✅")


# =====================================
# FEEDBACK FUNCTION
# =====================================

def generate_feedback(df):
    """
    Model prediction = BiLSTM

    This function = Explainable Feedback Layer

    We use reliable squat biomechanics
    instead of forcing exact trainer matching.
    """

    # ---------------------------------
    # Average User Angles
    # ---------------------------------

    avg_right_knee = df["Right_Knee"].mean()
    avg_left_knee = df["Left_Knee"].mean()
    avg_knee = (avg_right_knee + avg_left_knee) / 2

    avg_right_hip = df["Right_Hip"].mean()
    avg_left_hip = df["Left_Hip"].mean()
    avg_hip = (avg_right_hip + avg_left_hip) / 2

    avg_spine = df["Spine_Lean"].mean()

    issues = []
    suggestions = []

    # =====================================
    # BIOMECHANICAL THRESHOLDS
    # =====================================

    # ---------------------------------
    # 1. Squat Depth (Knee Angle)
    #
    # Good squat depth often falls
    # around 80°–120°
    # Higher angle = shallow squat
    # ---------------------------------

    if avg_knee > 125:
        issues.append(
            "Squat depth is too shallow"
        )
        suggestions.append(
            "Go deeper and improve knee flexion during descent"
        )

    elif avg_knee < 75:
        issues.append(
            "Squat depth may be excessively deep"
        )
        suggestions.append(
            "Maintain control and avoid unnecessary over-compression"
        )

    # ---------------------------------
    # 2. Hip Hinge
    #
    # Higher hip angle often indicates
    # poor posterior chain engagement
    # ---------------------------------

    if avg_hip > 135:
        issues.append(
            "Hip hinge mechanics need improvement"
        )
        suggestions.append(
            "Sit back more naturally and improve hip control"
        )

    # ---------------------------------
    # 3. Spine / Torso Posture
    #
    # Lower spine angle may indicate
    # excessive forward lean
    # ---------------------------------

    if avg_spine < 165:
        issues.append(
            "Excessive forward torso lean detected"
        )
        suggestions.append(
            "Keep chest upright and maintain neutral spine posture"
        )

    # ---------------------------------
    # No major issues found
    # ---------------------------------

    if len(issues) == 0:
        issues.append(
            "Movement pattern is stable and biomechanically strong"
        )
        suggestions.append(
            "Maintain consistency and continue strong squat mechanics"
        )

    return {
        "avg_knee": round(avg_knee, 2),
        "avg_hip": round(avg_hip, 2),
        "avg_spine": round(avg_spine, 2),
        "issues": issues,
        "suggestions": suggestions
    }


# =====================================
# RUN PREDICTION ON ALL FILES
# =====================================

results = []

print("\nRunning full squat analysis on all files...\n")

for file_name in os.listdir(DATA_FOLDER):

    if not file_name.endswith(".csv"):
        continue

    file_path = os.path.join(DATA_FOLDER, file_name)

    try:
        df = pd.read_csv(file_path)
        df = df.dropna()

        if len(df) < SEQUENCE_LENGTH:
            print(f"Skipped (too short): {file_name}")
            continue

        # =====================================
        # BUILD SEQUENCE INPUT
        # =====================================

        features = df[FEATURE_COLUMNS].values

        X_pred = []

        for i in range(len(df) - SEQUENCE_LENGTH + 1):
            seq = features[i:i + SEQUENCE_LENGTH]
            X_pred.append(seq)

        X_pred = np.array(X_pred)

        # =====================================
        # MODEL PREDICTION
        # =====================================

        pred_probs = model.predict(
            X_pred,
            verbose=0
        )

        pred_classes = np.argmax(
            pred_probs,
            axis=1
        )

        # Majority vote
        final_prediction = np.bincount(
            pred_classes
        ).argmax()

        confidence = np.max(
            np.mean(pred_probs, axis=0)
        )

        # =====================================
        # FEEDBACK EXPLANATION
        # =====================================

        feedback = generate_feedback(df)

        print("\n====================================")
        print(f"FILE: {file_name}")
        print("====================================")

        if final_prediction == 0:
            prediction_text = "Golden Reference Quality ✅"
        else:
            prediction_text = "Needs Improvement ⚠️"

        print(f"\nPrediction: {prediction_text}")
        print(f"Confidence: {round(confidence * 100, 2)}%")

        print("\nAverage Angles:")
        print(f"- Knee Angle  : {feedback['avg_knee']}")
        print(f"- Hip Angle   : {feedback['avg_hip']}")
        print(f"- Spine Lean  : {feedback['avg_spine']}")

        print("\nWhat Needs Improvement:")

        for issue in feedback["issues"]:
            print(f"- {issue}")

        print("\nSuggested Fixes:")

        for suggestion in feedback["suggestions"]:
            print(f"- {suggestion}")

        results.append({
            "file_name": file_name,
            "prediction": prediction_text,
            "confidence_percent": round(confidence * 100, 2),
            "avg_knee": feedback["avg_knee"],
            "avg_hip": feedback["avg_hip"],
            "avg_spine": feedback["avg_spine"],
            "issues": " | ".join(feedback["issues"])
        })

    except Exception as e:
        print(f"Error in {file_name}: {e}")


# =====================================
# SAVE RESULTS
# =====================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "feedback/final_prediction_results.csv",
    index=False
)

print("\n====================================")
print("FINAL PREDICTION ANALYSIS COMPLETE ✅")
print("Saved to: feedback/final_prediction_results.csv")
print("====================================")

print("\nNow you can manually verify model quality 🚀")