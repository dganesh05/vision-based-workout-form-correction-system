import os
import re
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# =====================================================
# PATHS
# =====================================================

MODEL_PATH = "feedback/final_bilstm_autoencoder_3d.keras"
SCALER_PATH = "feedback/golden_scaler.pkl"
DATA_DIR = "data/model_ready_reps"
SAVE_PATH = "feedback/final_prediction_results.csv"

print("Loading trained 3D model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully")

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)
print("Scaler loaded successfully")


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def extract_person_id(file_name):
    match = re.search(r"(IMG_\d+)", file_name)
    return match.group(1) if match else "Unknown"


def calculate_biomechanics(data):
    """
    Input shape: (41, 7, 3)
    Dataset-calibrated biomechanical proxies.
    """

    hip_y = np.mean(data[:, 0, 1])
    knee_y = np.mean(data[:, 1, 1])
    shoulder_y = np.mean(data[:, 3, 1])

    left_x = np.mean(data[:, 4, 0])
    right_x = np.mean(data[:, 5, 0])

    depth_score = abs(hip_y - knee_y)
    knee_angle = round(60 + depth_score * 200, 2)

    symmetry_score = round(abs(left_x - right_x) * 100, 2)

    torso_diff = abs(shoulder_y - hip_y)
    spine_angle = round(180 - torso_diff * 100, 2)

    return knee_angle, symmetry_score, spine_angle


def score_from_failure(failure_type, error, max_dev, bad_threshold):
    """
    Deterministic score mapping.
    No random scores.
    """

    if failure_type == "Great Rep ✅":
        score = 94 - (error * 80)

    elif failure_type == "Minor Technical Issue ⚠️":
        score = 84 - (error * 120)

    elif failure_type in ["Depth Failure ❌", "Knee Valgus ❌", "Forward Lean ❌"]:
        score = 72 - (error * 160)

    elif failure_type == "Movement Breakdown ❌":
        score = 55 - ((max_dev / bad_threshold) * 8)

    else:
        score = 70 - (error * 100)

    return max(0, min(100, round(score, 2)))


def determine_quality_from_score(score):
    if score >= 90:
        return "Advanced", "Excellent Squat ✅"
    elif score >= 75:
        return "Intermediate", "Good Squat ✅"
    elif score >= 60:
        return "Beginner+", "Needs Improvement ⚠️"
    else:
        return "Beginner", "High Injury Risk ❌"


# =====================================================
# LOAD + RESHAPE DATA
# =====================================================

model_data = []
raw_data_for_rules = []
file_names = []

files = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".npy") and "original data" in f.lower()
])

print(f"\nTotal ORIGINAL DATA files found: {len(files)}")

for file in files:
    path = os.path.join(DATA_DIR, file)

    try:
        arr = np.load(path)

        if arr.shape != (41, 7, 3):
            print(f"Skipping {file} -> unexpected shape {arr.shape}")
            continue

        raw_data_for_rules.append(arr)

        flat = arr.reshape(41, 21)
        scaled = scaler.transform(flat)

        model_data.append(scaled)
        file_names.append(file)

    except Exception as e:
        print(f"Error loading {file}: {e}")

X_test = np.array(model_data)
X_rules = np.array(raw_data_for_rules)

print("\nFinal Loaded Shape for Model:", X_test.shape)
print("Final Loaded Shape for Rules:", X_rules.shape)

if len(X_test) == 0:
    print("No valid files found.")
    exit()


# =====================================================
# MODEL PREDICTION
# =====================================================

print("\nRunning autoencoder prediction...")

X_pred = model.predict(X_test)

reconstruction_errors = np.mean(
    np.square(X_test - X_pred),
    axis=(1, 2)
)

max_deviations = np.max(
    np.abs(X_test - X_pred),
    axis=(1, 2)
)

print("\nSample Reconstruction Errors:")
print(reconstruction_errors[:10])


# =====================================================
# DYNAMIC THRESHOLDS
# =====================================================

great_rep_threshold = np.percentile(max_deviations, 25)
bad_form_threshold = np.percentile(max_deviations, 80)

print(f"\nGreat Rep Threshold: {great_rep_threshold:.4f}")
print(f"Bad Form Threshold: {bad_form_threshold:.4f}")


# =====================================================
# FINAL ANALYSIS
# =====================================================

results = []

for file_name, original_3d, error, max_dev in zip(
    file_names,
    X_rules,
    reconstruction_errors,
    max_deviations
):

    person_id = extract_person_id(file_name)

    knee_angle, symmetry_score, spine_angle = calculate_biomechanics(original_3d)

    # =================================================
    # CALIBRATED FAILURE LOGIC
    # =================================================

    if knee_angle > 65:
        failure_type = "Depth Failure ❌"
        coach_feedback = (
            f"Your squat depth is insufficient "
            f"(knee angle proxy: {knee_angle}°). "
            f"You are not reaching enough depth at the bottom position. "
            f"Squat deeper for stronger mechanics and better activation."
        )
        suggestion = "Go deeper during squat"

    elif symmetry_score > 1.8:
        failure_type = "Knee Valgus ❌"
        coach_feedback = (
            f"Left-right imbalance detected "
            f"(symmetry deviation: {symmetry_score}). "
            f"Your knees may be collapsing inward during the squat. "
            f"Push knees outward and improve balance."
        )
        suggestion = "Improve left-right balance and knee tracking"

    elif spine_angle < 171:
        failure_type = "Forward Lean ❌"
        coach_feedback = (
            f"Excessive forward lean detected "
            f"(spine angle proxy: {spine_angle}°). "
            f"Your torso is leaning too far forward. "
            f"Keep chest upright and brace your core."
        )
        suggestion = "Keep chest upright and reduce forward lean"

    elif max_dev > bad_form_threshold:
        failure_type = "Movement Breakdown ❌"
        coach_feedback = (
            f"Major movement deviation detected "
            f"(max deviation: {round(float(max_dev), 4)}). "
            f"Your squat differs significantly from the ideal movement pattern."
        )
        suggestion = "Reduce unnecessary movement and improve overall squat control"

    elif max_dev <= great_rep_threshold:
        failure_type = "Great Rep ✅"
        coach_feedback = (
            "Excellent squat mechanics. "
            "Movement closely matches the golden reference."
        )
        suggestion = "Maintain current squat consistency"

    else:
        failure_type = "Minor Technical Issue ⚠️"
        coach_feedback = (
            f"Some joint movement deviation detected "
            f"(max deviation: {round(float(max_dev), 4)}). "
            f"Good squat overall with small technical corrections needed."
        )
        suggestion = "Make minor adjustments to control and alignment"

    performance_score = score_from_failure(
        failure_type,
        error,
        max_dev,
        bad_form_threshold
    )

    athlete_level, squat_quality = determine_quality_from_score(performance_score)

    results.append({
        "Person ID": person_id,
        "Performance Score": performance_score,
        "Athlete Level": athlete_level,
        "Squat Quality": squat_quality,
        "Failure Type": failure_type,
        "Reconstruction Error": round(float(error), 6),
        "Max Deviation": round(float(max_dev), 4),
        "Knee Angle Proxy": knee_angle,
        "Symmetry Score": symmetry_score,
        "Spine Angle Proxy": spine_angle,
        "Coach Feedback": coach_feedback,
        "Improvement Suggestions": suggestion
    })


# =====================================================
# SAVE RESULTS
# =====================================================

df = pd.DataFrame(results)

df.to_csv(SAVE_PATH, index=False)

print("\n==================================================")
print("FINAL HYBRID SQUAT ANALYSIS COMPLETE")
print("==================================================")

print(df.head(30))

print(f"\nTotal Athlete Predictions Saved: {len(df)}")
print(f"Saved to: {SAVE_PATH}")

print("\nSystem Ready 🚀")