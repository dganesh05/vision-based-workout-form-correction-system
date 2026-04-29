# feedback/predict_autoencoder_quality.py

import os
import re
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# =====================================
# PATHS
# =====================================

MODEL_PATH = "feedback/final_bilstm_autoencoder.keras"
DATA_DIR = "data/model_ready_reps"
SAVE_PATH = "feedback/final_prediction_results.csv"

# =====================================
# LOAD MODEL
# =====================================

print("Loading model...")
model = load_model(MODEL_PATH, compile=False)

# =====================================
# HELPER FUNCTION
# Better extraction for Person ID + Rep Number
# =====================================

def extract_person_and_rep(file_name):
    """
    Handles examples like:

    original data__IMG_6991_angles__hole_22__label_0.npy
    original data__IMG_7005_3_angles__hole_18__label_0.npy
    golden reference__G002_45_angles__hole_44__label_1.npy
    """

    person_id = "Unknown"
    rep_number = "Unknown"

    # -----------------------------
    # CASE 1: IMG files
    # -----------------------------
    # Example:
    # IMG_6991
    # IMG_7005_3

    img_match = re.search(r"(IMG_\d+)(?:[_\-](\d+))?", file_name)

    if img_match:
        person_id = img_match.group(1)

        if img_match.group(2):
            rep_number = img_match.group(2)
        else:
            # fallback from hole number
            hole_match = re.search(r"hole[_\-](\d+)", file_name)
            if hole_match:
                rep_number = hole_match.group(1)

        return person_id, rep_number

    # -----------------------------
    # CASE 2: Golden reference files
    # -----------------------------
    # Example:
    # G002_45

    g_match = re.search(r"(G\d+)[_\-](\d+)", file_name)

    if g_match:
        person_id = g_match.group(1)
        rep_number = g_match.group(2)

        return person_id, rep_number

    return person_id, rep_number


# =====================================
# LOAD ONLY ORIGINAL DATA FILES
# =====================================

all_data = []
file_names = []
person_ids = []
rep_numbers = []

files = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".npy")
    and "original data" in f.lower()
])

print(f"Total ORIGINAL DATA files found: {len(files)}")

# debug preview
print("\nSample filenames:")
for f in files[:10]:
    print(f)

for file in files:
    file_path = os.path.join(DATA_DIR, file)

    try:
        x = np.load(file_path, allow_pickle=True)

        if x.shape != (41, 5):
            print(f"Skipping {file} -> unexpected shape {x.shape}")
            continue

        person_id, rep_number = extract_person_and_rep(file)

        all_data.append(x)
        file_names.append(file)
        person_ids.append(person_id)
        rep_numbers.append(rep_number)

    except Exception as e:
        print(f"Error reading {file}: {e}")

X_test = np.array(all_data)

print("\nFinal Loaded Shape:", X_test.shape)

# =====================================
# RUN PREDICTION
# =====================================

print("\nRunning squat quality prediction...")

X_pred = model.predict(X_test)

# =====================================
# RECONSTRUCTION ERROR
# =====================================

errors = np.mean(
    np.square(X_test - X_pred),
    axis=(1, 2)
)

print("\nSample Reconstruction Errors:")
print(errors[:10])

threshold = np.percentile(errors, 70)

print(f"\nDynamic Threshold: {threshold:.4f}")

# =====================================
# FINAL ATHLETE RESULTS
# =====================================

results = []

for file_name, person_id, rep_number, original, error in zip(
    file_names,
    person_ids,
    rep_numbers,
    X_test,
    errors
):

    avg_knee_angle = round(float(np.mean(original[:, 0])), 2)
    avg_hip_angle = round(float(np.mean(original[:, 1])), 2)
    avg_ankle_angle = round(float(np.mean(original[:, 2])), 2)
    avg_symmetry = round(float(np.mean(original[:, 3])), 2)
    avg_spine_angle = round(float(np.mean(original[:, 4])), 2)

    performance_score = max(
        0,
        round(100 - (error / threshold) * 40, 2)
    )

    if error <= threshold * 0.75:
        quality = "Excellent Squat ✅"
        level = "Advanced Athlete"
        coach_feedback = "Excellent squat mechanics with strong stability and depth."

    elif error <= threshold:
        quality = "Good Squat ✅"
        level = "Intermediate Athlete"
        coach_feedback = "Good squat overall with small areas for improvement."

    elif error <= threshold * 1.25:
        quality = "Needs Improvement ⚠️"
        level = "Beginner Athlete"
        coach_feedback = "Movement pattern shows noticeable deviation from ideal squat mechanics."

    else:
        quality = "Poor Squat ❌"
        level = "High Injury Risk"
        coach_feedback = "Significant squat form issues detected. Correct movement before increasing load."

    suggestions = []

    if avg_knee_angle > 130:
        suggestions.append("Go deeper during squat")

    if avg_spine_angle < 170:
        suggestions.append("Keep chest upright and reduce forward lean")

    if avg_symmetry > 12:
        suggestions.append("Improve left-right balance and knee tracking")

    if avg_ankle_angle < 90:
        suggestions.append("Improve ankle mobility and heel stability")

    if not suggestions:
        suggestions.append("Maintain current squat consistency")

    results.append({
        "Person ID": person_id,
        "Actual Rep Number": rep_number,
        "Video File": file_name,

        "Performance Score /100": performance_score,
        "Athlete Level": level,
        "Squat Quality": quality,
        "Reconstruction Error": round(float(error), 4),

        "Average Knee Angle": avg_knee_angle,
        "Average Hip Angle": avg_hip_angle,
        "Average Ankle Angle": avg_ankle_angle,
        "Average Symmetry Score": avg_symmetry,
        "Average Spine Angle": avg_spine_angle,

        "Coach Feedback": coach_feedback,
        "Improvement Suggestions": " | ".join(suggestions)
    })

df = pd.DataFrame(results)

# =====================================
# SAVE RESULTS
# =====================================

df.to_csv(SAVE_PATH, index=False)

print("\n==========================================")
print("FINAL ATHLETE PERFORMANCE ANALYSIS COMPLETE")
print("==========================================")

print(df.head(20))

print(f"\nTotal Athlete Predictions Saved: {len(df)}")
print(f"Saved to: {SAVE_PATH}")

print("\nSystem Ready 🚀")