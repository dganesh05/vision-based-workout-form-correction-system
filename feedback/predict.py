# feedback/predict.py

import joblib
import pandas as pd


# =====================================
# LOAD TRAINED MODEL
# =====================================

model_path = "feedback/squat_model.pkl"
model = joblib.load(model_path)

print("Model Loaded Successfully ✅")


# =====================================
# LABEL → STATUS CONVERSION
# =====================================

def convert_label_to_status(label):
    if label == 0:
        return "Excellent Squat ✅"

    elif label == 1:
        return "Good Squat + Minor Improvements ⚠️"

    elif label == 2:
        return "Needs Major Improvement ❌"

    return "Unknown"


# =====================================
# FEEDBACK ENGINE
# (Explain WHY)
# =====================================

def generate_feedback(avg_knee, avg_hip, spine_lean, knee_symmetry):
    feedback = []
    improvements = []

    # Squat depth
    if avg_knee > 145:
        improvements.append("Squat Depth")
        feedback.append(
            "Go deeper by bending your knees more and lowering your hips closer to parallel."
        )
    elif 130 <= avg_knee <= 145:
        improvements.append("Depth Refinement")
        feedback.append(
            "Depth is decent, but going slightly deeper would improve squat quality."
        )

    # Spine posture
    if spine_lean < 145:
        improvements.append("Torso Posture")
        feedback.append(
            "Keep your chest up and reduce excessive forward lean."
        )
    elif 145 <= spine_lean < 160:
        improvements.append("Chest Position")
        feedback.append(
            "Keeping your chest slightly higher will improve stability."
        )

    # Hip control
    if avg_hip < 90:
        improvements.append("Hip Positioning")
        feedback.append(
            "Sit back into your hips more for stronger squat mechanics."
        )
    elif 90 <= avg_hip < 125:
        improvements.append("Minor Hip Adjustment")
        feedback.append(
            "A slight hip adjustment could improve balance and control."
        )

    # Balance
    if knee_symmetry > 25:
        improvements.append("Balance / Symmetry")
        feedback.append(
            "Noticeable left-right imbalance detected. Try distributing weight evenly."
        )
    elif 15 <= knee_symmetry <= 25:
        improvements.append("Minor Balance Improvement")
        feedback.append(
            "Slight imbalance detected. Try keeping both legs more even."
        )

    if not feedback:
        feedback.append(
            "Excellent squat form. Strong depth, posture, and balance."
        )

    return improvements, feedback


# =====================================
# USER INPUT
# (Replace these values with real user input)
# =====================================

sample_input = {
    "avg_knee_angle": 132,
    "avg_hip_angle": 138,
    "spine_lean": 168,
    "knee_symmetry": 8
}

input_df = pd.DataFrame([sample_input])


# =====================================
# MODEL PREDICTION
# =====================================

prediction = model.predict(input_df)[0]
status = convert_label_to_status(prediction)


# =====================================
# FEEDBACK GENERATION
# =====================================

improvements, feedback = generate_feedback(
    sample_input["avg_knee_angle"],
    sample_input["avg_hip_angle"],
    sample_input["spine_lean"],
    sample_input["knee_symmetry"]
)


# =====================================
# FINAL USER OUTPUT
# =====================================

print("\n====================================")
print("FINAL USER SQUAT ANALYSIS")
print("====================================\n")

print("Input Metrics:")
print(f"Knee Angle     : {sample_input['avg_knee_angle']}")
print(f"Hip Angle      : {sample_input['avg_hip_angle']}")
print(f"Spine Lean     : {sample_input['spine_lean']}")
print(f"Knee Symmetry  : {sample_input['knee_symmetry']}")

print("\nPrediction:")
print(status)

print("\nAreas to Improve:")
if improvements:
    for item in improvements:
        print(f"- {item}")
else:
    print("- None")

print("\nPersonalized Feedback:")
for item in feedback:
    print(f"- {item}")

print("\n====================================")
print("Personal Squat Coach Ready ✅")
print("====================================")