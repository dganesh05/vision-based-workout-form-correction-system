import os
import pandas as pd


# =====================================================
# REP EVALUATION SYSTEM (Biomechanics Based)
# =====================================================

def evaluate_rep(right_knee, left_knee, right_hip, left_hip, spine_lean):
    """
    Final Professional Feedback System

    Output:
    - Quality Score (/100)
    - Status
    - Personalized Feedback
    """

    avg_knee = (right_knee + left_knee) / 2
    avg_hip = (right_hip + left_hip) / 2
    knee_symmetry = abs(right_knee - left_knee)

    score = 100
    feedback = []
    improvements = []

    # =====================================================
    # RULE 1 — SQUAT DEPTH (MOST IMPORTANT)
    # =====================================================
    # Ideal deep squat:
    # 90–130

    if avg_knee > 145:
        score -= 25
        improvements.append("Squat Depth")
        feedback.append(
            "Go deeper by bending your knees more and lowering your hips closer to parallel."
        )

    elif 130 <= avg_knee <= 145:
        score -= 8
        improvements.append("Depth Refinement")
        feedback.append(
            "Depth is decent, but going slightly deeper would improve overall squat quality."
        )

    # =====================================================
    # RULE 2 — SPINE / TORSO POSITION
    # =====================================================

    if spine_lean < 145:
        score -= 20
        improvements.append("Torso Posture")
        feedback.append(
            "Keep your chest up and reduce excessive forward lean for safer squat mechanics."
        )

    elif 145 <= spine_lean < 160:
        score -= 5
        improvements.append("Chest Position")
        feedback.append(
            "Torso position is good, but keeping your chest slightly higher will improve stability."
        )

    # =====================================================
    # RULE 3 — HIP CONTROL
    # =====================================================
    # Soft correction only

    if avg_hip < 90:
        score -= 8
        improvements.append("Hip Positioning")
        feedback.append(
            "Sit back into your hips more for better control and stronger squat mechanics."
        )

    elif 90 <= avg_hip < 125:
        score -= 4
        improvements.append("Minor Hip Adjustment")
        feedback.append(
            "Good squat overall, but a slight hip adjustment could improve balance and movement quality."
        )

    # =====================================================
    # RULE 4 — LEFT/RIGHT BALANCE
    # =====================================================

    if knee_symmetry > 25:
        score -= 10
        improvements.append("Balance / Symmetry")
        feedback.append(
            "Noticeable left-right imbalance detected. Try distributing your weight more evenly."
        )

    elif 15 <= knee_symmetry <= 25:
        score -= 5
        improvements.append("Minor Balance Improvement")
        feedback.append(
            "Slight imbalance detected. Try keeping both legs more even during the rep."
        )

    # =====================================================
    # FINAL SCORE + STATUS
    # =====================================================

    score = max(0, score)

    if score >= 90:
        status = "Excellent Squat ✅"

    elif 75 <= score < 90:
        status = "Good Squat + Minor Improvements ⚠️"

    else:
        status = "Needs Major Improvement ❌"

    if not feedback:
        feedback.append(
            "Excellent squat form. Strong depth, posture, and balance."
        )

    return {
        "avg_knee_angle": round(avg_knee, 2),
        "avg_hip_angle": round(avg_hip, 2),
        "spine_lean": round(spine_lean, 2),
        "knee_symmetry": round(knee_symmetry, 2),
        "quality_score": score,
        "status": status,
        "improvements": ", ".join(improvements) if improvements else "None",
        "feedback": " | ".join(feedback)
    }


# =====================================================
# DETECT TRUE SQUAT REPS
# =====================================================

def detect_rep_bottom_frames(df):
    """
    Detect ONLY real squat reps using:

    - local minimum knee angle
    - enough frame gap
    - proper squat threshold
    """

    df = df.copy()
    df["avg_knee"] = (df["Right_Knee"] + df["Left_Knee"]) / 2

    rep_frames = []
    last_selected_frame = -999

    for i in range(1, len(df) - 1):
        prev_angle = df.iloc[i - 1]["avg_knee"]
        curr_angle = df.iloc[i]["avg_knee"]
        next_angle = df.iloc[i + 1]["avg_knee"]

        current_frame = df.iloc[i]["Frame"]

        if (
            curr_angle < prev_angle
            and curr_angle < next_angle
            and curr_angle < 145
            and (current_frame - last_selected_frame) > 15
        ):
            rep_frames.append(df.iloc[i])
            last_selected_frame = current_frame

    return rep_frames


# =====================================================
# PROCESS SINGLE FILE
# =====================================================

def process_single_file(file_path):
    df = pd.read_csv(file_path)

    rep_bottoms = detect_rep_bottom_frames(df)

    results = []

    for rep_idx, row in enumerate(rep_bottoms, start=1):
        result = evaluate_rep(
            row["Right_Knee"],
            row["Left_Knee"],
            row["Right_Hip"],
            row["Left_Hip"],
            row["Spine_Lean"]
        )

        results.append({
            "file_name": os.path.basename(file_path),
            "rep_number": rep_idx,
            "bottom_frame": int(row["Frame"]),
            **result
        })

    return pd.DataFrame(results)


# =====================================================
# MAIN
# =====================================================

all_results = []

# FIRST TEST:
test_folder = "data/angles_csv/incorrect"

# LATER CHANGE TO:
# test_folder = "data/angles_csv/incorrect"

for file_name in os.listdir(test_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(test_folder, file_name)

        print(f"\nProcessing: {file_name}")

        result_df = process_single_file(file_path)

        if not result_df.empty:
            print(f"Detected REAL squat reps: {len(result_df)}")
            all_results.append(result_df)

if not all_results:
    print("\nNo valid squat reps detected.")
    exit()

final_df = pd.concat(all_results, ignore_index=True)

output_path = "feedback/final_feedback_results.csv"
final_df.to_csv(output_path, index=False)

print("\n====================================")
print("FINAL REP-BY-REP FEEDBACK SYSTEM READY ✅")
print("Saved to:", output_path)
print("====================================\n")

print("\n========== FULL RESULTS ==========\n")

for file_name in final_df["file_name"].unique():
    print(f"\nPerson/File: {file_name}")

    person_data = final_df[
        final_df["file_name"] == file_name
    ]

    for _, row in person_data.iterrows():
        print(
            f"Rep {row['rep_number']} | "
            f"Frame: {row['bottom_frame']} | "
            f"Score: {row['quality_score']}/100 | "
            f"{row['status']}"
        )

        print(
            f"Metrics → "
            f"Knee: {row['avg_knee_angle']} | "
            f"Hip: {row['avg_hip_angle']} | "
            f"Spine: {row['spine_lean']} | "
            f"Symmetry: {row['knee_symmetry']}"
        )

        print(f"Areas to Improve → {row['improvements']}")
        print(f"Feedback → {row['feedback']}")
        print("-" * 80)

print("\n==========================")
print("FINAL STATUS COUNTS")
print("==========================")
print(final_df["status"].value_counts())
print("==========================\n")