# feedback/compute_angles.py

import json
import math
import os
import pandas as pd

# ====================================
# CONFIG
# ====================================

# Change this depending on folder you want to test
data_folder = "data/correct_data"
# data_folder = "data/incorrect_data"

output_file = "feedback/final_features.csv"


# ====================================
# ANGLE CALCULATION
# ====================================

def calculate_angle(a, b, c):
    """
    Calculate angle using 3 points:
    angle ABC
    """

    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]

    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    cos_angle = max(min(cos_angle, 1), -1)

    return math.degrees(math.acos(cos_angle))


# ====================================
# REP DETECTION
# ====================================

def detect_rep_bottom_frames(json_path):
    """
    Detect real squat reps using:

    1. Local minimum knee angle
    2. Knee angle threshold
    3. Minimum frame gap
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    all_frames = []

    for frame_idx, frame_data in enumerate(data):
        if not frame_data["people"]:
            continue

        keypoints = frame_data["people"][0]["keypoints"]

        joints = {
            item["joint_id"]: (item["x"], item["y"])
            for item in keypoints
        }

        try:
            left_hip = joints[11]
            right_hip = joints[12]

            left_knee = joints[13]
            right_knee = joints[14]

            left_ankle = joints[15]
            right_ankle = joints[16]

            left_knee_angle = calculate_angle(
                left_hip, left_knee, left_ankle
            )

            right_knee_angle = calculate_angle(
                right_hip, right_knee, right_ankle
            )

            avg_knee_angle = (
                left_knee_angle + right_knee_angle
            ) / 2

            all_frames.append({
                "frame_index": frame_idx,
                "avg_knee_angle": avg_knee_angle,
                "joints": joints
            })

        except KeyError:
            continue

    rep_bottom_frames = []
    last_selected_frame = -999

    for i in range(1, len(all_frames) - 1):
        prev_angle = all_frames[i - 1]["avg_knee_angle"]
        curr_angle = all_frames[i]["avg_knee_angle"]
        next_angle = all_frames[i + 1]["avg_knee_angle"]

        if (
            curr_angle < prev_angle
            and curr_angle < next_angle
            and curr_angle < 150
            and (i - last_selected_frame) > 15
        ):
            rep_bottom_frames.append(all_frames[i])
            last_selected_frame = i

    return rep_bottom_frames


# ====================================
# FEATURE EXTRACTION ONLY
# NO LABEL LOGIC HERE
# ====================================

def extract_single_rep_features(joints, person_name, rep_number):
    """
    ONLY extract biomechanics features

    No hardcoded label logic
    No quality score
    No fake ML

    Model should learn from real data
    """

    try:
        left_shoulder = joints[5]
        right_shoulder = joints[6]

        left_hip = joints[11]
        right_hip = joints[12]

        left_knee = joints[13]
        right_knee = joints[14]

        left_ankle = joints[15]
        right_ankle = joints[16]

        # ====================================
        # KNEE ANGLES
        # ====================================

        left_knee_angle = calculate_angle(
            left_hip, left_knee, left_ankle
        )

        right_knee_angle = calculate_angle(
            right_hip, right_knee, right_ankle
        )

        avg_knee_angle = (
            left_knee_angle + right_knee_angle
        ) / 2

        # ====================================
        # HIP ANGLES
        # ====================================

        left_hip_angle = calculate_angle(
            left_shoulder, left_hip, left_knee
        )

        right_hip_angle = calculate_angle(
            right_shoulder, right_hip, right_knee
        )

        avg_hip_angle = (
            left_hip_angle + right_hip_angle
        ) / 2

        # ====================================
        # LEFT-RIGHT BALANCE
        # ====================================

        symmetry = abs(
            left_knee_angle - right_knee_angle
        )

        return {
            "person_file": person_name,
            "rep_number": rep_number,
            "avg_knee_angle": round(avg_knee_angle, 2),
            "avg_hip_angle": round(avg_hip_angle, 2),
            "symmetry": round(symmetry, 2)
        }

    except KeyError:
        return None


# ====================================
# MAIN PROCESS
# ====================================

all_results = []

for file_name in os.listdir(data_folder):
    if file_name.endswith(".json"):
        file_path = os.path.join(data_folder, file_name)

        print(f"\nProcessing person: {file_name}")

        rep_frames = detect_rep_bottom_frames(file_path)

        print(f"Detected REAL squat reps: {len(rep_frames)}")

        for rep_idx, rep_frame in enumerate(rep_frames, start=1):
            result = extract_single_rep_features(
                rep_frame["joints"],
                file_name,
                rep_idx
            )

            if result:
                all_results.append(result)

df = pd.DataFrame(all_results)

df.to_csv(
    output_file,
    index=False
)

print("\n====================================")
print(f"Total real squat reps processed: {len(df)}")
print("Feature Extraction Complete ✅")
print("Saved to:", output_file)
print("====================================\n")

print(df.head())