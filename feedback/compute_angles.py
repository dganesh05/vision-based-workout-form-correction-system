import json
import math
import pandas as pd


def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]

    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    cos_angle = max(min(cos_angle, 1), -1)

    angle = math.degrees(math.acos(cos_angle))
    return angle


def extract_features(json_path, label):
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []

    for frame_data in data:
        if not frame_data["people"]:
            continue

        keypoints = frame_data["people"][0]["keypoints"]

        joints = {
            item["joint_id"]: (item["x"], item["y"])
            for item in keypoints
        }

        try:
            # LEFT SIDE
            left_shoulder = joints[5]
            left_hip = joints[11]
            left_knee = joints[13]
            left_ankle = joints[15]

            # RIGHT SIDE
            right_shoulder = joints[6]
            right_hip = joints[12]
            right_knee = joints[14]
            right_ankle = joints[16]

            # Knee angles
            left_knee_angle = calculate_angle(
                left_hip, left_knee, left_ankle
            )

            right_knee_angle = calculate_angle(
                right_hip, right_knee, right_ankle
            )

            # Hip angles
            left_hip_angle = calculate_angle(
                left_shoulder, left_hip, left_knee
            )

            right_hip_angle = calculate_angle(
                right_shoulder, right_hip, right_knee
            )

            # Depth
            avg_hip_y = (left_hip[1] + right_hip[1]) / 2
            avg_knee_y = (left_knee[1] + right_knee[1]) / 2
            depth = avg_knee_y - avg_hip_y

            # Symmetry
            symmetry = abs(left_knee_angle - right_knee_angle)

            rows.append({
                "left_knee_angle": left_knee_angle,
                "right_knee_angle": right_knee_angle,
                "left_hip_angle": left_hip_angle,
                "right_hip_angle": right_hip_angle,
                "depth": depth,
                "symmetry": symmetry,
                "label": label
            })

        except KeyError:
            continue

    return rows


# Correct squat = 0
correct_rows = extract_features(
    "data/correct__5a10e657-8f8d-4685-a977-d60016b4e7db_keypoints.json",
    label=0
)

# Incorrect squat = 1
incorrect_rows = extract_features(
    "data/incorrect__Copy of IMG_9752_keypoints.json",
    label=1
)

all_rows = correct_rows + incorrect_rows

df = pd.DataFrame(all_rows)

df.to_csv("feedback/final_features.csv", index=False)

print(df.head())
print(f"\nTotal samples: {len(df)}")
print("\nfinal_features.csv created successfully ✅")