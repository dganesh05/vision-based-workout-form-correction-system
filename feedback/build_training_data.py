import os
import pandas as pd


def detect_rep_bottom_frames(df):
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


def process_file(file_path, label):
    df = pd.read_csv(file_path)
    rep_bottoms = detect_rep_bottom_frames(df)

    rows = []

    for rep_idx, row in enumerate(rep_bottoms, start=1):
        right_knee = row["Right_Knee"]
        left_knee = row["Left_Knee"]
        right_hip = row["Right_Hip"]
        left_hip = row["Left_Hip"]
        spine_lean = row["Spine_Lean"]

        avg_knee = (right_knee + left_knee) / 2
        avg_hip = (right_hip + left_hip) / 2
        knee_symmetry = abs(right_knee - left_knee)

        rows.append({
            "file_name": os.path.basename(file_path),
            "rep_number": rep_idx,
            "bottom_frame": int(row["Frame"]),
            "right_knee": round(right_knee, 2),
            "left_knee": round(left_knee, 2),
            "avg_knee": round(avg_knee, 2),
            "right_hip": round(right_hip, 2),
            "left_hip": round(left_hip, 2),
            "avg_hip": round(avg_hip, 2),
            "spine_lean": round(spine_lean, 2),
            "knee_symmetry": round(knee_symmetry, 2),
            "label": label
        })

    return rows


all_rows = []

folders = {
    "data/angles_csv/correct": 0,
    "data/angles_csv/incorrect": 1
}

for folder_path, label in folders.items():
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing: {file_path}")

            rows = process_file(file_path, label)
            all_rows.extend(rows)

training_df = pd.DataFrame(all_rows)

output_path = "feedback/training_data.csv"
training_df.to_csv(output_path, index=False)

print("\nTraining CSV created ✅")
print(f"Saved to: {output_path}")
print(f"Total reps: {len(training_df)}")
print("\nLabel counts:")
print(training_df["label"].value_counts())
print("\nPreview:")
print(training_df.head())