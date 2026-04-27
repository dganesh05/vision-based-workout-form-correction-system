import pandas as pd


# =====================================
# LOAD FINAL FEEDBACK RESULTS
# =====================================

file_path = "feedback/final_feedback_results.csv"
df = pd.read_csv(file_path)

print("Feedback Results Loaded ✅")
print(df.head())


# =====================================
# CONVERT STATUS → LABEL
# =====================================

def convert_status_to_label(status):
    if "Excellent Squat" in status:
        return 0

    elif "Good Squat + Minor Improvements" in status:
        return 1

    elif "Needs Major Improvement" in status:
        return 2

    return None


df["label"] = df["status"].apply(convert_status_to_label)


# =====================================
# SELECT FINAL TRAINING COLUMNS
# =====================================

training_df = df[
    [
        "file_name",
        "rep_number",
        "bottom_frame",
        "avg_knee_angle",
        "avg_hip_angle",
        "spine_lean",
        "knee_symmetry",
        "quality_score",
        "status",
        "label"
    ]
].copy()


# =====================================
# SAVE NEW DATASET
# =====================================

output_path = "feedback/training_data_v2.csv"
training_df.to_csv(output_path, index=False)

print("\n====================================")
print("training_data_v2.csv created ✅")
print("Saved to:", output_path)
print("====================================\n")

print("Label Counts:")
print(training_df["label"].value_counts())

print("\nPreview:")
print(training_df.head())