import os
import pandas as pd

DATA_FOLDER = "data/angles_csv"

golden_files = []

for file_name in os.listdir(DATA_FOLDER):
    if "golden reference" in file_name.lower() and file_name.endswith(".csv"):
        golden_files.append(
            os.path.join(DATA_FOLDER, file_name)
        )

print("Golden reference files found:")
for f in golden_files:
    print(f)

all_data = []

for file_path in golden_files:
    df = pd.read_csv(file_path)
    df = df.dropna()

    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

baseline = {
    "Right_Knee_mean": combined_df["Right_Knee"].mean(),
    "Left_Knee_mean": combined_df["Left_Knee"].mean(),
    "Right_Hip_mean": combined_df["Right_Hip"].mean(),
    "Left_Hip_mean": combined_df["Left_Hip"].mean(),
    "Spine_Lean_mean": combined_df["Spine_Lean"].mean()
}

baseline_df = pd.DataFrame([baseline])

baseline_df.to_csv(
    "feedback/golden_reference_baseline.csv",
    index=False
)

print("\n====================================")
print("Golden Reference Baseline Created ✅")
print("Saved to: feedback/golden_reference_baseline.csv")
print("====================================")

print("\nBaseline Values:")
print(baseline_df)