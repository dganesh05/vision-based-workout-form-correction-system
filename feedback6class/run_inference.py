from pathlib import Path
import numpy as np
import torch
import pandas as pd
import re

from feedback6class.inference import SquatInferenceEngine


# ─────────────────────────────────────────────
# RULE-BASED COACH
# ─────────────────────────────────────────────

def analyze_non_golden(angles: np.ndarray):

    knee_l = angles[:, 0]
    knee_r = angles[:, 1]
    hip = angles[:, 2]
    spine = angles[:, 3]
    symmetry = angles[:, 4]

    feedback = []

    if np.mean(spine) > 8:
        feedback.append("❗ Excess forward lean — keep chest up")

    if np.mean(symmetry) > 4:
        feedback.append("❗ Left/right imbalance — distribute weight evenly")

    if np.max(hip) < 2.0:
        feedback.append("⚠️ Insufficient depth — go lower")

    if np.std(knee_l) > 2 or np.std(knee_r) > 2:
        feedback.append("⚠️ Knee instability — control descent")

    if not feedback:
        feedback.append("⚠️ Non-golden classification but no major issues detected")

    return feedback


# ─────────────────────────────────────────────
# LOAD SAMPLE
# ─────────────────────────────────────────────

def load_sample(joints_path: Path, angles_path: Path):

    joints_np = np.load(joints_path).astype(np.float32)
    df = pd.read_csv(angles_path)

    angles_np_full = df[[
        "Right_Knee_Angle",
        "Left_Knee_Angle",
        "Right_Hip_Angle",
        "Left_Hip_Angle",
        "Spine_Lean_Angle"
    ]].values.astype(np.float32)

    T = joints_np.shape[0]

    # ─────────────────────────────────────────────
    # FIX: align CSV (shift by 1 frame)
    # np[t] == csv[t + 1]
    # ─────────────────────────────────────────────
    angles_np = angles_np_full[1:]  # drop first CSV frame

    # ensure same length as joints
    min_T = min(T, len(angles_np))

    joints_np = joints_np[:min_T]
    angles_np = angles_np[:min_T]

    joints = torch.from_numpy(joints_np).unsqueeze(0)
    angles = torch.from_numpy(angles_np).unsqueeze(0)

    lengths = torch.tensor([min_T])
    mask = torch.ones((1, min_T), dtype=torch.bool)

    return joints, angles, lengths, mask, angles_np


# ─────────────────────────────────────────────
# FILE MATCHING KEY (FIXED LOGIC)
# ─────────────────────────────────────────────

def normalize_key(name: str) -> str:
    name = name.replace("_angles", "")
    name = re.sub(r"_aug_\d+", "", name)
    name = name.replace(".npy", "").replace(".csv", "")
    return name


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    data_folder = Path(
        "model_ready_reps-20260429T143559Z-3-001/model_ready_reps/"
    )

    model_paths = [
        Path("runs/squat_kfold/fold_0/best_model.pt"),
        Path("runs/squat_kfold/fold_1/best_model.pt"),
        Path("runs/squat_kfold/fold_2/best_model.pt"),
        Path("runs/squat_kfold/fold_3/best_model.pt"),
        Path("runs/squat_kfold/fold_4/best_model.pt"),
    ]

    engine = SquatInferenceEngine(model_paths=model_paths)

    results = []
    missing = 0

    # ─────────────────────────────────────────────
    # PRE-COLLECT CSV MAP (FAST LOOKUP FIX)
    # ─────────────────────────────────────────────

    csv_map = {
        normalize_key(p.stem): p
        for p in data_folder.glob("*.csv")
    }

    npy_files = sorted(data_folder.glob("*.npy"))

    for joints_file in npy_files:

        key = normalize_key(joints_file.stem)

        if key not in csv_map:
            missing += 1
            continue

        csv_file = csv_map[key]

        try:
            joints, angles, lengths, mask, angles_np = load_sample(
                joints_file,
                csv_file
            )

            result = engine.analyze((joints, angles, lengths, mask))

            pred_name = result["prediction"]["name"]
            conf = result["prediction"]["confidence"]

            print(f"\n================ {joints_file.name} ================")
            print("Class:", pred_name)
            print("Confidence:", conf)

            if pred_name == "non_golden":
                print("\n🚨 FORM ISSUES DETECTED:")
                for f in analyze_non_golden(angles_np):
                    print("-", f)
            else:
                print("✅ Good form — no extra analysis needed")

            results.append((pred_name, conf))

        except Exception as e:
            print("❌ Failed:", joints_file.name)
            print(e)

    # ─────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────

    if results:
        acc_like = sum(1 for r in results if r[0] == "non_golden") / len(results)
        avg_conf = sum(r[1] for r in results) / len(results)

        print("\n🔥 FINAL SUMMARY")
        print("Total processed:", len(results))
        print("Missing CSVs:", missing)
        print("Avg confidence:", round(avg_conf, 4))
        print("Non-golden ratio:", round(acc_like, 4))

    print("\n🔥 DONE.")