import argparse
import json
import math
import os
from pathlib import Path

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

    return math.degrees(math.acos(cos_angle))


def generate_feedback(avg_knee_angle, symmetry, avg_hip_angle):
    feedback = []

    # Depth / Knee check
    if avg_knee_angle > 160:
        feedback.append("Go deeper and bend your knees more.")

    # Symmetry check (feedback only)
    if symmetry > 15:
        feedback.append("Maintain balanced weight on both legs.")

    # Torso / Chest check
    if avg_hip_angle < 140:
        feedback.append("Keep your chest up and maintain a stronger torso position.")

    if not feedback:
        feedback.append("Good squat form ✅")

    return " | ".join(feedback)


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

    # Strong rep filtering
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


def evaluate_single_rep(joints, person_name, rep_number):
    try:
        left_shoulder = joints[5]
        right_shoulder = joints[6]

        left_hip = joints[11]
        right_hip = joints[12]

        left_knee = joints[13]
        right_knee = joints[14]

        left_ankle = joints[15]
        right_ankle = joints[16]

        # Knee Angles
        left_knee_angle = calculate_angle(
            left_hip, left_knee, left_ankle
        )

        right_knee_angle = calculate_angle(
            right_hip, right_knee, right_ankle
        )

        avg_knee_angle = (
            left_knee_angle + right_knee_angle
        ) / 2

        # Hip / Torso Angles
        left_hip_angle = calculate_angle(
            left_shoulder, left_hip, left_knee
        )

        right_hip_angle = calculate_angle(
            right_shoulder, right_hip, right_knee
        )

        avg_hip_angle = (
            left_hip_angle + right_hip_angle
        ) / 2

        # Symmetry
        symmetry = abs(
            left_knee_angle - right_knee_angle
        )

        # =========================
        # FINAL LABEL LOGIC
        # =========================

        # Only hard-fail on real squat issues
        label = 0

        if avg_knee_angle > 160:
            label = 1

        if avg_hip_angle < 140:
            label = 1

        # Symmetry is feedback only, not hard fail

        # =========================
        # QUALITY SCORE
        # =========================

        quality_score = 100

        if avg_knee_angle > 160:
            quality_score -= 40

        if avg_hip_angle < 140:
            quality_score -= 20

        if symmetry > 15:
            quality_score -= 10

        quality_score = max(0, quality_score)

        # Feedback
        feedback = generate_feedback(
            avg_knee_angle,
            symmetry,
            avg_hip_angle
        )

        return {
            "person_file": person_name,
            "rep_number": rep_number,
            "avg_knee_angle": round(avg_knee_angle, 2),
            "avg_hip_angle": round(avg_hip_angle, 2),
            "symmetry": round(symmetry, 2),
            "quality_score": quality_score,
            "label": label,
            "feedback": feedback
        }

    except KeyError:
        return None


def _dir_has_keypoint_json(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    return any(p.suffix.lower() == ".json" for p in folder.iterdir())


def resolve_default_input_dir() -> Path:
    """
    Pick a sensible default keypoints folder for a fresh clone.

    Order:
    1) golden reference outputs (when present)
    2) local extraction output directory (gitignored, but common during dev)
    3) small committed sample fixtures for smoke-testing the feedback script
    """
    candidates = [
        Path("golden_reference/processed_outputs/keypoints"),
        Path("outputs/keypoints"),
        Path("data/sample_keypoints"),
    ]
    for folder in candidates:
        if _dir_has_keypoint_json(folder):
            return folder
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract rep-level squat features from YOLO-style keypoint JSON files "
            "and write a CSV summary."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Folder containing per-video *_keypoints.json files. "
            "If omitted, uses the first available of: "
            "golden_reference/processed_outputs/keypoints, outputs/keypoints, data/sample_keypoints."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="feedback/final_features.csv",
        help="Where to write the extracted rep features CSV.",
    )
    args = parser.parse_args()
    if args.input_dir is None:
        args.input_dir = str(resolve_default_input_dir())
    return args


def main() -> None:
    args = parse_args()

    data_folder = Path(args.input_dir)
    if not data_folder.exists() or not data_folder.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {data_folder}. "
            "Pass --input-dir to a folder containing keypoint JSON files."
        )
    if not _dir_has_keypoint_json(data_folder):
        raise FileNotFoundError(
            f"No .json keypoint files found in: {data_folder}. "
            "Pass --input-dir to a folder containing keypoint JSON files."
        )

    all_results: list[dict] = []

    for file_name in sorted(os.listdir(data_folder)):
        if not file_name.endswith(".json"):
            continue

        file_path = data_folder / file_name

        print(f"\nProcessing person: {file_name}")

        rep_frames = detect_rep_bottom_frames(str(file_path))

        print(f"Detected REAL squat reps: {len(rep_frames)}")

        for rep_idx, rep_frame in enumerate(rep_frames, start=1):
            result = evaluate_single_rep(
                rep_frame["joints"],
                file_name,
                rep_idx,
            )

            if result:
                all_results.append(result)

    df = pd.DataFrame(all_results)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\n====================================")
    print(f"Total real squat reps processed: {len(df)}")
    print("Final Rep-by-Rep Feedback System Ready ✅")
    print(f"Wrote: {output_csv}")
    print("====================================\n")

    if df.empty:
        print("No reps detected. Check your input JSONs and rep detection thresholds.")
        return

    print("\n========== FULL REP RESULTS ==========\n")

    for person in df["person_file"].unique():
        print(f"\nPerson: {person}")

        person_data = df[df["person_file"] == person]

        for _, row in person_data.iterrows():
            status = "Good Squat ✅" if row["label"] == 0 else "Needs Improvement ⚠️"

            print(
                f"Rep {row['rep_number']} | "
                f"Knee Angle: {row['avg_knee_angle']} | "
                f"Hip Angle: {row['avg_hip_angle']} | "
                f"Symmetry: {row['symmetry']} | "
                f"Score: {row['quality_score']} | "
                f"{status}"
            )

            print(f"Feedback: {row['feedback']}")
            print("-" * 60)

    print("\n==========================")
    print("Label Counts")
    print("==========================")
    print(df["label"].value_counts())
    print("==========================\n")


if __name__ == "__main__":
    main()