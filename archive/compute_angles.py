import argparse
import json
import math
from pathlib import Path


def calculate_angle(a, b, c):
    ab = [a[0] - b[0], a[1] - b[1]]
    cb = [c[0] - b[0], c[1] - b[1]]

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)

    if mag_ab == 0 or mag_cb == 0:
        return None

    cos_angle = dot / (mag_ab * mag_cb)
    cos_angle = max(min(cos_angle, 1), -1)
    return math.degrees(math.acos(cos_angle))


def summary_stats(values):
    if not values:
        return None, None, None
    return sum(values) / len(values), min(values), max(values)


def process_file(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    total_frames = len(data)

    knee_angles = []
    hip_angles = []
    torso_angles = []
    per_frame_angles = []

    for frame in data:
        people = frame.get("people", [])
        if not people:
            continue

        keypoints = people[0].get("keypoints", [])
        if len(keypoints) < 17:
            continue

        # Consistent side choice: left-side joints
        shoulder = (keypoints[5]["x"], keypoints[5]["y"])
        hip = (keypoints[11]["x"], keypoints[11]["y"])
        knee = (keypoints[13]["x"], keypoints[13]["y"])
        ankle = (keypoints[15]["x"], keypoints[15]["y"])

        # Knee: hip -> knee -> ankle
        knee_angle = calculate_angle(hip, knee, ankle)
        # Hip: shoulder -> hip -> knee
        hip_angle = calculate_angle(shoulder, hip, knee)
        # Torso: angle between vertical axis and shoulder -> hip line
        vertical = (hip[0], hip[1] - 100)
        torso_angle = calculate_angle(shoulder, hip, vertical)

        if knee_angle is not None:
            knee_angles.append(knee_angle)
        if hip_angle is not None:
            hip_angles.append(hip_angle)
        if torso_angle is not None:
            torso_angles.append(torso_angle)

        per_frame_angles.append(
            {
                "frame": frame.get("frame"),
                "knee_angle": knee_angle,
                "hip_angle": hip_angle,
                "torso_angle": torso_angle,
            }
        )

    avg_knee, min_knee, max_knee = summary_stats(knee_angles)
    avg_hip, min_hip, max_hip = summary_stats(hip_angles)
    avg_torso, min_torso, max_torso = summary_stats(torso_angles)

    return {
        "video_id": json_path.stem.replace("_keypoints", ""),
        "total_frames": total_frames,
        "frames_with_person": len(per_frame_angles),
        "avg_knee_angle": avg_knee,
        "min_knee_angle": min_knee,
        "max_knee_angle": max_knee,
        "avg_hip_angle": avg_hip,
        "min_hip_angle": min_hip,
        "max_hip_angle": max_hip,
        "avg_torso_angle": avg_torso,
        "min_torso_angle": min_torso,
        "max_torso_angle": max_torso,
        "per_frame_angles": per_frame_angles,
    }


def process_all_files(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    keypoint_files = sorted(input_dir.glob("*_keypoints.json"))

    for keypoint_file in keypoint_files:
        result = process_file(keypoint_file)
        output_path = output_dir / f"{result['video_id']}_angles.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved angles: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute knee/hip/torso angles from keypoint JSON files.")
    parser.add_argument("--input-dir", default="outputs/keypoints", help="Input keypoint JSON directory.")
    parser.add_argument("--output-dir", default="outputs/angles", help="Output angle JSON directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all_files(Path(args.input_dir), Path(args.output_dir))