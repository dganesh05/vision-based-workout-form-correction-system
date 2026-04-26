import argparse
import json
from pathlib import Path
import os

import cv2
import numpy as np
from ultralytics import YOLO

os.makedirs("outputs/annotated_videos", exist_ok=True)
os.makedirs("outputs/keypoints", exist_ok=True)


COCO_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        return 30.0
    return float(fps)


def compute_hard_attention_scores(xyxy: np.ndarray, confs: np.ndarray) -> np.ndarray:
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return areas * confs


def select_primary_index(
    track_ids: np.ndarray | None,
    scores: np.ndarray,
    active_track_id: int | None,
    missing_frames: int,
    max_missing: int,
) -> tuple[int | None, int | None, int]:
    if scores.size == 0:
        if active_track_id is not None:
            missing_frames += 1
        return None, active_track_id, missing_frames

    if active_track_id is not None and track_ids is not None:
        matches = np.where(track_ids == active_track_id)[0]
        if matches.size > 0:
            return int(matches[0]), active_track_id, 0

    if active_track_id is not None:
        missing_frames += 1
        if missing_frames < max_missing:
            return None, active_track_id, missing_frames

    primary_index = int(np.argmax(scores))
    new_track_id = None
    if track_ids is not None:
        value = track_ids[primary_index]
        if not np.isnan(value):
            new_track_id = int(value)

    return primary_index, new_track_id, 0


def draw_primary_subject(
    frame: np.ndarray,
    box: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    min_kpt_conf: float,
) -> None:
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 180, 255), 2)
    cv2.putText(
        frame,
        "Primary Subject",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (60, 180, 255),
        2,
        cv2.LINE_AA,
    )

    for i, (x, y) in enumerate(keypoints_xy):
        if keypoints_conf[i] < min_kpt_conf:
            continue
        cv2.circle(frame, (int(x), int(y)), 4, (0, 220, 255), -1)

    for a, b in COCO_SKELETON:
        if keypoints_conf[a] < min_kpt_conf or keypoints_conf[b] < min_kpt_conf:
            continue
        x1_line, y1_line = keypoints_xy[a]
        x2_line, y2_line = keypoints_xy[b]
        cv2.line(
            frame,
            (int(x1_line), int(y1_line)),
            (int(x2_line), int(y2_line)),
            (0, 255, 80),
            2,
        )


def list_videos(input_root: Path) -> list[Path]:
    files = [
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(files)


def output_stem(input_root: Path, video_path: Path) -> str:
    relative_no_suffix = video_path.relative_to(input_root).with_suffix("")
    return "__".join(relative_no_suffix.parts)


def process_video(
    model: YOLO,
    video_path: Path,
    input_root: Path,
    output_video_dir: Path,
    output_keypoint_dir: Path,
    conf: float,
    kpt_conf: float,
    max_missing: int,
) -> None:
    stem = output_stem(input_root, video_path)
    output_video_path = output_video_dir / f"{stem}_annotated.mp4"
    output_keypoint_path = output_keypoint_dir / f"{stem}_keypoints.json"

    all_frames: list[dict] = []
    frame_idx = 0
    fps = get_video_fps(video_path)

    writer = None
    active_track_id = None
    missing_frames = 0

    results_stream = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        conf=conf,
        verbose=False,
    )

    for result in results_stream:
        frame = result.orig_img.copy()

        if writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        frame_data = {"frame": frame_idx, "people": []}

        primary_box = None
        primary_keypoints_xy = None
        primary_keypoints_conf = None

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None

            scores = compute_hard_attention_scores(xyxy, confs)
            primary_index, active_track_id, missing_frames = select_primary_index(
                track_ids=track_ids,
                scores=scores,
                active_track_id=active_track_id,
                missing_frames=missing_frames,
                max_missing=max_missing,
            )

            if primary_index is not None and result.keypoints is not None:
                primary_box = xyxy[primary_index]
                primary_keypoints_xy = result.keypoints.xy[primary_index].cpu().numpy()

                if result.keypoints.conf is not None:
                    primary_keypoints_conf = result.keypoints.conf[primary_index].cpu().numpy()
                else:
                    primary_keypoints_conf = np.ones(primary_keypoints_xy.shape[0], dtype=float)

        else:
            if active_track_id is not None:
                missing_frames += 1
                if missing_frames >= max_missing:
                    active_track_id = None

        if (
            primary_box is not None
            and primary_keypoints_xy is not None
            and primary_keypoints_conf is not None
        ):
            draw_primary_subject(
                frame=frame,
                box=primary_box,
                keypoints_xy=primary_keypoints_xy,
                keypoints_conf=primary_keypoints_conf,
                min_kpt_conf=kpt_conf,
            )

            person_data = {"person_id": 0, "keypoints": []}
            for joint_idx in range(len(primary_keypoints_xy)):
                x = float(primary_keypoints_xy[joint_idx][0])
                y = float(primary_keypoints_xy[joint_idx][1])
                conf_value = float(primary_keypoints_conf[joint_idx])
                person_data["keypoints"].append(
                    {
                        "joint_id": joint_idx,
                        "x": x,
                        "y": y,
                        "confidence": conf_value,
                    }
                )
            frame_data["people"].append(person_data)

        if writer is not None:
            writer.write(frame)

        all_frames.append(frame_data)
        frame_idx += 1

    if writer is not None:
        writer.release()

    with output_keypoint_path.open("w", encoding="utf-8") as f:
        json.dump(all_frames, f, indent=2)

    print(f"Finished {video_path}")
    print(f"  Video: {output_video_path}")
    print(f"  Keypoints: {output_keypoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch process videos and keep only the primary subject with hard attention."
    )
    parser.add_argument("--weights", default="yolov8x-pose-p6.pt", help="YOLOv8 pose weights path")
    parser.add_argument("--input-folder", default="videos", help="Root folder containing videos")
    parser.add_argument(
        "--output-video-folder",
        default="outputs/annotated_videos",
        help="Output folder for annotated videos",
    )
    parser.add_argument(
        "--output-keypoint-folder",
        default="outputs/keypoints",
        help="Output folder for frame-by-frame keypoint JSON",
    )
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--kpt-conf", type=float, default=0.5, help="Minimum keypoint confidence")
    parser.add_argument(
        "--max-missing",
        type=int,
        default=10,
        help="Frames to wait before reacquiring a new primary subject",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_folder)
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    output_video_dir = Path(args.output_video_folder)
    output_keypoint_dir = Path(args.output_keypoint_folder)
    output_video_dir.mkdir(parents=True, exist_ok=True)
    output_keypoint_dir.mkdir(parents=True, exist_ok=True)

    video_files = list_videos(input_root)
    if not video_files:
        print(f"No video files found in {input_root}")
        return

    print(f"Found {len(video_files)} videos")

    model = YOLO(args.weights)
    for video_path in video_files:
        process_video(
            model=model,
            video_path=video_path,
            input_root=input_root,
            output_video_dir=output_video_dir,
            output_keypoint_dir=output_keypoint_dir,
            conf=args.conf,
            kpt_conf=args.kpt_conf,
            max_missing=args.max_missing,
        )


if __name__ == "__main__":
    main()
