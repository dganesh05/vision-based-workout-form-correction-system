import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


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


def get_video_metadata(video_path: Path) -> tuple[float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not fps or fps <= 0:
        fps = 30.0
    return fps, width, height


def compute_hard_attention_scores(xyxy: np.ndarray, confs: np.ndarray) -> np.ndarray:
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return areas * confs


def select_primary_index(track_ids, scores, active_track_id, missing_frames, max_missing):
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


def draw_primary_subject(frame, box, keypoints_xy, keypoints_conf, min_kpt_conf):
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


def write_primary_label(label_path: Path, box, keypoints_xy, keypoints_conf, frame_w, frame_h):
    if box is None:
        label_path.write_text("", encoding="utf-8")
        return

    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / frame_w
    cy = ((y1 + y2) / 2.0) / frame_h
    bw = (x2 - x1) / frame_w
    bh = (y2 - y1) / frame_h

    cx = np.clip(cx, 0.0, 1.0)
    cy = np.clip(cy, 0.0, 1.0)
    bw = np.clip(bw, 0.0, 1.0)
    bh = np.clip(bh, 0.0, 1.0)

    values = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for i in range(keypoints_xy.shape[0]):
        x_norm = np.clip(keypoints_xy[i, 0] / frame_w, 0.0, 1.0)
        y_norm = np.clip(keypoints_xy[i, 1] / frame_h, 0.0, 1.0)
        k_conf = float(np.clip(keypoints_conf[i], 0.0, 1.0))
        values.append(f"{x_norm:.6f} {y_norm:.6f} {k_conf:.6f}")

    label_path.write_text(" ".join(values) + "\n", encoding="utf-8")


def run_hard_attention(args):
    source_path = Path(args.source)
    output_video_path = Path(args.output_video)
    labels_dir = Path(args.output_labels)

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    fps, width, height = get_video_metadata(source_path)

    model = YOLO(args.weights)

    writer = None
    active_track_id = None
    missing_frames = 0

    frame_index = 0

    results_stream = model.track(
        source=str(source_path),
        stream=True,
        persist=True,
        conf=args.conf,
        verbose=False,
    )

    for result in results_stream:
        frame_index += 1
        frame = result.orig_img.copy()

        if writer is None:
            h, w = frame.shape[:2]
            if width <= 0 or height <= 0:
                width, height = w, h
            writer = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )

        primary_box = None
        primary_keypoints_xy = None
        primary_keypoints_conf = None

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            track_ids = None
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy()

            scores = compute_hard_attention_scores(xyxy, confs)
            primary_index, active_track_id, missing_frames = select_primary_index(
                track_ids=track_ids,
                scores=scores,
                active_track_id=active_track_id,
                missing_frames=missing_frames,
                max_missing=args.max_missing,
            )

            if primary_index is not None:
                primary_box = xyxy[primary_index]
                if result.keypoints is not None:
                    primary_keypoints_xy = result.keypoints.xy[primary_index].cpu().numpy()
                    if result.keypoints.conf is not None:
                        primary_keypoints_conf = result.keypoints.conf[primary_index].cpu().numpy()
                    else:
                        primary_keypoints_conf = np.ones(primary_keypoints_xy.shape[0], dtype=float)

        else:
            if active_track_id is not None:
                missing_frames += 1
                if missing_frames >= args.max_missing:
                    active_track_id = None

        if primary_box is not None and primary_keypoints_xy is not None and primary_keypoints_conf is not None:
            draw_primary_subject(
                frame=frame,
                box=primary_box,
                keypoints_xy=primary_keypoints_xy,
                keypoints_conf=primary_keypoints_conf,
                min_kpt_conf=args.kpt_conf,
            )

        writer.write(frame)

        label_path = labels_dir / f"{source_path.stem}_{frame_index}.txt"
        write_primary_label(
            label_path=label_path,
            box=primary_box,
            keypoints_xy=primary_keypoints_xy if primary_keypoints_xy is not None else np.zeros((0, 2)),
            keypoints_conf=primary_keypoints_conf if primary_keypoints_conf is not None else np.zeros((0,)),
            frame_w=frame.shape[1],
            frame_h=frame.shape[0],
        )

        if args.show:
            cv2.imshow("Primary Subject (Hard Attention)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"Saved primary-only video to: {output_video_path}")
    print(f"Saved primary-only labels to: {labels_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 pose with hard attention to keep only the primary subject."
    )
    parser.add_argument("--weights", default="yolov8n-pose.pt", help="YOLOv8 pose weights path")
    parser.add_argument("--source", default="sample_video.MP4", help="Input video path")
    parser.add_argument(
        "--output-video",
        default="runs/pose/primary/primary_subject.mp4",
        help="Output video path for primary-subject rendering",
    )
    parser.add_argument(
        "--output-labels",
        default="runs/pose/primary/labels",
        help="Output directory for per-frame primary-subject labels",
    )
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument(
        "--kpt-conf",
        type=float,
        default=0.5,
        help="Minimum keypoint confidence for drawing",
    )
    parser.add_argument(
        "--max-missing",
        type=int,
        default=10,
        help="Frames to wait before primary subject reacquisition",
    )
    parser.add_argument("--show", action="store_true", help="Show live output window")
    return parser.parse_args()


if __name__ == "__main__":
    run_hard_attention(parse_args())
