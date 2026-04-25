import os
import json
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

input_folder = "videos"
output_video_folder = "outputs/annotated_videos"
output_keypoint_folder = "outputs/keypoints"

os.makedirs(output_video_folder, exist_ok=True)
os.makedirs(output_keypoint_folder, exist_ok=True)

video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".MOV")

for filename in os.listdir(input_folder):
    if not filename.endswith(video_extensions):
        continue

    video_path = os.path.join(input_folder, filename)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open {video_path}")
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    base_name = os.path.splitext(filename)[0]
    output_video_path = os.path.join(output_video_folder, f"{base_name}_annotated.mp4")
    output_keypoint_path = os.path.join(output_keypoint_folder, f"{base_name}_keypoints.json")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    all_frames = []
    frame_idx = 0

    print()
    print("===================================")
    print(f"Processing: {filename}")

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=0.3, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        frame_data = {
            "frame": frame_idx,
            "people": []
        }

        if results[0].keypoints is not None:
            keypoints_xy = results[0].keypoints.xy.cpu().numpy()
            keypoints_conf = results[0].keypoints.conf.cpu().numpy()

            for person_idx in range(len(keypoints_xy)):
                person_data = {
                    "person_id": person_idx,
                    "keypoints": []
                }

                for joint_idx in range(len(keypoints_xy[person_idx])):
                    x = float(keypoints_xy[person_idx][joint_idx][0])
                    y = float(keypoints_xy[person_idx][joint_idx][1])
                    conf = float(keypoints_conf[person_idx][joint_idx])

                    person_data["keypoints"].append({
                        "joint_id": joint_idx,
                        "x": x,
                        "y": y,
                        "confidence": conf
                    })

                frame_data["people"].append(person_data)

        all_frames.append(frame_data)
        frame_idx += 1

    cap.release()
    out.release()

    with open(output_keypoint_path, "w") as f:
        json.dump(all_frames, f, indent=2)

    print(f"Saved video to: {output_video_path}")
    print(f"Saved keypoints to: {output_keypoint_path}")
    print("Finished processing")
    print("===================================")