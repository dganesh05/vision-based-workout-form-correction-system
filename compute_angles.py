import json
import math
import os

def calculate_angle(a, b, c):
    ab = [a[0] - b[0], a[1] - b[1]]
    cb = [c[0] - b[0], c[1] - b[1]]

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)

    if mag_ab == 0 or mag_cb == 0:
        return 0

    cos_angle = dot / (mag_ab * mag_cb)
    cos_angle = max(min(cos_angle, 1), -1)

    return math.degrees(math.acos(cos_angle))

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def process_file(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    min_knee = 180
    max_torso = 0

    bad_torso = 0
    valgus = 0
    total = 0

    for frame in data:
        if len(frame["people"]) == 0:
            continue

        total += 1

        kp = frame["people"][0]["keypoints"]

        shoulder = (kp[5]["x"], kp[5]["y"])
        hip = (kp[11]["x"], kp[11]["y"])
        knee = (kp[13]["x"], kp[13]["y"])
        ankle = (kp[15]["x"], kp[15]["y"])

        hip_r = (kp[12]["x"], kp[12]["y"])
        knee_r = (kp[14]["x"], kp[14]["y"])
        ankle_r = (kp[16]["x"], kp[16]["y"])

        vertical = (hip[0], hip[1] - 100)

        knee_angle = calculate_angle(hip, knee, ankle)
        torso_angle = calculate_angle(shoulder, hip, vertical)

        if knee_angle < min_knee:
            min_knee = knee_angle

        if torso_angle > max_torso:
            max_torso = torso_angle

        if torso_angle > 40:
            bad_torso += 1

        left_diff = abs(knee[0] - ankle[0]) / (distance(hip, ankle) + 1e-6)
        right_diff = abs(knee_r[0] - ankle_r[0]) / (distance(hip_r, ankle_r) + 1e-6)

        if left_diff > 0.15 or right_diff > 0.15:
            valgus += 1

    depth_score = 100 if min_knee < 80 else 70 if min_knee < 100 else 40
    torso_score = 100 - (bad_torso / total) * 100
    knee_score = 100 - (valgus / total) * 100

    final_score = (0.4 * depth_score) + (0.3 * torso_score) + (0.3 * knee_score)

    print()
    print("===================================")
    print(f"File: {json_path}")
    print(f"Final Form Score: {final_score:.2f}/100")

    if final_score > 85:
        print("Overall: Excellent form")
    elif final_score > 70:
        print("Overall: Good form")
    else:
        print("Overall: Needs improvement")

    print("===================================")

def process_all_files(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            full_path = os.path.join(folder_path, file)
            process_file(full_path)

process_all_files("outputs/keypoints")