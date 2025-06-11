import cv2
import numpy as np
import time
from djitellopy import Tello
from pupil_apriltags import Detector
import os
import final_rewrite_setting as frs  # exposes ar_word, professor_landing_spots, unkonwn_tags, real_unknown_tags
from ultralytics import YOLO

WALL_1_BACK = 150
WALL_2_BACK = 130
WALL_3_BACK = 80 #70 - 100
WALL_4_BACK = 130



###########################
# === Global Parameters ===
###########################

# Use settings from final_rewrite_setting -----------------------------------
APRILTAG_WORLD_COORDS = {tid: np.array([xy[0], xy[1], 0.0]) for tid, xy in frs.ar_word.items()}
PROFESSOR_LANDING_SPOTS = frs.professor_landing_spots  # dict[str, [x, y]]

# ––––– Intrinsics (fx, fy, cx, cy) –––––  (update to your calibration)
CAMERA_PARAMS = [911.008, 909.733, 493.744, 360.485]
CAMERA_MATRIX = np.array([[CAMERA_PARAMS[0], 0, CAMERA_PARAMS[2]],
                          [0, CAMERA_PARAMS[1], CAMERA_PARAMS[3]],
                          [0, 0, 1]], dtype=np.float32)
TAG_SIZE = 0.166  # metres



def tello_command(tello: Tello, movement: tuple[str, int]) -> np.ndarray:
    """Execute a basic Tello SDK movement and return a 3‑vector Δp in metres (body frame)."""
    cmd, value = movement
    dp = np.zeros(3)
    try:
        match cmd:
            case "forward": tello.move_forward(value); dp = np.array([0,  value, 0])
            case "back":    tello.move_back(value);    dp = np.array([0, -value, 0])
            case "right":   tello.move_right(value);   dp = np.array([ value, 0, 0])
            case "left":    tello.move_left(value);    dp = np.array([-value, 0, 0])
            case "up":      tello.move_up(value);      dp = np.array([0, 0,  value])
            case "down":    tello.move_down(value);    dp = np.array([0, 0, -value])
            case "cw":      tello.rotate_clockwise(value)
            case "ccw":     tello.rotate_counter_clockwise(value)
            case _:
                print(f"[WARN] Unknown movement {cmd}")
        time.sleep(1)  # allow motion to complete
    except Exception as e:
        print(f"[ERR] movement {movement} failed: {e}")
    return dp * 0.01  # convert cm -> m


def detect_apriltag(frame_read, at_detector, camera_params, tag_size):
    """Detect AprilTags in the current frame,
       回傳: tags_info (list of dicts with tag_id and pose),
       Each dict contains:
        - 'id': AprilTag ID
        - 'pose': (x, y, z) position relative to camera in meters"""

    print("Detecting AprilTag...")
    tags_info = []
    frame = frame_read.frame
    if frame is None:
        return tags_info
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect tags
    tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

    for tag in tags:
        if tag.pose_t is not None:
            t = tag.pose_t.flatten()  # (x, y, z) relative to camera
            tags_info.append({'id': tag.tag_id, 'pose': t})

            # Draw tag on image
            corners = np.int32(tag.corners)
            center = tuple(np.int32(tag.center))
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {tag.tag_id}", (center[0]+10, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    # Show live feed
    cv2.imshow("Tello Stream with AprilTag", frame)
    #save image
    try:
        filename = f"./images/tello_apriltag_{time.time()}.jpg"
        cv2.imwrite(filename, frame)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
    except Exception as e:
        print(f"Failed to save image: {e}")
        pass
    cv2.waitKey(1)

    return tags_info


def average_apriltag_detection(frame_read, at_detector, camera_params, tag_size = TAG_SIZE, num=10):
    """Detect AprilTags multiple times and return average pose for each tag id"""
    tag_accumulator = {}
    tag_counts = {}

    for _ in range(num):
        tags_info = detect_apriltag(frame_read, at_detector, camera_params, tag_size)
        for tag in tags_info:
            tag_id = tag['id']
            pose = np.array(tag['pose'])
            if tag_id not in tag_accumulator:
                tag_accumulator[tag_id] = pose
                tag_counts[tag_id] = 1
            else:
                tag_accumulator[tag_id] += pose
                tag_counts[tag_id] += 1
        time.sleep(0.1)  # 避免太快

    avg_tags = []
    for tag_id in tag_accumulator:
        avg_pose = tag_accumulator[tag_id] / tag_counts[tag_id]
        avg_tags.append({'id': tag_id, 'pose': avg_pose})
    return avg_tags

# 使用方式（在 main 或 for 迴圈裡）：
# tags_info_avg = average_apriltag_detection(frame_read, at_detector, camera_params, tag_size, num=10)




def align_tag(tello, frame_read, at_detector, camera_params, tag_size, target_tag_id, front_distance):
    """
    讓 Tello 對齊指定 tag，並停在 tag 正前方 front_distance (單位: 公尺)
    """
    max_attempts = 10
    retry = False
    for attempt in range(max_attempts):
        tags_info = detect_apriltag(frame_read, at_detector, camera_params, tag_size)
        tag = next((t for t in tags_info if t['id'] == target_tag_id), None)
        if tag is None:
            print(f"Tag {target_tag_id} not found, retrying...")
            time.sleep(0.5)
            continue

        # tag['pose']: [x, y, z] (單位: 公尺)，x: 左右, y: 上下, z: 前後
        x, y, z = tag['pose']
        print(f"Aligning to tag {target_tag_id}: x={x:.2f}m, y={y:.2f}m, z={z:.2f}m")

        # 計算需要移動的距離
        move_left_right = int(-x * 100)   # x>0: tag在右邊，要往左移
        move_up_down   = int(-y * 100)    # y>0: tag在下方，要往上移
        move_forward   = int((z - front_distance) * 100)  # z>front_distance: 要往前靠近

                # 左右對齊
        if abs(move_left_right) > 10:
            move_cm = min(max(abs(move_left_right), 20), 500)
            if move_left_right > 0:
                tello.move_left(move_cm)
            else:
                tello.move_right(move_cm)
            time.sleep(1)

        # 上下對齊
        if abs(move_up_down) > 10:
            move_cm = min(max(abs(move_up_down), 20), 500)
            if move_up_down > 0:
                tello.move_up(move_cm)
            else:
                tello.move_down(move_cm)
            time.sleep(1)

        # 前後對齊
        if abs(move_forward) > 10:
            move_cm = min(max(abs(move_forward), 20), 500)
            if move_forward > 0:
                tello.move_forward(move_cm)
            else:
                tello.move_back(move_cm)
            time.sleep(1)

        # 如果都在容許範圍內就結束
        if abs(move_left_right) <= 10 and abs(move_up_down) <= 10 and abs(move_forward) <= 10:
            print("Alignment complete.")
            return True
            break
        if retry != True and attempt >= max_attempts - 1:
            tello.move_back(50)
            print("Failed to align, try again.")
            max_attempts = 0
            retry = True
    else:
        print(f"Failed to align to tag {target_tag_id}")
        return False


def estimate_pose(tag_corners, tag_size=TAG_SIZE):
    object_points = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [ tag_size/2, -tag_size/2, 0],
        [ tag_size/2,  tag_size/2, 0],
        [-tag_size/2,  tag_size/2, 0]
    ], dtype=np.float32)

    image_points = np.array(tag_corners, dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, CAMERA_MATRIX, distCoeffs=None)
    if not success:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec