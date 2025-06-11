import math

import numpy as np
import time
from djitellopy import Tello
from pupil_apriltags import Detector
import cv2
import matplotlib.pyplot as plt
import final_rewrite_setting 

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
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"./images/tello_apriltag_{timestamp}.jpg", frame)

        
    except Exception as e:
        print(f"Failed to save image: {e}")
        pass
    cv2.waitKey(1)

    return tags_info

def tello_command(tello, movement_request):
    """Execute movement command and return displacement"""
    dp = np.array([0.0, 0.0, 0.0])
    cmd, value = movement_request
    
    try:
        if cmd == "forward":
            tello.move_forward(value)
            dp = np.array([0.0, value * 0.01, 0.0])  # Convert cm to m
        elif cmd == "back":
            tello.move_back(value)
            dp = np.array([0.0, -value * 0.01, 0.0])
        elif cmd == "right":
            tello.move_right(value)
            dp = np.array([value * 0.01, 0.0, 0.0])
        elif cmd == "left":
            tello.move_left(value)
            dp = np.array([-value * 0.01, 0.0, 0.0])
        elif cmd == "up":
            tello.move_up(value)
            dp = np.array([0.0, 0.0, value * 0.01])
        elif cmd == "down":
            tello.move_down(value)
            dp = np.array([0.0, 0.0, -value * 0.01])
        elif cmd == "cw":
            tello.rotate_clockwise(value)
            dp = np.array([0.0, 0.0, 0.0])
        elif cmd == "ccw":
            tello.rotate_counter_clockwise(value)
            dp = np.array([0.0, 0.0, 0.0])
        
        time.sleep(1.5)  # Wait for movement to complete
    except Exception as e:
        print(f"Movement command failed: {e}")
        
    return dp


def average_apriltag_detection(frame_read, at_detector, camera_params, tag_size, num=10):
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



# filepath: c:\Users\jianh\OneDrive\Documents\大學\sophomore_spring\人本\HCClab\final\final_rewrite.py
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
        
    
        
        

def main():
    start_time = time.time()
    camera_params = [ 917.0, 917.0, 480.0, 360.0]  # fx, fy, cx, cy
    tag_size = 0.166  # Tag size in meters
    unknown_tags_location = []


    # AprilTag detector
    at_detector = Detector(families='tag36h11')
    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
    if tello.get_battery() < 20:
        print("Battery too low! Please charge the drone.")
        
    tello.streamon()
    frame_read = tello.get_frame_read()
    # Wait for video stream to start
    frame = None
    retry_count = 0
    while frame is None and retry_count < 10:
        frame = frame_read.frame
        time.sleep(0.5)
        retry_count += 1
    
    if frame is None:
        print("Failed to get video stream")
        return
    # Take off
    print("Taking off...")
    tello.takeoff()
    time.sleep(3)
    align_count = 0
    #------------end of take off----------------


    # test align_tag
    while True and align_count <= 5:
        # Detect AprilTags
        tags_info = detect_apriltag(frame_read, at_detector, camera_params, tag_size)
        if not tags_info:
            print("No AprilTags detected.")
            time.sleep(1)
            continue

        # Print detected tags
        for tag in tags_info:
            print(f"Detected tag ID: {tag['id']}, Pose: {tag['pose']}")

        # Align to a specific tag (e.g., tag_id 100)
        target_tag_id = 108
        align_tag(tello, frame_read, at_detector, camera_params, tag_size, target_tag_id, front_distance=1)
        align_count+=1
if __name__ == "__main__":
    main()