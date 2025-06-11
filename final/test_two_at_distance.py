# 65 cm
import numpy as np
import time
from djitellopy import Tello
from pupil_apriltags import Detector
import cv2
import matplotlib.pyplot as plt
import threading
import final_setting  # Import the settings from final_setting.py
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
        cv2.imwrite(f"./images/tello_apriltag_{time.time()}.jpg", frame)
        
    except Exception as e:
        print(f"Failed to save image: {e}")
        pass
    cv2.waitKey(1)

    return tags_info

# ...existing code...

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

def main():
    # camera_params = [1.84825711e+03, 1.85066209e+03, 1.29929007e+03, 9.04726581e+02]  # fx, fy, cx, cy
    # 917.0, 917.0, 480.0, 360.0
    camera_params = [ 917.0, 917.0, 480.0, 360.0]  # fx, fy, cx, cy
    tag_size = 0.166  # Tag size in meters
    # AprilTag detector
    at_detector = Detector(families='tag36h11')
     # Kalman Filter
    
    # Initialize Tello
    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
     # Start video stream
    tello.streamon()
    frame_read = tello.get_frame_read()
    # Wait for video stream to start
    frame = None
    retry_count = 0
    know_tag = None
    unknown_tag = None
    while frame is None and retry_count < 10:
        frame = frame_read.frame
        time.sleep(0.5)
        retry_count += 1
    
    if frame is None:
        print("Failed to get video stream")
        return
    
    while True:
        # Detect AprilTags
        tags_info = average_apriltag_detection(frame_read, at_detector, camera_params, tag_size, num=10)
        
        if not tags_info:
            print("No AprilTags detected")
            continue
        print(f"tags_info: {tags_info}")
        # if not tags_info:
        #     print("No AprilTags detected")
        #     continue

        # for tag in tags_info:
        #     if tag['id'] == 0:
        #         know_tag = tag
        #     elif tag['id']  == 108:
        #         unknown_tag = tag
        # if know_tag and unknown_tag:
        # # known_tag (0, 0)
        #     print (f"unknown_tag pose: {unknown_tag['pose']}")
        #     print(f"known_tag pose: {know_tag['pose']}")
        #     # Calculate distance
        #     print(f"unknown_tag location: {-1 * (unknown_tag['pose'][0] - know_tag['pose'][0])}")
                
        time.sleep(1)


if __name__ == '__main__':
    main()