import numpy as np
import time
from djitellopy import Tello
from pupil_apriltags import Detector
import cv2
import matplotlib.pyplot as plt
import threading
import final_setting  # Import the settings from final_setting.py

# You'll need to create this file with your professor images and model
# import professor_recognition  # Your image recognition module

class KalmanFilter:
    def __init__(self):
        """
        Improved Kalman Filter for better position tracking
        """
        # State dimension: [x, y, z] = 3
        n = 3  # state dimension
        m = 3  # control input dimension
        k = 3  # observation dimension
        self.A = np.eye(n)
        self.B = np.eye(n)
        self.C = np.eye(k)
        # Reduced process noise for better stability
        self.R = np.diag([0.005, 0.005, 0.005])  # Very small process noise
        # Adjusted measurement noise based on AprilTag accuracy
        self.Q = np.diag([0.02, 0.02, 0.05])  # Lower noise for x,y, higher for z
        self.mu = np.zeros((n, 1))
        self.Sigma = np.eye(n) * 0.5  # Smaller initial uncertainty

    def predict(self, u):
        """
        Prediction step of Kalman Filter
        Args:
            u: Control input (3x1 numpy array) - displacement [dx, dy, dz]
        """
        # Predict state: mu_pred = A * mu + B * u
        self.mu = self.A @ self.mu + self.B @ u
        
        # Predict covariance: Sigma_pred = A * Sigma * A^T + R
        self.Sigma = self.A @ self.Sigma @ self.A.T + self.R

    def update(self, z):
        """
        Update step of Kalman Filter
        Args:
            z: Observation (3x1 numpy array) - measured position [x, y, z]
        Returns:
            Updated state estimate
        """
        # Innovation (measurement residual): y = z - C * mu
        y = z - self.C @ self.mu
        
        # Innovation covariance: S = C * Sigma * C^T + Q
        S = self.C @ self.Sigma @ self.C.T + self.Q
        
        # Kalman gain: K = Sigma * C^T * S^(-1)
        K = self.Sigma @ self.C.T @ np.linalg.inv(S)
        
        # Update state estimate: mu = mu + K * y
        self.mu = self.mu + K @ y
        
        # Update covariance: Sigma = (I - K * C) * Sigma
        I = np.eye(self.mu.shape[0])
        self.Sigma = (I - K @ self.C) @ self.Sigma

        return self.mu

    def get_state(self):
        return self.mu, self.Sigma

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
        cv2.imwrite(f"./tello_apriltag_{time.time}.jpg", frame)
        
    except Exception as e:
        print(f"Failed to save image: {e}")
        pass
    cv2.waitKey(1)

    return tags_info

def recognize_professor(frame): # TODO 
    pass 
    # """
    # Recognize professor from the current frame
    # Returns: professor_name (string) or None if not recognized
    # """
    # # This function should be implemented based on your trained model
    # # For now, returning a placeholder
    # try:
    #     # Call your professor recognition function here
    #     result = professor_recognition.predict(frame)
    #     return result
    # except:
    #     print("Professor recognition not implemented yet")
    #     return None

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

def calculate_drone_position(known_tag_id, tag_pose, ar_word, drone_yaw=0):
    """Calculate drone position based on known AprilTag"""
    if known_tag_id in ar_word:
        # Tag position in camera frame (x: right, y: down, z: forward)
        # Convert to world frame considering drone's orientation
        cos_yaw = np.cos(np.radians(drone_yaw))
        sin_yaw = np.sin(np.radians(drone_yaw))
        
        # Camera to world transformation
        # Camera: x=right, z=forward -> World: x=right, y=forward
        camera_x = tag_pose[0]  # right in camera frame
        camera_z = tag_pose[2]  # forward in camera frame
        
        # Transform to world coordinates considering drone rotation
        world_offset_x = cos_yaw * camera_x + sin_yaw * camera_z
        world_offset_y = -sin_yaw * camera_x + cos_yaw * camera_z
        
        # Drone position = Tag world position - offset
        drone_x = ar_word[known_tag_id][0] - world_offset_x
        drone_y = ar_word[known_tag_id][1] - world_offset_y
        
        return np.array([drone_x, drone_y])
    return None

def move_to_target(tello, current_pos, target_pos, kf, tolerance=0.1):
    """Move drone from current position to target position"""
    print(f"Moving from {current_pos} to {target_pos}")
    
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    movements = []
    
    # Move in X direction
    if abs(dx) > tolerance:
        if dx > 0:
            move_cm = int(abs(dx) * 100)  # Convert to cm
            movements.append(("right", min(move_cm, 500)))  # Limit max movement
        else:
            move_cm = int(abs(dx) * 100)
            movements.append(("left", min(move_cm, 500)))
    
    # Move in Y direction
    if abs(dy) > tolerance:
        if dy > 0:
            move_cm = int(abs(dy) * 100)
            movements.append(("forward", min(move_cm, 500)))
        else:
            move_cm = int(abs(dy) * 100)
            movements.append(("back", min(move_cm, 500)))
    
    # Execute movements
    current_estimated_pos = current_pos.copy()
    for movement in movements:
        dp = tello_command(tello, movement)
        current_estimated_pos += dp[:2]  # Update estimated position
        
        # Update Kalman filter
        kf.predict(np.expand_dims(dp, axis=1))
    
    return current_estimated_pos

def plot_trajectory(control_poses, tag_pose, kalmanfilter_pose):
    """Plot trajectory comparison"""
    if not control_poses or not tag_pose or not kalmanfilter_pose:
        print("No trajectory data to plot")
        return
        
    poses_ct = np.array(control_poses)
    poses_at = np.array(tag_pose)
    poses_kf = np.array(kalmanfilter_pose)
    
    plt.figure(figsize=(10, 8))
    plt.plot(poses_ct[:, 0], poses_ct[:, 1], 'ko--', label='Motion Model', linewidth=2)
    plt.plot(poses_at[:, 0], poses_at[:, 1], 'rx-', label='AprilTag', linewidth=2)
    plt.plot(poses_kf[:, 0], poses_kf[:, 1], 'b^-', label='Kalman Filter', linewidth=2)
    plt.title("2D Trajectory Tracking")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def calculate_unknown_tag_position(drone_pos, tag_pose, drone_yaw=0):
    """Calculate unknown tag's world position from drone position and relative pose"""
    cos_yaw = np.cos(np.radians(drone_yaw))
    sin_yaw = np.sin(np.radians(drone_yaw))
    
    # Tag position in camera frame
    camera_x = tag_pose[0]  # right
    camera_z = tag_pose[2]  # forward
    
    # Transform to world coordinates
    world_offset_x = cos_yaw * camera_x + sin_yaw * camera_z
    world_offset_y = -sin_yaw * camera_x + cos_yaw * camera_z
    
    # Unknown tag world position
    tag_world_x = drone_pos[0] + world_offset_x
    tag_world_y = drone_pos[1] + world_offset_y
    
    return np.array([tag_world_x, tag_world_y])
    


def main():
    # Competition start time
    start_time = time.time()
    
    # Camera calibration parameters
    camera_params = [917.0, 917.0, 480.0, 360.0]  # fx, fy, cx, cy
    tag_size = 0.166  # Tag size in meters
    
    # AprilTag detector
    at_detector = Detector(families='tag36h11')
    
    # Kalman Filter
    KF = KalmanFilter()
    
    # Initialize Tello
    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
    
    if tello.get_battery() < 20:
        print("Battery too low! Please charge the drone.")
        return
    
    # Start video stream
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
    tello.move_up(50)
    
    # Competition variables
    drone_wpose_ct = np.array([0.0, 0.0, 0.0])  # Control model position
    d_wposes_ct = []  # Control model trajectory
    d_wposes_at = []  # AprilTag measurement trajectory
    d_wposes_kf = []  # Kalman filter trajectory
    detected_tag_id = None
    unknown_tags = {}  # Store unknown tag positions
    professor_detected = None
    landing_spot = None
    drone_yaw = 0.0  # Track drone's orientation
    
    try:
        # Phase 1: Object Detection (Professor Recognition)


        #DEBUG
        professor_detected = "hh_shuai"
        print("Phase 1: Professor Recognition")
        recognition_attempts = 0
        max_recognition_attempts = 20
        
        while professor_detected is None and recognition_attempts < max_recognition_attempts:
            frame = frame_read.frame
            if frame is not None:
                professor_detected = recognize_professor(frame)
                if professor_detected:
                    print(f"Professor detected: {professor_detected}")
                    # Get corresponding landing spot
                    if hasattr(final_setting, 'professor_landing_spots'):
                        landing_spot = final_setting.professor_landing_spots.get(professor_detected)
                    break
            
            # Move around to get better view
            if recognition_attempts % 5 == 0:
                dp, dyaw = tello_command(tello, ("cw", 30))  # Rotate to get different angle
                drone_wpose_ct += dp
                drone_yaw += dyaw
            
            recognition_attempts += 1
            time.sleep(0.5)
        
        if professor_detected is None:
            print("Failed to detect professor, using default landing spot")
            landing_spot = [0, 0]  # Default position
        
        # Phase 2: AprilTag Detection and Position Estimation
        print("Phase 2: AprilTag Detection")
        # tello.move_forward(200)
        # Move forward to detect AprilTags
        dp = tello_command(tello, ("forward", 200))
        drone_wpose_ct += dp
        
        # Find known AprilTag for localization
        localization_found = False
        search_attempts = 0
        
        while not localization_found and search_attempts < 12:  # 360 degrees / 30 degrees = 12
            tags_info = detect_apriltag(frame_read, at_detector, camera_params, tag_size)
            
            if tags_info:
                for tag_info in tags_info:
                    if tag_info['id'] in final_setting.ar_word.keys():
                        # Found known tag for localization
                        detected_tag_id = tag_info['id']
                        at_pose = tag_info['pose']
                        drone_wpose_at = calculate_drone_position(detected_tag_id, at_pose, final_setting.ar_word, drone_yaw)
                        
                        if drone_wpose_at is not None:
                            # Update Kalman filter
                            KF.predict(np.expand_dims(dp, axis=1))
                            drone_wpose_kf = KF.update(np.expand_dims(np.append(drone_wpose_at, 0), axis=1))
                            localization_found = True
                            print(f"Localized using tag {detected_tag_id} at drone position: {drone_wpose_at}, yaw: {drone_yaw}°")
                            break
            
            if not localization_found:
                # Rotate to search for tags
                dp, dyaw = tello_command(tello, ("cw", 30))
                drone_wpose_ct += dp
                drone_yaw += dyaw
                search_attempts += 1
        
        if not localization_found:
            print("Failed to localize using AprilTags")
            drone_wpose_at = drone_wpose_ct[:2]  # Use control model as fallback
        
        # Phase 3: Scan for Unknown AprilTags
        print("Phase 3: Scanning for Unknown AprilTags")
        
        for scan_step in range(12):  # 360-degree scan
            tags_info = detect_apriltag(frame_read, at_detector, camera_params, tag_size)
            
            if tags_info:
                for tag_info in tags_info:
                    tag_id = tag_info['id']
                    
                    if tag_id not in final_setting.ar_word.keys():
                        # Unknown tag found
                        if tag_id not in unknown_tags:
                            unknown_tags[tag_id] = []
                        
                        # Calculate unknown tag's world position
                        # tag_world_pos = drone_wpose_at + np.array([tag_info['pose'][0], tag_info['pose'][2]])
                        tag_world_pos = np.array(drone_wpose_kf[:2]).flatten() + np.array([tag_info['pose'][0], tag_info['pose'][2]]) # try to use kalman filter position
                        unknown_tags[tag_id].append(tag_world_pos)
                        print(f"Unknown tag {tag_id} detected at: {tag_world_pos}")
                    
                    else:
                        # Update localization with known tag
                        drone_wpose_at = calculate_drone_position(tag_id, tag_info['pose'], final_setting.ar_word)
                        if drone_wpose_at is not None:
                            KF.predict(np.zeros((3, 1)))
                            drone_wpose_kf = KF.update(np.expand_dims(np.append(drone_wpose_at, 0), axis=1))
            
            # Record trajectory
            d_wposes_ct.append(drone_wpose_ct[:2].copy())
            d_wposes_at.append(drone_wpose_at.copy() if drone_wpose_at is not None else [0, 0])
            d_wposes_kf.append(drone_wpose_kf[:2].flatten() if drone_wpose_kf is not None else [0, 0])
            
            # Rotate for next scan
            if scan_step < 11:
                dp = tello_command(tello, ("cw", 30))
                drone_wpose_ct += dp
        
        # Average unknown tag positions
        for tag_id in unknown_tags:
            if unknown_tags[tag_id]:
                avg_pos = np.mean(unknown_tags[tag_id], axis=0)
                print(f"Unknown tag {tag_id} average position: {avg_pos}")
        
        # Phase 4: Navigate to Landing Spot
        print("Phase 4: Navigate to Landing Spot")
        landing_spot = (0.5, 2.0)
        if landing_spot:
            current_pos = drone_wpose_kf[:2].flatten() if drone_wpose_kf is not None else drone_wpose_at
            target_pos = np.array(landing_spot)
            
            # Move to target position
            final_pos = move_to_target(tello, current_pos, target_pos, KF)
            
            # Fine adjustment for landing accuracy
            print("Fine adjustment for landing...")
            time.sleep(1)
            
            # Final position check
            tags_info = detect_apriltag(frame_read, at_detector, camera_params, tag_size)
            if tags_info:
                for tag_info in tags_info:
                    if tag_info['id'] in final_setting.ar_word.keys():
                        refined_pos = calculate_drone_position(tag_info['id'], tag_info['pose'], final_setting.ar_word)
                        if refined_pos is not None:
                            error = np.linalg.norm(refined_pos - target_pos)
                            print(f"Landing accuracy: {error:.3f}m")
        
        # Phase 5: Land
        print("Phase 5: Landing")
        tello.land()
        
        # Calculate total time
        total_time = time.time() - start_time
        print(f"Total competition time: {total_time:.2f} seconds")
        
        # Print results
        print("\n=== Competition Results ===")
        print(f"Professor detected: {professor_detected}")
        print(f"Landing spot: {landing_spot}")
        print(f"Unknown tags found: {list(unknown_tags.keys())}")
        print(f"Final position: {final_pos if 'final_pos' in locals() else 'Unknown'}")
        print(f"Total time: {total_time:.2f}s")
        # print("error unknown tags positions:")

        # Calculate errors for known tags
        print("\n=== Error Analysis ===")
        # error200 = sqrt(np.sum((np.array([1.05, 0]) - np.mean(unknown_tags[200], axis=0))**2))
        error200 = np.linalg.norm(np.array([1.05, 0]) - np.mean(unknown_tags.get(200, [[0, 0]]), axis=0))
        print(f"Error for tag 200: {error200:.3f}m")
        error201 = np.linalg.norm(np.array([1.55, 1.07]) - np.mean(unknown_tags.get(201, [[0, 0]]), axis=0))
        print(f"Error for tag 201: {error201:.3f}m")
        error202 = np.linalg.norm(np.array([-1.5, 2.08]) - np.mean(unknown_tags.get(202, [[0, 0]]), axis=0))
        print(f"Error for tag 202: {error202:.3f}m")

        
    except Exception as e:
        print(f"Error during competition: {e}")
        tello.land()
    
    finally:
        # Cleanup
        tello.streamoff()
        cv2.destroyAllWindows()
        
        # Plot trajectory if data available
        if d_wposes_ct and d_wposes_at and d_wposes_kf:
            plot_trajectory(d_wposes_ct, d_wposes_at, d_wposes_kf)

if __name__ == '__main__':
    main()