import numpy as np
import time
from djitellopy import Tello
from pupil_apriltags import Detector
import cv2
import matplotlib.pyplot as plt
import threading
import final_setting  # Import the settings from final_setting.py


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


def calculate_drone_position(known_tag_id, tag_pose, ar_word):
    """Calculate drone position based on known AprilTag"""
    if known_tag_id in ar_word:
        
        # Drone position = Tag world position - offset
        drone_x = ar_word[known_tag_id][0] + tag_pose[0]
        drone_y = ar_word[known_tag_id][1] - tag_pose[2]
        
        return np.array([drone_x, drone_y])
    return None

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


def main():
    camera_params = [1.84825711e+03, 1.85066209e+03, 1.29929007e+03, 9.04726581e+02]  # fx, fy, cx, cy
    tag_size = 0.166  # Tag size in meters
    # AprilTag detector
    at_detector = Detector(families='tag36h11')
     # Kalman Filter
    KF = KalmanFilter()
    
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
    while frame is None and retry_count < 10:
        frame = frame_read.frame
        time.sleep(0.5)
        retry_count += 1
    
    if frame is None:
        print("Failed to get video stream")
        return
    
    # Competition variables
    drone_wpose_ct = np.array([0.0, 0.0, 0.0])  # Control model position
    drone_wpose_at = np.array([0.0, 0.0, 0.0])  # AprilTag position
    drone_wpose_kf = np.array([0.0, 0.0, 0.0])  # Kalman filter position
    d_wposes_ct = []  # Control model trajectory
    d_wposes_at = []  # AprilTag measurement trajectory
    d_wposes_kf = []  # Kalman filter trajectory
    detected_tag_id = None
    unknown_tags_at = {}  # Store unknown tag positions for april tags localization
    unknown_tags_kf = {}  # Store unknown tag positions for kalman filter localization
    
    try:
    # AprilTag Detection and Position Estimation
        # Find known AprilTag for localization
        localization_found = False
        search_attempts = 0
        while not localization_found :
            tags_info = detect_apriltag(frame_read, at_detector, camera_params, tag_size)
            
            if tags_info:
                for tag_info in tags_info:
                    if tag_info['id'] in final_setting.ar_word.keys():
                        # Found known tag for localization
                        detected_tag_id = tag_info['id']
                        at_pose = tag_info['pose']
                        print(f"Detected known tag {detected_tag_id} at pose: {at_pose}")
                        drone_wpose_at = calculate_drone_position(detected_tag_id, at_pose, final_setting.ar_word)
                        print(f"drone_wpose_at: {drone_wpose_at}")
                        if drone_wpose_at is not None:
                            # Update Kalman filter
                            # KF.predict(np.expand_dims(dp, axis=1))
                            drone_wpose_kf = KF.update(np.expand_dims(np.append(drone_wpose_at, 0), axis=1))
                            localization_found = True
                            print(f"drone_wpose_kf: {drone_wpose_kf}")
                            break
            time.sleep(1)
        
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
                        if tag_id not in unknown_tags_at:
                            unknown_tags_at[tag_id] = []
                            unknown_tags_kf[tag_id] = []  # Initialize for Kalman filter
                        
                        # Calculate unknown tag's world position
                        tag_world_pos = drone_wpose_at + np.array([tag_info['pose'][0], tag_info['pose'][2]]) 
                        if drone_wpose_kf is not None:
                            tag_world_pos_kf = np.array(drone_wpose_kf[:2]).flatten() + np.array([tag_info['pose'][0], tag_info['pose'][2]])
                        else:
                            tag_world_pos_kf = tag_world_pos  # fallback

                        unknown_tags_at[tag_id].append(tag_world_pos)
                        unknown_tags_kf[tag_id].append(tag_world_pos_kf) # Store for Kalman filter
                        print(f"Unknown tag {tag_id} detected at: {tag_world_pos}")
                    
                    else:
                        # Update localization with known tag
                        drone_wpose_at = calculate_drone_position(tag_id, tag_info['pose'], final_setting.ar_word)
                        print(f"Known tag {tag_id}, drone_wpose_at: {drone_wpose_at}")
                    
                        if drone_wpose_at is not None:
                            KF.predict(np.zeros((3, 1)))
                            drone_wpose_kf = KF.update(np.expand_dims(np.append(drone_wpose_at, 0), axis=1))
                            print(f"Updated drone_wpose_kf: {drone_wpose_kf}")
            
            # Record trajectory
            d_wposes_ct.append(drone_wpose_ct[:2].copy())
            d_wposes_at.append(drone_wpose_at.copy() if drone_wpose_at is not None else [0, 0])
            d_wposes_kf.append(drone_wpose_kf[:2].flatten() if drone_wpose_kf is not None else [0, 0])
            time.sleep(1)  # Simulate time delay for each scan step
            
        
        # Average unknown tag positions
        for tag_id in unknown_tags_at:
            if unknown_tags_at[tag_id]:
                avg_pos = np.mean(unknown_tags_at[tag_id], axis=0)
                print(f"Unknown tag {tag_id} average position: {avg_pos}")
        
         # Calculate errors for known tags
        print("\n=== Error Analysis: April tag ===")
        # error200 = sqrt(np.sum((np.array([1.05, 0]) - np.mean(unknown_tags_at[200], axis=0))**2))
        error200 = np.linalg.norm(np.array([1.05, 0]) - np.mean(unknown_tags_at.get(200, [[0, 0]]), axis=0))
        print(f"Error for tag 200: {error200:.3f}m")
        error201 = np.linalg.norm(np.array([1.55, 1.07]) - np.mean(unknown_tags_at.get(201, [[0, 0]]), axis=0))
        print(f"Error for tag 201: {error201:.3f}m")
        error202 = np.linalg.norm(np.array([-1.5, 2.08]) - np.mean(unknown_tags_at.get(202, [[0, 0]]), axis=0))
        print(f"Error for tag 202: {error202:.3f}m")

        print("\n=== Error Analysis: Kalman Filter ===")
        # error200_kf = sqrt(np.sum((np.array([1.05, 0]) - np.mean(unknown_tags_kf[200], axis=0))**2))
        error200_kf = np.linalg.norm(np.array([1.05, 0]) - np.mean(unknown_tags_kf.get(200, [[0, 0]]), axis=0))
        print(f"Error for tag 200: {error200_kf:.3f}m")
        error201_kf = np.linalg.norm(np.array([1.55, 1.07]) - np.mean(unknown_tags_kf.get(201, [[0, 0]]), axis=0))
        print(f"Error for tag 201: {error201_kf:.3f}m")
        error202_kf = np.linalg.norm(np.array([-1.5, 2.08]) - np.mean(unknown_tags_kf.get(202, [[0, 0]]), axis=0))
        print(f"Error for tag 202: {error202_kf:.3f}m")
        

        
    except Exception as e:
        print(f"Error during competition: {e}")
        import traceback
        traceback.print_exc()
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