import numpy as np
import time
from djitellopy import Tello
from pupil_apriltags import Detector
import cv2
import matplotlib.pyplot as plt
import final_setting # Import the settings from final_setting.py
class KalmanFilter:
    def __init__(self):
        """
        Parameters:
        -----------
        u : Control input (m x 1)
        z : Observation (k x 1)
        A : State transition matrix (n x n)
        B : Control input model (n x m)
        C : Observation matrix (k x n)
        R : Process noise covariance matrix (n x n)
        Q : Measurement noise covariance matrix (k x k)
        mu : state estimate (n x 1)
        Sigma : state covariance (n x n)
        """
        # TODO_1
        # State dimension: [x, y, z] = 3
        n = 3  # state dimension
        m = 3  # control input dimension
        k = 3  # observation dimension
        self.A = np.eye(n)
        self.B = np.eye(n)
        self.C = np.eye(k)
        self.R = np.diag([0.01, 0.01, 0.01])  # Small process noise
        self.Q = np.diag([0.05, 0.05, 0.05])  # Moderate measurement noise
        self.mu = np.zeros((n, 1))
        self.Sigma = np.eye(n) * 1.0

    def predict(self, u):
        # TODO_2
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
        # TODO_3
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

def Detect_AprilTag(frame_read):
    print("detecting april tag")
    tags_info = []
    frame = frame_read.frame
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
    key = cv2.waitKey(1)
    # 可根據需要決定是否要用 key 來中斷

    return tags_info
    
def tello_command(movement_request):
    dp = [0.0, 0.0]
    cmd, value = movement_request
    if cmd == "forward":
        tello.move_forward(value)
        dp = [0.0, value]
    elif cmd == "back":
        tello.move_back(value)
        dp = [0.0, -value]
    elif cmd == "right":
        tello.move_right(value)
        dp = [value, 0.0]
    elif cmd == "left":
        tello.move_left(value)
        dp = [-value, 0.0]
    elif cmd == "up":
        tello.move_up(value)
        # dp = [0.0, 0.0, value]
    elif cmd == "down":
        tello.move_down(value)
        # dp = [0.0, 0.0, -value]
    elif cmd == "cw":
        tello.rotate_clockwise(value)
        dp = [0.0, 0.0]
    elif cmd == "ccw":
        tello.rotate_counter_clockwise(value)
        dp = [0.0, 0.0]
    time.sleep(1)

    return np.array(dp) * 0.01

def plot_trajectory(control_poses, tag_pose, kalmanfilter_pose):
    poses_ct = np.array(control_poses)
    poses_at = np.array(tag_pose)
    poses_kf = np.array(kalmanfilter_pose)
    plt.figure(figsize=(8, 6))
    plt.plot(poses_ct[:, 0], poses_ct[:, 1], 'ko--', label='Motion Model')
    plt.plot(poses_at[:, 0], poses_at[:, 1], 'rx-', label='AprilTag')
    plt.plot(poses_kf[:, 0], poses_kf[:, 1], 'b^-', label='Kalman Filter')
    plt.title("2D Trajectory Tracking")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
    

if __name__ == '__main__':
    # movement_sequence = [
    # ("forward", 30),
    # ("forward", 20),
    # ("forward", 20),
    # ("left", 20),
    # ("forward", 30),
    # ("right", 20),
    # ("forward", 30),
    # ("right", 20),
    # ("back", 30),
    # ("back", 20),
    # ("back", 20)
    # ]

    # Camera calibration 
    #camera_params = [911.0081620140941, 909.7331518528662, 493.74409668840053, 360.48512415375063]  # fx, fy, cx, cy?? # fx, fy, cx, cy UAV01
    camera_params = [917.0, 917.0, 480.0, 360.0]
    tag_size = 0.166  # 5cm tag size - adjust based on your actual tag?? # meters

    # AprilTag detector
    # at_word = np.array([0.1, 3.0, 0.0]) # AprilTag在世界座標的真實位置（x, y, z）。
    at_detector = Detector(families='tag36h11')

    # Kalman Filter
    KF = KalmanFilter()

    # Initialize
    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")

    tello.streamon()
    frame_read = tello.get_frame_read()
    frame = None
    while frame is None:
        frame = frame_read.frame
    # time.sleep(3)

    tello.takeoff()
    time.sleep(2)

    drone_wpose_ct = np.array([0.0, 0.0]) # 無人機根據運動模型推算的世界座標
    d_wposes_ct = [] # 運動模型軌跡列表。
    d_wposes_at = [] # AprilTag量測軌跡列表
    d_wposes_kf = [] # 卡爾曼濾波軌跡列表
    detected_tag_id = None # 偵測到的 AprilTag id
    landing_spot = None
    unkown_tags = {} # 用來儲存未知的 tag id 與其位置

    # 往前移動, 拍到 tag_id=0 的 AprilTag
    dp = tello_command(("forward", 200))
    drone_wpose_ct += dp # 更新無人機位置
    # 如果偵測到的 aprit tag 

    while detected_tag_id not in final_setting.ar_word.keys():
        # at_pose, detected_tag_id = Detect_AprilTag(frame_read) # AprilTag偵測到的位置（相機座標）return value: (x, y, z),tag_id
        tags_info = Detect_AprilTag(frame_read) # 偵測到的 AprilTag資訊
        
        if tags_info:
            for i in tags_info:
                if i['id'] in final_setting.ar_word.keys():
                    detected_tag_id = i['id']
                    at_pose = i['pose']
    # 根據特定的 tag id, 計算 drone_wpose_at 的位置 
    drone_wpose_at = [final_setting.ar_word[detected_tag_id][0] + at_detector.tag_size - at_pose[0], \
                      final_setting.ar_word[detected_tag_id][1] - at_pose[2]] # AprilTag在世界座標的真實位置（x, y）
    KF.predict(np.expand_dims(dp, axis=1)) # 卡爾曼濾波預測
    drone_wpose_kf = KF.update(np.expand_dims(drone_wpose_at, axis=1)) # 卡爾曼濾波更新\

    # scan around, find unknow tag
    # unkown tags 可能跟已知的 ar_word 在同一畫面，也可能不在同一畫面。如果在不同畫面，那就用無人機本身的位置推算unknow tag的位置
    # 使用當前畫面，根據畫面上已知的 tag 定位自己，再用自己的位置尋找並定位未知的 tag, 然後旋轉 60 度，再重複上述動作

    for i in range(6):
        # 偵測 AprilTag
        tags_info = Detect_AprilTag(frame_read) # 偵測到的 AprilTag資訊
        if tags_info:
            for tag_info in tags_info:
                if tag_info['id'] not in final_setting.ar_word.keys():
                    # 如果偵測到未知的 tag, 則儲存其位置
                    if tag_info['id'] not in unkown_tags:
                        unkown_tags[tag_info['id']] = []
                    unkown_tags[tag_info['id']].append(np.array(drone_wpose_at) + np.array(tag_info['pose'][:2]))
                else:
                    # 如果偵測到已知的 tag, 則更新無人機位置
                    drone_wpose_at = [final_setting.ar_word[tag_info['id']][0] + at_detector.tag_size - tag_info['pose'][0], \
                                      final_setting.ar_word[tag_info['id']][1] - tag_info['pose'][2]]
                    drone_wpose_kf = KF.update(np.expand_dims(drone_wpose_at, axis=1)) # 卡爾曼濾波更新
        tello.rotate_clockwise(60) # 旋轉 60 度
        tags_info = Detect_AprilTag(frame_read) # 偵測到的 AprilTag資訊
        if tags_info:
            for tag_info in tags_info:
                if tag_info['id'] in final_setting.ar_word.keys():
                   drone_wpose_at = [final_setting.ar_word[tag_info['id']][0] + at_detector.tag_size - tag_info['pose'][0], \
                                      final_setting.ar_word[tag_info['id']][1] - tag_info['pose'][2]]

        d_wposes_ct.append(drone_wpose_ct.copy())
        d_wposes_at.append(drone_wpose_at)
        d_wposes_kf.append(drone_wpose_kf.flatten())


    



    print(f"drone_wpose_ct: {drone_wpose_ct}")
    print(f"drone_wpose_at: {drone_wpose_at}")
    print(f"drone_wpose_kf: {drone_wpose_kf.flatten()}")
    print(f"detected_tag_id: {detected_tag_id}")
    print(f"unkown_tags: {unkown_tags}")
        

    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()

# === Plot 2D trajectory ===
    plot_trajectory(d_wposes_ct, d_wposes_at, d_wposes_kf)

