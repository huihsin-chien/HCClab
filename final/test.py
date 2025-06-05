import numpy as np

# Kalman Filter 簡化實作
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


# 已知 Tag 的世界座標 (tag_id: [x, y, z])
known_tags_world = {
    0: np.array([0.0, 0.0, 0.0]),
    1: np.array([0.5, 0.0, 0.0]),
    2: np.array([1.0, 0.0, 0.0]),
    3: np.array([1.5, 0.0, 0.0]),
    4: np.array([0.0, 0.5, 0.0]),
    5: np.array([0.5, 0.5, 0.0]),
    6: np.array([1.0, 0.5, 0.0]),
    7: np.array([1.5, 0.5, 0.0]),
    8: np.array([2.0, 0.5, 0.0]),
}

unknown_tags = {}

# 根據已知Tag反推出無人機位置（以平均值推估）
def estimate_drone_pose_from_known_tags(detected_tags):
    positions = []
    for tag in detected_tags:
        if tag.tag_id in known_tags_world and tag.pose_t is not None:
            tag_world = known_tags_world[tag.tag_id]
            camera_pos_in_world = tag_world - tag.pose_t.flatten()
            positions.append(camera_pos_in_world)
    if positions:
        return np.mean(positions, axis=0)
    return None

# 推估未知Tag世界座標位置
def collect_unknown_tag_positions(detected_tags, drone_pos):
    for tag in detected_tags:
        if tag.tag_id not in known_tags_world and tag.pose_t is not None:
            tag_pos_camera = tag.pose_t.flatten()
            tag_world_pos = drone_pos + tag_pos_camera
            if tag.tag_id not in unknown_tags:
                unknown_tags[tag.tag_id] = []
            unknown_tags[tag.tag_id].append(tag_world_pos)

# 模擬一段無人機飛行時的 Tag 偵測資料
KF = KalmanFilter()
position_list = []

# 模擬 10 個影格
for frame_index in range(10):
    # 假設每幀看到一個已知Tag（id=0）和一個未知Tag（id=10）
    class Tag:
        def __init__(self, tag_id, pose_t):
            self.tag_id = tag_id
            self.pose_t = pose_t

    tags = [
        Tag(0, np.array([[0.1], [0.0], [0.0]])),
        Tag(10, np.array([[0.3], [0.2], [0.0]]))  # 未知Tag
    ]

    # 使用已知Tag估計無人機位置
    drone_pose = estimate_drone_pose_from_known_tags(tags)
    control_input = np.zeros((3, 1))  # 假設無運動控制

    if drone_pose is not None:
        KF.predict(control_input)
        est_pose = KF.update(drone_pose.reshape(3, 1))
        collect_unknown_tag_positions(tags, est_pose.flatten())
    else:
        KF.predict(control_input)
        est_pose = KF.x.flatten()

    position_list.append(est_pose.copy())

# 統計每個未知Tag的平均世界位置
results = {}
for tag_id, poses in unknown_tags.items():
    mean_pos = np.mean(poses, axis=0)
    results[tag_id] = mean_pos

# 輸出未知Tag的世界座標估計
for tag_id, pos in results.items():
    print(f"未知 Tag {tag_id} 的估計位置為：{pos}")
