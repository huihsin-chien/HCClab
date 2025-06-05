import cv2
import numpy as np
from djitellopy import Tello
from pupil_apriltags import Detector

# ------------- Configuration -------------
TAG_SIZE = 0.16  # meters
CAMERA_PARAMS = [920, 920, 480, 360]  # fx, fy, cx, cy (example values)

KNOWN_TAGS = {
    0: np.array([0.0, 0.0, 0.0]),
    1: np.array([1.0, 0.0, 0.0]),
    2: np.array([2.0, 0.0, 0.0]),
    3: np.array([0.0, 1.0, 0.0]),
    4: np.array([1.0, 1.0, 0.0]),
    5: np.array([2.0, 1.0, 0.0]),
    6: np.array([0.0, 2.0, 0.0]),
    7: np.array([1.0, 2.0, 0.0]),
    8: np.array([2.0, 2.0, 0.0]),
}
UNKNOWN_TAG_IDS = [9, 10, 11]


# ------------- Kalman Filter Class -------------
class TagKalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        for i in range(3):
            self.kf.transitionMatrix[i, i + 3] = 1
        self.kf.measurementMatrix = np.hstack([np.eye(3), np.zeros((3, 3))])
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)

    def update(self, measurement):
        measurement = np.array(measurement, dtype=np.float32).reshape((3, 1))
        self.kf.correct(measurement)
        pred = self.kf.predict()
        return pred[:3].flatten()


# ------------- Utilities -------------
def get_relative_tag_pose(tag):
    t = tag.pose_t.reshape((3,))
    R = tag.pose_R
    return t, R


def get_drone_world_pose(tag_id, t_rel, R_rel):
    tag_pos_world = KNOWN_TAGS[tag_id]
    R_world_to_cam = R_rel.T
    cam_pos_in_world = -R_world_to_cam @ t_rel + tag_pos_world
    return cam_pos_in_world, R_world_to_cam


def transform_to_world(drone_pos, drone_rot, tag_cam_pos):
    return drone_rot @ tag_cam_pos + drone_pos

# Kalman Filter 簡化實作
class KalmanFilter:
    def __init__(self):
        self.x = np.zeros((3, 1))  # [x, y, theta] (只用前兩項)
    
    def predict(self, u):
        # 預測步驟可以根據運動模型寫更複雜，這裡簡化
        self.x += u
    
    def update(self, z):
        self.x = z  # 簡單地取代更新
        return self.x

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



# ------------- Main Program -------------
def main():
    tello = Tello()
    tello.connect()
    tello.streamon()
    cap = tello.get_frame_read()

    detector = Detector(families='tag36h11')
    unknown_tag_filters = {uid: TagKalmanFilter() for uid in UNKNOWN_TAG_IDS}

    while True:
        frame = cap.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray, estimate_tag_pose=True,
                               camera_params=CAMERA_PARAMS, tag_size=TAG_SIZE)

        drone_pos = None # 記錄無人機在絕對坐標中的位置
        drone_rot = None # 記錄無人機絕對坐標中的的旋轉矩陣

        for tag in tags:
            tag_id = tag.tag_id
            t_rel, R_rel = get_relative_tag_pose(tag) #apriltag 相對於無人機的位姿。
            #t_rel：AprilTag 相對於相機的三維位置（translation，平移向量）R_rel：AprilTag 相對於相機的旋轉矩陣（rotation matrix）

            if tag_id in KNOWN_TAGS:
                drone_pos, drone_rot = get_drone_world_pose(tag_id, t_rel, R_rel)
                break

        if drone_pos is not None:
            for tag in tags:
                tag_id = tag.tag_id
                if tag_id in UNKNOWN_TAG_IDS:
                    t_rel, _ = get_relative_tag_pose(tag)
                    tag_world_pos = transform_to_world(drone_pos, drone_rot, t_rel)
                    filtered_pos = unknown_tag_filters[tag_id].update(tag_world_pos)
                    print(f"Filtered Position of Tag {tag_id}: {filtered_pos}")

        cv2.imshow("Tello View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()


main()
