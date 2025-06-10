import cv2
import numpy as np
import matplotlib.pyplot as plt
from pupil_apriltags import Detector
from djitellopy import Tello
import time

# === AprilTag 世界座標 ===
apriltag_world_coords = {
    100: np.array([0.0, 0.0, 0.0]),
    101: np.array([0.8, 0.0, 0.0]),
    102: np.array([1.3, 0.0, 0.0]),
    103: np.array([1.55, 1.9, 0.0]),
    104: np.array([1.05, 3.0, 0.0]),
    105: np.array([-1.5, 1.1, 0.0]),
    106: np.array([-1.2, 0.0, 0.0]),
    107: np.array([-0.7, 0.0, 0.0])
}

# === 相機參數 ===
camera_matrix = np.array([[911.0081620140941, 0.00000000e+00, 493.74409668840053],
                          [0.00000000e+00, 909.7331518528662, 360.48512415375063],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# dist_coeffs = np.array([0.06537366, -0.14799897, -0.01066338, 0.0006893, 0.64260302])

# === 建立 AprilTag 偵測器 ===
detector = Detector(families='tag36h11')

# === 初始化 Tello ===
tello = Tello()
tello.connect()
tello.streamon()
tello.takeoff()
time.sleep(1)

tello.move_forward(200)
time.sleep(0.5)
tello.move_up(50)
time.sleep(1)

target_pos = np.array([0.75, 2.25, 0.0])
landed = False
positions = []

def estimate_pose(tag_corners, tag_size=0.166):
    object_points = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [ tag_size/2, -tag_size/2, 0],
        [ tag_size/2,  tag_size/2, 0],
        [-tag_size/2,  tag_size/2, 0]
    ], dtype=np.float32)

    image_points = np.array(tag_corners, dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distCoeffs=None)
    if not success:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec


# === 主迴圈 ===
while True:
    frame = tello.get_frame_read().frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)

    camera_points = []

    for tag in tags:
        tag_id = tag.tag_id
        if tag_id not in apriltag_world_coords:
            continue

        R, t = estimate_pose(tag.corners)
        if R is None:
            continue

        camera_pos_tag = -R.T @ t  # 相機在 tag 座標下的位置
        tag_world = apriltag_world_coords[tag_id][:3].reshape(3, 1)  # Tag 在世界座標的位置 (column vector)

        # 相機座標系 ➜ 世界座標系
        x_world = -camera_pos_tag[0][0] + tag_world[0][0]
        y_world = camera_pos_tag[2][0] + tag_world[1][0]

        cam_pos_world = np.array([x_world, y_world, 0.0])
        camera_points.append(cam_pos_world.flatten())

        for pt in tag.corners:
            pt = tuple(map(int, pt))
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
        cv2.putText(frame, f"ID: {tag_id}", tuple(map(int, tag.center)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if camera_points:
        cam_pos = np.mean(camera_points, axis=0)[:2]


        # print(f"🔵 Raw Position: {cam_pos}, Filtered Position: {filtered_pos}")
        print(f"🔵 Raw Position: {cam_pos}")

        # dx = target_pos[0] - filtered_pos[0]
        # dy = target_pos[1] - filtered_pos[1]
        dx = target_pos[0] - cam_pos[0]
        dy = target_pos[1] - cam_pos[1]
        distance = np.linalg.norm([dx, dy])

        print(f"📏 距離目標: {distance:.2f} m")
        print(f"➡️ 移動向量 dx: {dx:.2f}, dy: {dy:.2f}")

        if distance < 0.1:
            print("✅ 到達目標，準備降落...")
            tello.land()
            landed = True
            break

        move_x = int(np.clip(dx * 100, -50, 50))  # 前後
        move_y = int(np.clip(dy * 100, -50, 50))  # 左右

        if abs(move_y) > 20:
            if move_y > 0:
                tello.move_back(abs(move_y))
            else:
                tello.move_forward(abs(move_y))
        elif abs(move_y) > 10:
            tello.move_forward(abs(2*move_y))

        if abs(move_x) > 20:
            if move_x > 0:
                tello.move_left(abs(move_x))
            else:
                tello.move_right(abs(move_x))
        elif abs(move_x) > 10:
            tello.move_right(abs(2*move_x))
        time.sleep(1)
    # else:
        # print("⚠️ 沒有偵測到任何 AprilTag")

    cv2.imshow("Tello AprilTag Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if not landed:
    tello.land()
tello.streamoff()
tello.end()
cv2.destroyAllWindows()

