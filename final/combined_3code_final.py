import cv2
import numpy as np
import time
from djitellopy import Tello
from pupil_apriltags import Detector
from ultralytics import YOLO

# === Project‑specific configuration comes from a single settings module ===
import final_rewrite_setting as frs  # exposes ar_word, professor_landing_spots, unkonwn_tags, real_unknown_tags

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

########################################################
# === Helper: send command & return displacement (m) ===
########################################################

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

########################################################
# === Helper: AprilTag detection (single frame)     ===
########################################################

def detect_apriltags(detector: Detector, frame) -> list[dict]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray, estimate_tag_pose=True, camera_params=CAMERA_PARAMS, tag_size=TAG_SIZE)
    return [{"id": d.tag_id, "pose": d.pose_t.flatten()} for d in detections if d.pose_t is not None]

########################################################
# === Average AprilTag pose over N frames           ===
########################################################

def average_apriltag_pose(frame_read, detector: Detector, samples: int = 10):
    accum: dict[int, np.ndarray] = {}
    cnt: dict[int, int] = {}
    for _ in range(samples):
        if frame_read.stopped: break
        for tag in detect_apriltags(detector, frame_read.frame):
            accum[tag["id"]] = accum.get(tag["id"], 0) + tag["pose"]
            cnt[tag["id"]] = cnt.get(tag["id"], 0) + 1
        time.sleep(0.1)
    return [{"id": tid, "pose": accum[tid]/cnt[tid]} for tid in accum]

#########################################
# === Phase 1: recognise professor     ===
#########################################

def recognise_professor(tello: Tello, yolo_model: YOLO, attempts=20, consecutive=4, img_size=640):
    fr = tello.get_frame_read(); last_lbl, streak = None, 0
    for i in range(attempts):
        frame = fr.frame
        if frame is None: time.sleep(0.2); continue
        frame_resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (img_size, img_size))
        res = yolo_model.predict(frame_resized, conf=0.25, iou=0.5, imgsz=img_size, device="cpu", save=False)[0]
        if len(res.boxes) == 0:
            last_lbl, streak = None, 0
            print(f"[Prof] attempt {i+1}: none")
            time.sleep(0.4); continue
        best = max(res.boxes, key=lambda b: b.conf.item())
        lbl = yolo_model.names[int(best.cls)]
        if lbl == last_lbl:
            streak += 1
        else:
            last_lbl, streak = lbl, 1
        print(f"[Prof] attempt {i+1}: {lbl} ({streak}/{consecutive})")
        if streak >= consecutive:
            return lbl
        time.sleep(0.5)
    return None

########################################################
# === Helper: camera pose from one tag              ===
########################################################

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

########################################################
# === Phase 2: map unknown tags on 4 walls          ===
########################################################

def map_unknown_tags(tello: Tello, frame_read, detector: Detector):
    locations = []
    for wall in ["wall_1", "wall_2", "wall_3", "wall_4"]:
        expected = frs.unkonwn_tags.get(wall, [])
        print(f"[Map] {wall} – expected {expected}")
        if not expected:
            tello_command(tello, ("ccw", 90)); continue
        if wall != "wall_1":
            tello_command(tello, ("back", 50))
        avg_tags = average_apriltag_pose(frame_read, detector, 10)
        ref_tag = next((t for t in avg_tags if t["id"] in frs.ar_word), None)
        for t in avg_tags:
            if t["id"] not in expected: continue
            pose = t["pose"]
            if wall == "wall_1":
                x = -(pose[0] - (ref_tag["pose"][0] if ref_tag else 0)) + (frs.ar_word.get(ref_tag["id"], [0,0])[0] if ref_tag else 0)
                y = 0.0
            elif wall == "wall_2":
                x, y = 1.55, -(pose[0] - (ref_tag["pose"][0] if ref_tag else 0)) + (frs.ar_word.get(ref_tag["id"], [0,0])[1] if ref_tag else 0)
            elif wall == "wall_3":
                x = (pose[0] - (ref_tag["pose"][0] if ref_tag else 0)) - (frs.ar_word.get(ref_tag["id"], [0,0])[0] if ref_tag else 0)
                y = 3.0
            else:  # wall_4
                x, y = -1.5, (pose[0] - (ref_tag["pose"][0] if ref_tag else 0)) + (frs.ar_word.get(ref_tag["id"], [0,0])[1] if ref_tag else 0)
            locations.append((t["id"], x, y))
            print(f"[Map] tag {t['id']} @ ({x:.2f}, {y:.2f})")
        tello_command(tello, ("forward", 150 if wall == "wall_1" else 50))
        tello_command(tello, ("ccw", 90))
    return locations

########################################################
# === Phase 3: navigate to landing spot & land      ===
########################################################

def navigate_and_land(tello: Tello, frame_read, detector: Detector, target: np.ndarray):
    while True:
        frame = frame_read.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)
        camera_points = []

        for tag in tags:
            tag_id = tag.tag_id
            if tag_id not in APRILTAG_WORLD_COORDS:
                continue
            R, t = estimate_pose(tag.corners)
            if R is None:
                continue

            camera_pos_tag = -R.T @ t
            tag_world = APRILTAG_WORLD_COORDS[tag_id][:3].reshape(3, 1)

            x_world = -camera_pos_tag[0][0] + tag_world[0][0]
            y_world = camera_pos_tag[2][0] + tag_world[1][0]

            cam_pos_world = np.array([x_world, y_world, 0.0])
            camera_points.append(cam_pos_world.flatten())

        if camera_points:
            cam_pos = np.mean(camera_points, axis=0)[:2]
            print(f"[Land] Current Position: {cam_pos}")
            if cam_pos is None:
                print("[Land] no ref tag – hover")
                tello_command(tello, ("back", 20))
                time.sleep(0.5); continue
            dx = target[0] - cam_pos[0]
            dy = target[1] - cam_pos[1]
            distance = np.linalg.norm([dx, dy])
            print(f"[Land] Δ({dx:.2f} ,  {dy:.2f}) d = {distance:.2f} m")

            if distance < 0.1:
                print("[Land] arrive target, really to land...")
                tello.land()
                return True

            move_x = int(np.clip(dx * 100, -50, 50))
            move_y = int(np.clip(dy * 100, -50, 50))
            if abs(move_y) > 20: tello_command(tello, ("forward" if move_y < 0 else "back", abs(move_y)))
            elif abs(move_y) > 10: tello_command(tello, ("back", 20))
            if abs(move_x) > 20: tello_command(tello, ("right" if move_x < 0 else "left", abs(move_x)))
            elif abs(move_x) > 10: tello_command(tello, ("right", 20))
            time.sleep(0.5)

###################
# === MAIN =======
###################

def main():
    start = time.time()
    tello = Tello(); tello.connect()
    if (bat:=tello.get_battery()) < 20:
        print(f"[LowBat] {bat}% – abort"); return
    print(f"[INFO] Battery {bat}%")
    tello.streamon(); fr = tello.get_frame_read()

    detector = Detector(families="tag36h11")
    yolo_model = YOLO("best.pt")

    tello.takeoff(); time.sleep(1)

    try:
        # Phase 1: recognise professor
        print("[INFO] Recognising professor...")
        label = recognise_professor(tello, yolo_model)
        if label is None or label not in PROFESSOR_LANDING_SPOTS:
            raise RuntimeError("Professor not found → abort")
        target_pos = np.array(PROFESSOR_LANDING_SPOTS[label] + [0])
        print(f"[TARGET] {label}@{target_pos}")

        # Phase 2: map unknown tags
        print("[INFO] Mapping unknown tags...")
        tello_command(tello, ("up", 50))
        tello_command(tello, ("forward", 80))
        unknowns = map_unknown_tags(tello, fr, detector)
        print("[RESULT] Unknown tag positions:")
        for tid, x, y in unknowns:
            err = np.linalg.norm(np.array([x, y]) - frs.real_unknown_tags.get(tid, np.zeros(2)))
            print(f"  id {tid}: ({x:.2f},{y:.2f}) err {err:.2f}m")

        # Phase 3: navigate to landing spot
        tello_command(tello, ("back", 100))
        target_pos = np.array(PROFESSOR_LANDING_SPOTS[label] + [0])
        print("[INFO] Navigating to landing spot...")
        navigate_and_land(tello, fr, detector, target_pos)

    except Exception as e:
        import traceback; print(f"[EXCEPT] {e}"); traceback.print_exc(); tello.land()
    finally:
        tello.streamoff(); tello.end(); cv2.destroyAllWindows(); print(f"[TIME] {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
