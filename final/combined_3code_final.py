import cv2
import numpy as np
import time
from djitellopy import Tello
from pupil_apriltags import Detector
from ultralytics import YOLO

# === Project‑specific configuration comes from a single settings module ===
import final_rewrite_setting as frs  # exposes ar_word, professor_landing_spots, unkonwn_tags, real_unknown_tags
import final_utile as fu  # exposes detect_apriltags, align_tag







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
# === Phase 2: map unknown tags on 4 walls          ===
########################################################

def map_unknown_tags(tello: Tello, frame_read, detector: Detector):
    locations = []
    for wall in ["wall_1", "wall_2", "wall_3", "wall_4"]:
        expected = frs.unkonwn_tags.get(wall, [])
        print(f"[Map] {wall} – expected {expected}")
        if not expected:
            fu.tello_command(tello, ("ccw", 90)); continue
        if wall != "wall_1":
            fu.tello_command(tello, ("back", 50))
        avg_tags = fu.average_apriltag_detection(frame_read, detector, 10)
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
        fu.tello_command(tello, ("forward", 150 if wall == "wall_1" else 50))
        fu.tello_command(tello, ("ccw", 90))
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
            if tag_id not in fu.APRILTAG_WORLD_COORDS:
                continue
            R, t = fu.estimate_pose(tag.corners)
            if R is None:
                continue

            camera_pos_tag = -R.T @ t
            tag_world = fu.APRILTAG_WORLD_COORDS[tag_id][:3].reshape(3, 1)

            x_world = -camera_pos_tag[0][0] + tag_world[0][0]
            y_world = camera_pos_tag[2][0] + tag_world[1][0]

            cam_pos_world = np.array([x_world, y_world, 0.0])
            camera_points.append(cam_pos_world.flatten())

        if camera_points:
            cam_pos = np.mean(camera_points, axis=0)[:2]
            print(f"[Land] Current Position: {cam_pos}")
            if cam_pos is None:
                print("[Land] no ref tag – hover")
                fu.tello_command(tello, ("back", 20))
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
            if abs(move_y) > 20: fu.tello_command(tello, ("forward" if move_y < 0 else "back", abs(move_y)))
            elif abs(move_y) > 10: fu.tello_command(tello, ("back", 20))
            if abs(move_x) > 20: fu.tello_command(tello, ("right" if move_x < 0 else "left", abs(move_x)))
            elif abs(move_x) > 10: fu.tello_command(tello, ("right", 20))
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
        if label is None or label not in fu.PROFESSOR_LANDING_SPOTS:
            raise RuntimeError("Professor not found → abort")
        target_pos = np.array(fu.PROFESSOR_LANDING_SPOTS[label] + [0])
        print(f"[TARGET] {label}@{target_pos}")

        # Phase 2: map unknown tags
        print("[INFO] Mapping unknown tags...")
        fu.tello_command(tello, ("up", 50))
        fu.tello_command(tello, ("forward", 80))
        unknowns = map_unknown_tags(tello, fr, detector)
        print("[RESULT] Unknown tag positions:")
        for tid, x, y in unknowns:
            err = np.linalg.norm(np.array([x, y]) - frs.real_unknown_tags.get(tid, np.zeros(2)))
            print(f"  id {tid}: ({x:.2f},{y:.2f}) err {err:.2f}m")

        # Phase 3: navigate to landing spot
        fu.tello_command(tello, ("back", 100))
        target_pos = np.array(fu.PROFESSOR_LANDING_SPOTS[label] + [0])
        print("[INFO] Navigating to landing spot...")
        navigate_and_land(tello, fr, detector, target_pos)

    except Exception as e:
        import traceback; print(f"[EXCEPT] {e}"); traceback.print_exc(); tello.land()
    finally:
        tello.streamoff(); tello.end(); cv2.destroyAllWindows(); print(f"[TIME] {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
