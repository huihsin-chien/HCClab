import cv2
import tyro
from ultralytics import YOLO
from djitellopy import Tello
import time
import random

def main(weights: str = "best.pt", img_size: int = 640, conf_threshold: float = 0.25, nms_threshold: float = 0.5, device: str = "cpu"):
    # 連接 Tello
    tello = Tello()
    tello.connect()
    print(f"✅ 已連接 Tello，電池電量：{tello.get_battery()}%")
    tello.streamon()
    # tello.takeoff()

    model = YOLO(weights)

    last_check_time = time.time()
    last_label = None
    same_label_count = 0

    try:
        while True:
            frame = tello.get_frame_read().frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (img_size, img_size))

            # YOLO 預測
            results = model.predict(source=frame, conf=conf_threshold, iou=nms_threshold, imgsz=img_size, device=device, save=False)[0]

            # 找出最高置信度的標籤
            if len(results.boxes) > 0:
                best_box = max(results.boxes, key=lambda b: b.conf.item())
                best_label_name = model.names[int(best_box.cls)]
                best_label_conf = best_box.conf.item()
                best_label = f"{best_label_name} {best_label_conf:.2f}"

                now = time.time()
                if now - last_check_time >= 0.5:
                    print("🎯 畫面中最可能的物件是：", best_label)
                    if best_label_name == last_label:
                        same_label_count += 1
                    else:
                        same_label_count = 1
                        last_label = best_label_name
                    last_check_time = now

                    if same_label_count >= 5:
                        print("✅ 連續 2.5 秒是同一個標籤")
                        return
    finally:
        tello.land()
        tello.streamoff()
        tello.end()
        print("🛬 已降落並關閉 Tello")

if __name__ == "__main__":
    tyro.cli(main)
