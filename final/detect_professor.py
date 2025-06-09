import cv2
import time
import numpy as np
from ultralytics import YOLO
from djitellopy import Tello
import random

COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

class ProfessorClassifier:
    def __init__(
        self,
        weights: str = "bests.pt",
        img_size: int = 640,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.5,
        device: str = "cpu"
    ):
        self.weights = weights
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.model = YOLO(self.weights)
        self.landing_positions = {
            'hhs': (75, 225),
            'lcw': (0, 150),
            'lwk': (0, 75),
            'ccw': (-75, 225)
        }
        self.last_label = None
        self.same_label_count = 0
        self.detected_label = None

    def predict_frame(self, frame):
        """
        預測輸入影像中的教授標籤。
        回傳：偵測到的標籤名稱或 None
        """
        # 影像預處理
        img = cv2.resize(frame, (self.img_size, self.img_size))
        
        results = self.model.predict(
            source=img,
            conf=self.conf_threshold,
            iou=self.nms_threshold,
            imgsz=self.img_size,
            device=self.device,
            save=False
        )[0]

        if len(results.boxes) == 0:
            # 沒有偵測到任何物件
            self.same_label_count = 0
            self.last_label = None
            return None

        # 取最高信心分數的框
        best_box = max(results.boxes, key=lambda b: b.conf.item())
        best_label_name = self.model.names[int(best_box.cls)]
        best_label_conf = best_box.conf.item()

        # 持續確認同一標籤
        if best_label_name == self.last_label:
            self.same_label_count += 1
        else:
            self.same_label_count = 1
            self.last_label = best_label_name

        if self.same_label_count >= 5:
            self.detected_label = best_label_name
            return best_label_name

        return None
    
class CompetitionDrone:
    def detect_and_classify_image(self, max_attempts=10):
        """Detect and classify professor image"""
        print("Starting image recognition...")

        for attempt in range(max_attempts):
            frame = self.frame_read.frame
            if frame is not None:
                # Preprocess image for better recognition
                enhanced_frame = self.enhance_image(frame)

                # Classify professor (with stability check inside)
                professor = self.professor_classifier.predict_frame(enhanced_frame)

                if professor:
                    self.target_professor = professor
                    self.landing_position = self.professor_classifier.landing_positions[professor]
                    print(f"✅ Recognized professor: {professor}")
                    print(f"🎯 Target landing position: {self.landing_position}")
                    self.task_scores['recognition'] = 20
                    return True

            time.sleep(0.5)  # 0.5 秒偵測一次

        print("❌ Failed to recognize professor within time limit")
        return False
