import numpy as np
import time
from djitellopy import Tello
from pupil_apriltags import Detector
import cv2
import matplotlib.pyplot as plt
import threading
import final_setting  # Import the settings from final_setting.py

def main():
    camera_params = [1.84825711e+03, 1.85066209e+03, 1.29929007e+03, 9.04726581e+02]  # fx, fy, cx, cy
    tag_size = 0.166  # Tag size in meters
    # AprilTag detector
    at_detector = Detector(families='tag36h11')