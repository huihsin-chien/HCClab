import numpy as np

ar_word = { # AprilTag在世界座標的真實位置（x, y）。meters
    "100": np.array([0.0, 0.0]),       # tag_id 100
    "101": np.array([0.8, 0.0]),     # tag_id 101
    "102": np.array([1.30, 0.0]),    # tag_id 102
    "103": np.array([1.55, 1.90]), # tag_id 103
    "104": np.array([1.05, 3.0]),  # tag_id 104
    "105": np.array([-1.0, 1.1]),    # tag_id 105
    "106": np.array([-1.2, 0.0]),    # tag_id 106
    "107": np.array([-0.7, 0.0]),    # tag_id 107
}

landing_spot = None