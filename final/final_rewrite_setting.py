import numpy as np

ar_word = { # AprilTag在世界座標的真實位置（x, y）。meters
    100: np.array([0.0, 0.0]),       # tag_id 100, wall 1
    101: np.array([0.8, 0.0]),     # tag_id 101, wall 1
    102: np.array([1.30, 0.0]),    # tag_id 102, wa11 1
    103: np.array([1.55, 1.90]), # tag_id 103, wall 2
    104: np.array([1.05, 3.0]),  # tag_id 104, wall 3
    105: np.array([-1.50, 1.1]),    # tag_id 105, wall 4
    106: np.array([-1.2, 0.0]),    # tag_id 106, wall 1
    107: np.array([-0.7, 0.0]),    # tag_id 107, wall 1
}

#200: 1.05, 0
#201: 1.55, 1.07
#202: -1.5, 2.08
# 起飛到中間：2.51

# DEBUG 用：109 110 111 112
# 109: wall 2
# 110 111: wall 3
# 112: wall 4


real_unknown_tags = {
    200: np.array([1.05, 0.0]),  # tag_id 200, wall 3
    201: np.array([1.55, 1.07]), # tag_id 201, wall 2
    202: np.array([-1.5, 2.08]), # tag_id 202, wall 4
    111: np.array([-0.7, 3.0]),  # tag_id 111, wall 3
}


professor_landing_spots = {
    'hhs': [-0.75, 2.25],
    'lcw': [0.75, 0.75], 
    'lwk': [0, 0.75],
    'ccw': [0, 2.25]
}

unkonwn_tags = {
    'wall_1': [],
    'wall_2': [],
    'wall_3': [],
    'wall_4': [],
}

# Landing spots
#                   (0, 0)

# (75, 75)     (0, 75)     (-75, 75)

# (75, 150)   (0, 150)   (-75, 150)

# (75, 225)   (0, 225)   (-75, 225)


landing_spot = None
