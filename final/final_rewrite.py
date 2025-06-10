import numpy as np
import time
from djitellopy import Tello
from pupil_apriltags import Detector
import cv2
import matplotlib.pyplot as plt
import final_rewrite_setting 

def detect_apriltag(frame_read, at_detector, camera_params, tag_size):
    """Detect AprilTags in the current frame,
       回傳: tags_info (list of dicts with tag_id and pose),
       Each dict contains:
        - 'id': AprilTag ID
        - 'pose': (x, y, z) position relative to camera in meters"""

    print("Detecting AprilTag...")
    tags_info = []
    frame = frame_read.frame
    if frame is None:
        return tags_info
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect tags
    tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

    for tag in tags:
        if tag.pose_t is not None:
            t = tag.pose_t.flatten()  # (x, y, z) relative to camera
            tags_info.append({'id': tag.tag_id, 'pose': t})

            # Draw tag on image
            corners = np.int32(tag.corners)
            center = tuple(np.int32(tag.center))
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {tag.tag_id}", (center[0]+10, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    # Show live feed
    cv2.imshow("Tello Stream with AprilTag", frame)
    #save image
    try:
        cv2.imwrite(f"./tello_apriltag_{time.time()}.jpg", frame)
        
    except Exception as e:
        print(f"Failed to save image: {e}")
        pass
    cv2.waitKey(1)

    return tags_info

def tello_command(tello, movement_request):
    """Execute movement command and return displacement"""
    dp = np.array([0.0, 0.0, 0.0])
    cmd, value = movement_request
    
    try:
        if cmd == "forward":
            tello.move_forward(value)
            dp = np.array([0.0, value * 0.01, 0.0])  # Convert cm to m
        elif cmd == "back":
            tello.move_back(value)
            dp = np.array([0.0, -value * 0.01, 0.0])
        elif cmd == "right":
            tello.move_right(value)
            dp = np.array([value * 0.01, 0.0, 0.0])
        elif cmd == "left":
            tello.move_left(value)
            dp = np.array([-value * 0.01, 0.0, 0.0])
        elif cmd == "up":
            tello.move_up(value)
            dp = np.array([0.0, 0.0, value * 0.01])
        elif cmd == "down":
            tello.move_down(value)
            dp = np.array([0.0, 0.0, -value * 0.01])
        elif cmd == "cw":
            tello.rotate_clockwise(value)
            dp = np.array([0.0, 0.0, 0.0])
        elif cmd == "ccw":
            tello.rotate_counter_clockwise(value)
            dp = np.array([0.0, 0.0, 0.0])
        
        time.sleep(1.5)  # Wait for movement to complete
    except Exception as e:
        print(f"Movement command failed: {e}")
        
    return dp

# ...existing code...

def average_apriltag_detection(frame_read, at_detector, camera_params, tag_size, num=10):
    """Detect AprilTags multiple times and return average pose for each tag id"""
    tag_accumulator = {}
    tag_counts = {}

    for _ in range(num):
        tags_info = detect_apriltag(frame_read, at_detector, camera_params, tag_size)
        for tag in tags_info:
            tag_id = tag['id']
            pose = np.array(tag['pose'])
            if tag_id not in tag_accumulator:
                tag_accumulator[tag_id] = pose
                tag_counts[tag_id] = 1
            else:
                tag_accumulator[tag_id] += pose
                tag_counts[tag_id] += 1
        time.sleep(0.1)  # 避免太快

    avg_tags = []
    for tag_id in tag_accumulator:
        avg_pose = tag_accumulator[tag_id] / tag_counts[tag_id]
        avg_tags.append({'id': tag_id, 'pose': avg_pose})
    return avg_tags

# 使用方式（在 main 或 for 迴圈裡）：
# tags_info_avg = average_apriltag_detection(frame_read, at_detector, camera_params, tag_size, num=10)



def main():
    start_time = time.time()
    camera_params = [ 917.0, 917.0, 480.0, 360.0]  # fx, fy, cx, cy
    tag_size = 0.166  # Tag size in meters
    unknown_tags_location = []


    # AprilTag detector
    at_detector = Detector(families='tag36h11')
    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
    if tello.get_battery() < 20:
        print("Battery too low! Please charge the drone.")
        return
    tello.streamon()
    frame_read = tello.get_frame_read()
    # Wait for video stream to start
    frame = None
    retry_count = 0
    while frame is None and retry_count < 10:
        frame = frame_read.frame
        time.sleep(0.5)
        retry_count += 1
    
    if frame is None:
        print("Failed to get video stream")
        return
    # Take off
    print("Taking off...")
    tello.takeoff()
    time.sleep(3)
    tello.move_up(50)
    # tello_command(tello, ("move_forward", 100))  
    tello.move_forward(80)
    #------------end of take off----------------

    try:
        # Phase 1: detect professor and landing spot
        #pass

        # Phase 2: detect Unknown apriltags
        # 1. detect tags at wall 1 first, drone will de on the entrence of the playground
            # 如果 Wall 1 存在 tag (unkonwn_tags['wall_1'] != []):
                # 則定位wall 1 上的 tag 的 y 座標為 0.0，利用 apriltag 的位置來定位該 tag 的 x 座標
                # 非 wall 1 上的 unknown tag 則不予理位
                # 前進 100, 轉 90 度 (counterclock wise)，面相 wall 2
            # 如果 wall 1 沒有 tag (unkonwn_tags['wall_1'] == []): 直接轉向 wall 2
        # 2. detect tags at wall 2, drone will be facing wall 2
            # 如果 Wall 2 存在 tag (unkonwn_tags['wall_2'] != []):
                # 往後退，避免視角過小看不到
                # 則定位該 tag 的 x 座標為 1.55，利用 apriltag 的位置來定位該 tag 的 y 座標
                # 非 wall 2 上的 unknown tag 則不予理位
                # 定位完 wall 2 上的 tag 後，前進回到中心點，ccw 90 度，面相 wall 3
            # 如果 wall 2 沒有 tag (unkonwn_tags['wall_2'] == []): 直接轉向 wall 3

        # 3. detect tags at wall 3, drone will be facing wall 3
            # 如果 Wall 3 存在 tag (unkonwn_tags['wall_3'] != []):
                # 往後退，避免視角過小看不到
                # 則定位該 tag 的 y 座標為 3.0，利用 apriltag 的位置來定位該 tag 的 x 座標
                # 非 wall 3 上的 unknown tag 則不予理位
                # 定位完 wall 3 上的 tag 後，前進回到中心點，ccw 90 度，面相 wall 4   
            # 如果 wall 3 沒有 tag (unkonwn_tags['wall_3'] == []): 直接轉向 wall 4
        # 4. detect tags at wall 4, drone will be facing wall 4
             # 如果 Wall 4 存在 tag (unkonwn_tags['wall_4'] != []):
                 # 往後退，避免視角過小看不到
                 # 則定位該 tag 的 x 座標為 -1.5，利用 apriltag 的位置來定位該 tag 的 y 座標
                # 非 wall 4 上的 unknown tag 則不予理位
                # 定位完 wall 4 上的 tag 後，前進回到中心點，ccw 90 度，面相 wall 1
             # 如果 wall 4 沒有 tag (unkonwn_tags['wall_4'] == []): 直接轉向 wall 1 
     
        # Phase 3: landing
        #pass 
        unknown_tags = final_rewrite_setting.unkonwn_tags
        
        for wall, tags in unknown_tags.items():
            print(f"Detecting tags at {wall}...")
            if tags:
                # Move back to avoid small field of view
                if wall != 'wall_1':
                    tello_command(tello, ("back", 50))
                
                
                # Detect AprilTags
                # 取十次平均
                tags_info_avg = average_apriltag_detection(frame_read, at_detector, camera_params, tag_size, num=10)
                known_tag = None
                print(f"tags_info_avg: {tags_info_avg}")
                for tag in tags_info_avg:
                    if tag["id"] in final_rewrite_setting.ar_word:
                        known_tag = tag
                        print(f"Known tag found: {known_tag}")
                        break
            

                for tag in tags_info_avg:
                    if tag['id'] in tags:
                        # Calculate position based on wall
                        if wall == 'wall_1':
                            if known_tag is not None:
                                x = -1 * (tag['pose'][0] - known_tag['pose'][0]) + final_rewrite_setting.ar_word[known_tag["id"]][0]
                            else:
                                x = -1 * tag['pose'][0]  # Use x from pose
                            y = 0.0  # Fixed y for wall 1
                            
                        elif wall == 'wall_2':
                            x = 1.55  # Fixed x for wall 2
                            # y = tag['pose'][1]  # Use y from pose
                            # wall 2 的  y 座標為 已知tag  再加上 -1 * tag['pose'][0]
                            if known_tag is not None:
                                y =  -1 * (tag['pose'][0] - known_tag['pose'][0]) + final_rewrite_setting.ar_word[known_tag["id"]][1]
                            else:
                                y = tag['pose'][0]  
                                print("Warning: No known tag found for wall 2, using tag pose directly.")
                        elif wall == 'wall_3':
                            # x = tag['pose'][0]  # Use x from pose
                            if known_tag is not None:
                                x = -1 * (tag['pose'][0] - known_tag['pose'][0]) + final_rewrite_setting.ar_word[known_tag["id"]][0]
                            else:
                                x = tag['pose'][0] * -1  
                                print("Warning: No known tag found for wall 3, using tag pose directly.")
                            y = 3.0  # Fixed y for wall 3
                        elif wall == 'wall_4':
                            x = -1.5  # Fixed x for wall 4
                            if known_tag is not None:
                                y = (tag['pose'][0] - known_tag['pose'][0]) + final_rewrite_setting.ar_word[known_tag["id"]][1]
                            else:
                                # y = tag['pose'][1] * -1  
                                y = tag['pose'][0] * -1  
                                print("Warning: No known tag found for wall 4, using tag pose directly.")
                            # y = tag['pose'][2]  # Use y from pose
                        
                        print(f"Detected tag {tag['id']} at ({x}, {y}) on {wall}")
                        unknown_tags_location.append((tag['id'], x, y))
                # Move back to center and turn to next wall
                if wall == 'wall_1':
                    tello_command(tello, ("forward", 150))
                else:
                    tello_command(tello, ("forward", 50))
                tello_command(tello, ("ccw", 90))
            else:
                print(f"No tags detected at {wall}, turning to next wall.")
                tello_command(tello, ("ccw", 90))
        
    except Exception as e:
        print(f"Error during competition: {e}")
        import traceback
        traceback.print_exc()
        tello.land()

    finally:
        # Cleanup
        tello.streamoff()
        cv2.destroyAllWindows()
        # print execution time
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        # print unknown tags locations and error
        print("Unknown tags locations:")
        for tag_id, x, y in unknown_tags_location:
            print(f"Tag {tag_id}: ({x}, {y}), error: {np.linalg.norm(np.array([x, y]) - final_rewrite_setting.real_unknown_tags[tag_id]) if tag_id in final_rewrite_setting.real_unknown_tags else 'N/A'}")
        
        


# TODO
# 測試 wall 3 辨識正確
# 調整：若 tello 視窗內沒有辦法把所有已知+未知 tag 照進去，可能要退後或左右移


if __name__ == '__main__':
    main()