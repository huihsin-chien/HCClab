import numpy as np
import time
from djitellopy import Tello
from pupil_apriltags import Detector
import cv2
import matplotlib.pyplot as plt
import final_rewrite_setting 
import os
import final_utile as fu  


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
                if wall == 'wall_2':
                    fu.tello_command(tello, ("back", fu.WALL_2_BACK))
                if wall == 'wall_3':
                    fu.tello_command(tello, ("back", fu.WALL_3_BACK))
                if wall == 'wall_4':
                    fu.tello_command(tello, ("back", fu.WALL_4_BACK))
                
                
                # Detect AprilTags
                # 取十次平均
                tags_info_avg = fu.average_apriltag_detection(frame_read, at_detector, camera_params, tag_size, num=10)
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
                                print("Warning: No known tag found for wall 1, using tag pose directly.")
                            y = 0.0  # Fixed y for wall 1
                            
                        elif wall == 'wall_2':
                            x = 1.55  # Fixed x for wall 2
                            if known_tag is not None:
                                y =  -1 * (tag['pose'][0] - known_tag['pose'][0]) + final_rewrite_setting.ar_word[known_tag["id"]][1]
                            else:
                                y = -1 * tag['pose'][0] + 1.5
                                print("Warning: No known tag found for wall 2, using tag pose directly.")

                        elif wall == 'wall_3':
                            if known_tag is not None:
                                x = (tag['pose'][0] - known_tag['pose'][0]) + final_rewrite_setting.ar_word[known_tag["id"]][0]
                            else:
                                x = -1 * tag['pose'][0] + 1.5 
                                print("Warning: No known tag found for wall 3, using tag pose directly.")
                            y = 3.0  # Fixed y for wall 3
                        elif wall == 'wall_4':
                            x = -1.5  # Fixed x for wall 4
                            if known_tag is not None:
                                y = (tag['pose'][0] - known_tag['pose'][0]) + final_rewrite_setting.ar_word[known_tag["id"]][1]
                            else:
                                y = tag['pose'][0] * -1 + 1.5
                                print("Warning: No known tag found for wall 4, using tag pose directly.")
                        print(f"Detected tag {tag['id']} at ({x}, {y}) on {wall}")
                        unknown_tags_location.append((tag['id'], x, y))
                # Move back to center and turn to next wall
                
                if wall == 'wall_1':
                    fu.tello_command(tello, ("forward", fu.WALL_1_BACK))
                    fu.align_tag(tello, frame_read, at_detector, camera_params, tag_size, 100, front_distance=1.5)
                    fu.tello_command(tello, ("ccw", 90))  # Turn to face wall 2
                elif wall == 'wall_2':
                    fu.tello_command(tello, ("forward", fu.WALL_2_BACK))
                    fu.tello_command(tello, ("cw", 90))  # Turn to face (0, 0)
                    fu.align_tag(tello, frame_read, at_detector, camera_params, tag_size, 100, front_distance=1.0)  # Align to tag 100, distance 100 cm
                    fu.tello_command(tello, ("cw", 180)) # Turn to face wall 3
                elif wall == 'wall_3':
                    fu.tello_command(tello, ("forward", fu.WALL_3_BACK)) # Move back to center
                    fu.tello_command(tello, ("cw", 180)) # Turn to face wall_1
                    fu.align_tag(tello, frame_read, at_detector, camera_params, tag_size, 100, front_distance=1.5)  # Align to tag 100, distance 150 cm
                    fu.tello_command(tello, ("cw", 90)) # Turn to face wall 4
                elif wall == 'wall_4':
                    fu.tello_command(tello, ("forward", fu.WALL_4_BACK)) # Move back to center
                    fu.tello_command(tello, ("ccw", 90)) # Turn to face wall 1


            else:
                print(f"No tags detected at {wall}, turning to next wall.")
                fu.tello_command(tello, ("ccw", 90))
        
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
        
        


# 若 tello 視窗內沒有辦法把所有已知+未知 tag 照進去，可能要退後或左右移
# tello.move_xxxx 的移動很不準，無法用預定前進多少，可以轉彎後、前進前/後都拍照，

# 在拍攝 wall 2 之前，先對齊 tag 100，回到距離(0,0)正前方 150 cm 的位置，然後轉90度

# wall 3 先轉回正面，對齊 tag 100，回到距離(0,0)正前方 100 cm 的位置，然後轉180度，再退後 50 cm
# wall 4 先轉回正面，對齊 tag 100，回到距離(0,0)正前方 150 cm 的位置，然後轉90度，再退後 30 cm

# 起飛的時候稍微往左放一點
# checked: 修好沒有 known tag 時的定位


if __name__ == '__main__':
    main()