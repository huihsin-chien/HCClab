import threading
import socket
import time
import cv2
import os  

host = ''
port = 9000
locaddr = (host, port) 

# 建立 UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(locaddr)

# Tello 無人機的 IP 和 Port
tello_address = ('192.168.10.1', 8889)

running = True  # 控制執行狀態

def recv():
    """接收 Tello 回傳的訊息"""
    global running
    while running:
        try:
            data, server = sock.recvfrom(1518)
            print("Tello 回應: {}".format(data.decode("utf-8")))
        except Exception:
            print('\nExit recv thread...\n')
            break

# 啟動接收執行緒
recvThread = threading.Thread(target=recv, daemon=True)
recvThread.start()

# 傳送初始化指令
sock.sendto("command".encode("utf-8"), tello_address)
time.sleep(3)
sock.sendto("streamon".encode("utf-8"), tello_address)

# 檢查並建立影片存放資料夾
save_dir = "./video"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 設定影片存檔路徑
video_filename = os.path.join(save_dir, "tello_video.avi")

# 設定影像串流
cap = cv2.VideoCapture("udp://192.168.10.1:11111")

# 檢查影像解析度（確保正確）
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # 設定影片幀率，Tello 預設為 30 FPS

if width == 0 or height == 0:
    width, height = 960, 720  # Tello 預設解析度

# 設定影片編碼格式
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # AVI 格式
video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

# 另一個執行緒處理終端機輸入
def user_input():
    """讓使用者輸入指令並發送到 Tello"""
    global running
    while running:
        cmd = input("請輸入 Tello 指令 (輸入 'exit' 結束): ").strip()
        if cmd.lower() == "exit":
            running = False
            break
        sock.sendto(cmd.encode("utf-8"), tello_address)
        time.sleep(5)

# 啟動用戶輸入執行緒
inputThread = threading.Thread(target=user_input, daemon=True)
inputThread.start()

# 影像處理迴圈（錄影）
while running:
    isFrame, frame = cap.read()
    if isFrame:
        cv2.imshow("UAV video", frame)
        video_writer.write(frame)  # 直接寫入影片檔案

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

# 釋放資源
cap.release()
video_writer.release()
cv2.destroyAllWindows()
sock.close()
print(f"影片已儲存至 {video_filename}")
print("程式已結束。")
