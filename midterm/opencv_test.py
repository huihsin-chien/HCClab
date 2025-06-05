import threading 
import socket
import sys
import time
import cv2


host = ''
port = 9000
locaddr = (host,port) 


# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(locaddr)

#please fill UAV IP address
tello_address1 = ('192.168.10.1', 8889)
#tello_address2 = ('???', 8889)

message1=["command", "takeoff",  "forward 150", "up 50", "forward 150", "land"]
#message2=["command", "takeoff",  "forward 200", "land"]
delay=[2,3,5,4,5,1]

def recv():
    count = 0
    while True: 
        try:
            data, server = sock.recvfrom(1518)
            print("{} : {}".format(server,data.decode(encoding="utf-8")))
        except Exception:
            print ('\nExit . . .\n')
            break


print ('\r\n\r\nTello Python3 Demo.\r\n')

print ('Tello: command takeoff land flip forward back left right \r\n       up down cw ccw speed speed?\r\n')

print ('end -- quit demo.\r\n')


#recvThread create
recvThread = threading.Thread(target=recv)
recvThread.start()


for i in range(0,len(message1)):
    msg1=message1[i]
    #msg2=message2[i]
    sock.sendto(msg1.encode("utf-8"), tello_address1)
    #sock.sendto(msg2.encode("utf-8"), tello_address2)
    time.sleep(delay[i])


cap=cv2.VideoCapture("udp://192.168.10.1:11111")

while True:
	isFrame, frame=cap.read()

	if isFrame:
		cv2.imshow("UAV video",frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
