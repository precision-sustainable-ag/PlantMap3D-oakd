import cv2
import depthai as dai
import sys

import socket, pickle, struct

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_name = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
print('HOST IP:', host_ip)
port = 6666
socket_address = (host_ip, port)
server_socket.bind(socket_address)
server_socket.listen(5)
print("Listening at:", socket_address)

pipeline = dai.Pipeline()
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRgb.setPreviewSize(640, 480)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

with dai.Device(pipeline) as device:
    print('Connected')
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qt = False
    while True:
        client_socket, addr = server_socket.accept()
        print('Got Connection from:', addr)

        if client_socket:
            while True:
                inRgb = qRgb.get()
                frame = inRgb.getCvFrame()
                a = pickle.dumps(frame)
                message = struct.pack("Q", len(a))+a
                client_socket.sendall(message)
                cv2.imshow('Tranmitting video', frame)
                if cv2.waitKey(1) == ord('q'):
                    client_socket.close()
                    qt = True
                    break
        if qt:
            break

print("Program Ended")

