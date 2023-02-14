import cv2
import depthai as dai
import numpy as np
import sys
import json
import socket
import time

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_name = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
port = 6666
socket_address = (host_ip, port)
server_socket.bind(socket_address)
server_socket.listen(5)
print("Listening at:", socket_address)

client_socket, addr = server_socket.accept()
print('Got Connection from:', addr)

err_dict = {
	"key":"Error",
	"data":None,
	"error": {
		"message": "",
		"cause": ""
	}
}

img_dict = {
	"key":"Image",
	"type": "",
	"state": "Progress",
	"data":{
		"length":270000,
		"width":300,
		"height":300,
		"pixels":[]
	},
	"error":None
}

pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)

xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)


if client_socket:
    with dai.Device(pipeline) as device:
        queue = device.getOutputQueue(name="rgb", maxSize=2, blocking=False)

        while True:
            cmd = client_socket.recv(1024)
            if not cmd:
                break
            try:
                d_rcv = json.loads(cmd)
            except Exception:
                print("Receive Error")

            c_key = d_rcv.get("key")

            if(c_key == "Start"):
                print("\nStarting")

            if(c_key == "Preview"):
                img_dict["type"] = "Preview"
                try:
                    print("\nSending data")
                    for i in range(30):
                        frame = queue.get()
                        imOut = frame.getCvFrame()
                        img_dict["data"]["pixels"] = imOut.tolist()
                        j_p = json.dumps(img_dict)
                        message = j_p.encode()
                        print(len(message))
                        client_socket.sendall(message)
                except:
                    err_dict["error"]["message"] = "Something wrong with camera"
                    j_e = json.dumps(err_dict).encode()
                    client_socket.sendall(j_e)
                    print('Error!!')
                    break
                    
            if(c_key == "Stop"):
                print("\nStoping")
                break

client_socket.close()
print("Server Closed")
