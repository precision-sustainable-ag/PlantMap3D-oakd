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


err_dict = {
	"key":"Error",
	"data":None,
	"error": {
		"message": "",
		"cause": ""
	}
}

prv_dict = {
	"key":"Image",
	"type": "Preview",
	"state": "Progress",
	"data":{
		"length":270000,
		"width":300,
		"height":300,
		"pixels":[]
	},
	"error": None
}

img_dict_c = {
	"key":"Image",
	"type": "RGB",
	"state": "Progress",
	"data":{
		"length":270000,
		"width":300,
		"height":300,
		"pixels":[]
	},
	"error": None
}

img_dict_l = {
	"key":"Image",
	"type": "MonoLeft",
	"state": "Progress",
	"data":{
		"length":921600,
		"width":1280,
		"height":720,
		"pixels":[]
	},
	"error": None
}

img_dict_r = {
	"key":"Image",
	"type": "MonoRight",
	"state": "Progress",
	"data":{
		"length":921600,
		"width":1280,
		"height":720,
		"pixels":[]
	},
	"error": None
}

pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)

xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

monoRight = pipeline.create(dai.node.MonoCamera)
xoutRight = pipeline.create(dai.node.XLinkOut)
xoutRight.setStreamName("right")
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.out.link(xoutRight.input)

monoLeft = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutLeft.setStreamName("left")
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.out.link(xoutLeft.input)

for p in range(1):
    client_socket, addr = server_socket.accept()
    print('Got Connection from:', addr)

    if client_socket:
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
                print("\nStarting camera")
            
                with dai.Device(pipeline) as device:
                    queue = device.getOutputQueue(name="rgb", maxSize=2, blocking=False)
                    qRight = device.getOutputQueue(name="right", maxSize=2, blocking=False)
                    qLeft = device.getOutputQueue(name="left", maxSize=2, blocking=False)
            
                    while True:
                        cmd = client_socket.recv(1024)
                        try:
                            d_rcv = json.loads(cmd)
                        except Exception:
                            break

                        c_key = d_rcv.get("key")

                        # send preview
                        if(c_key == "Preview"):
                            try:
                                imOut = queue.get().getCvFrame()
                                prv_dict["data"]["pixels"] = imOut.tolist()
                                j_p = json.dumps(prv_dict)
                                message = j_p.encode()
                                #print(len(message))
                                ln = json.dumps(len(message)).encode()
                                client_socket.sendall(ln)
                                time.sleep(0.05)
                                client_socket.sendall(message)
                            except:
                                err_dict["error"]["message"] = "Couldn't fetch preview frame"
                                j_e = json.dumps(err_dict).encode()
                                client_socket.sendall(j_e)
                                print('Error!!')
                                break

                        if(c_key == "TakeSnapshot"):
                            try:
                                inRgb = queue.get().getCvFrame()
                                inRight = qRight.get().getFrame()
                                inLeft = qLeft.get().getFrame()

                                img_dict_c["data"]["pixels"] = inRgb.tolist()
                                img_dict_l["data"]["pixels"] = inLeft.tolist()
                                img_dict_r["data"]["pixels"] = inRight.tolist()

                                j_p = json.dumps(img_dict_c)
                                message = j_p.encode()
                                print(len(message))
                                ln = json.dumps(len(message)).encode()
                                client_socket.sendall(ln)
                                client_socket.sendall(message)

                                j_p = json.dumps(img_dict_l)
                                message = j_p.encode()
                                print(len(message))
                                ln = json.dumps(len(message)).encode()
                                client_socket.sendall(ln)
                                client_socket.sendall(message)

                                j_p = json.dumps(img_dict_r)
                                message = j_p.encode()
                                print(len(message))
                                ln = json.dumps(len(message)).encode()
                                client_socket.sendall(ln)
                                client_socket.sendall(message)
                            except:
                                err_dict["error"]["message"] = "Couldn't fetch frames"
                                j_e = json.dumps(err_dict).encode()
                                client_socket.sendall(j_e)
                                break
                    
                        if(c_key == "StopPreview"):
                            print("Stoping Camera")
                            break

            if(c_key == "Close"):
                client_socket.close()
                print("Connection closed\n")
                break
