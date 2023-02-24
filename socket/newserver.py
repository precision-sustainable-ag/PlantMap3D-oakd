import depthai as dai
import numpy as np
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
	"data":{
		"length":12257280,
		"width":4032,
		"height":3040,
		"pixels":[]
	},
	"error": None
}

img_dict_l = {
	"key":"Image",
	"type": "MonoLeft",
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
	"data":{
		"length":921600,
		"width":1280,
		"height":720,
		"pixels":[]
	},
	"error": None
}

img_dict_d = {
	"key":"Image",
	"type": "Depth",
	"state": "Progress",
	"data":{
		"length":921600,
		"width":1280,
		"height":720,
		"pixels":[]
	},
	"error": None
}


# OAK-D pipeline
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)

# RGB
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.isp.link(xoutRgb.input)

xoutRgbp = pipeline.createXLinkOut()
xoutRgbp.setStreamName("preview")
camRgb.preview.link(xoutRgbp.input)

# monoRight
monoRight = pipeline.create(dai.node.MonoCamera)
xoutRight = pipeline.create(dai.node.XLinkOut)
xoutRight.setStreamName("right")
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.out.link(xoutRight.input)

# monoLeft
monoLeft = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutLeft.setStreamName("left")
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.out.link(xoutLeft.input)

# depth
Depth = pipeline.create(dai.node.StereoDepth)
Depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
Depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
Depth.setLeftRightCheck(True)
Depth.setExtendedDisparity(False)
Depth.setSubpixel(False)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("disparity")
monoRight.out.link(Depth.right)
monoLeft.out.link(Depth.left)
Depth.disparity.link(xoutDepth.input)



for p in range(1):
    client_socket, addr = server_socket.accept()
    print('Got Connection from:', addr)

    if client_socket:
        while True:
            try:
                cmd = client_socket.recv(1024)
                if not cmd:
                    break
            except:
                print("Connection dropped by client")
                break
            
            try:
                d_rcv = json.loads(cmd)
            except Exception:
                print("Invalid message")
                break

            c_key = d_rcv.get("key")

            if(c_key == "Start"):
                print("\nStarting camera")
            
                with dai.Device(pipeline) as device:
                    queue = device.getOutputQueue(name="preview", maxSize=2, blocking=False)
                    qRGB = device.getOutputQueue(name="rgb", maxSize=2, blocking=False)
                    qRight = device.getOutputQueue(name="right", maxSize=2, blocking=False)
                    qLeft = device.getOutputQueue(name="left", maxSize=2, blocking=False)
                    qDepth = device.getOutputQueue(name="disparity", maxSize=2, blocking=False)
                    init_dsp = (255 / Depth.initialConfig.getMaxDisparity())
            
                    while True:
                        try:
                            cmd = client_socket.recv(1024)
                            d_rcv = json.loads(cmd)
                        except:
                            err_dict["error"]["message"] = "Error with message received"
                            j_e = json.dumps(err_dict).encode()
                            client_socket.sendall(j_e)
                            break

                        c_key = d_rcv.get("key")

                        # send preview
                        if(c_key == "Preview"):
                            try:
                                imOut = queue.get().getCvFrame()
                                prv_dict["data"]["pixels"] = imOut.tolist()
                                j_p = json.dumps(prv_dict)
                                message = j_p.encode()
                                ln = json.dumps(len(message)).encode()
                                client_socket.sendall(ln)
                                client_socket.sendall(message)
                            except:
                                err_dict["error"]["message"] = "Issue with camera preview/encoding"
                                j_e = json.dumps(err_dict).encode()
                                try:
                                    client_socket.sendall(j_e)
                                except:
                                    print("Client connection dropped")
                                    break
                                continue

                        # send image set
                        if(c_key == "TakeSnapshot"):
                            try:
                                inRgb = qRGB.get().getCvFrame()
                                inRight = qRight.get().getFrame()
                                inLeft = qLeft.get().getFrame()
                                inDepth = qDepth.get().getFrame()
                                dframe = (inDepth * init_dsp).astype(np.uint8)

                                img_dict_c["data"]["pixels"] = inRgb.tolist()
                                img_dict_l["data"]["pixels"] = inLeft.tolist()
                                img_dict_r["data"]["pixels"] = inRight.tolist()
                                img_dict_d["data"]["pixels"] = dframe.tolist()

                                lst = [img_dict_c, img_dict_d, img_dict_l, img_dict_r]
                                for msg in lst:
                                    j_p = json.dumps(msg)
                                    message = j_p.encode()
                                    print(len(message))
                                    ln = json.dumps(len(message)).encode()
                                    client_socket.sendall(ln)
                                    client_socket.sendall(message)
                            except:
                                err_dict["error"]["message"] = "Issue with sending image set"
                                j_e = json.dumps(err_dict).encode()
                                try:
                                    client_socket.sendall(j_e)
                                except:
                                    print("Client connection dropped")
                                    break
                                continue
                    
                        if(c_key == "StopPreview"):
                            print("Stoping Camera")
                            break

            if(c_key == "Close"):
                print("Closing Connection")
                client_socket.close()
                time.sleep(1)
                print("Connection closed\n")
                break
