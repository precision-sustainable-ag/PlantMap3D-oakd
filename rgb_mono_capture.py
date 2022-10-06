import sys
import argparse
import time
from pathlib import Path
import cv2
import depthai as dai


dirName = "oak_images"
def dirsetup():
    Path(dirName).mkdir(parents=True, exist_ok=True)


# arguments
argn = argparse.ArgumentParser()
argn.add_argument('-n', type=int, default=6)
argn.add_argument('-iso', type=int, default=1000)
argn.add_argument('-fps', type=int, default=5)
argn.add_argument('-ss', type=int, default=30000)
args = argn.parse_args()


# Create pipeline
pipeline = dai.Pipeline()

# RGB camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
camRgb.setInterleaved(False)
camRgb.initialControl.setSharpness(0)     # range: 0..4, default: 1		
camRgb.initialControl.setLumaDenoise(0)   # range: 0..4, default: 1		
camRgb.initialControl.setChromaDenoise(4) # range: 0..4, default: 1

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.video.link(xoutRgb.input)

xin = pipeline.create(dai.node.XLinkIn)
xin.setStreamName("control")
xin.out.link(camRgb.inputControl)

# Properties
videoEnc = pipeline.create(dai.node.VideoEncoder)
videoEnc.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
camRgb.still.link(videoEnc.input)

# Linking
xoutStill = pipeline.create(dai.node.XLinkOut)
xoutStill.setStreamName("still")
videoEnc.bitstream.link(xoutStill.input)

# Mono cameras
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


def manualExposure(expTimeMs, sensIso):
	expTimeUs = int(round(expTimeMs * 1000))
	ctrl = dai.CameraControl()
	ctrl.setManualExposure(expTimeUs, sensIso)
	qControl.send(ctrl)


def set_fps_and_focus(fps):
	camRgb.setFps(fps)
	videoEnc.setFrameRate(fps)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:	
	i = 0
	# image count
	n = args.n
	dirsetup()

	qRight = device.getOutputQueue(name="right", maxSize=2, blocking=False)
	qLeft = device.getOutputQueue(name="left", maxSize=2, blocking=False)
	qRGB = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
	qStill = device.getOutputQueue(name="still", maxSize=4, blocking=True)
	qControl = device.getInputQueue(name="control")
	# iso and shutter speed
	# manualExposure(args.ss, args.iso)
  # fps
  # set_fps_and_focus(args.fps)

	ctrl = dai.CameraControl()
	ctrl.setCaptureStill(True)

	while(i<n*2):
		inRgb = qRGB.tryGet()
		t = str(int(time.time()))

		if qStill.has():
			fName = f"{dirName}/{t}_Rgb.png"
			with open(fName, "wb") as f:
				f.write(qStill.get().getData())

			inRight = qRight.get()
			cv2.imwrite(f"{dirName}/{t}_Right.png", inRight.getFrame())

			inLeft = qLeft.get()
			cv2.imwrite(f"{dirName}/{t}_Left.png", inLeft.getFrame())

		qControl.send(ctrl)
		i += 1
		time.sleep(1)
