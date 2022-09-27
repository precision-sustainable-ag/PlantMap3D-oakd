from pathlib import Path
import cv2
import depthai as dai
import time
import numpy as np


colordirName = "color_data"
dirName = "mono_data"
def dirsetup():
    Path(dirName).mkdir(parents=True, exist_ok=True)
    Path(colordirName).mkdir(parents=True, exist_ok=True)

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output links, streams
monoRight = pipeline.create(dai.node.MonoCamera)
xoutRight = pipeline.create(dai.node.XLinkOut)

monoLeft = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)

camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)

Depth = pipeline.create(dai.node.StereoDepth)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRight.setStreamName("right")
xoutLeft.setStreamName("left")
xoutRgb.setStreamName("rgb")
xoutDepth.setStreamName("disparity")

# Properties
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRgb.setPreviewSize(3840, 2160)
camRgb.setInterleaved(False)
camRgb.initialControl.setSharpness(0)     # range: 0..4, default: 1
camRgb.initialControl.setLumaDenoise(0)   # range: 0..4, default: 1
camRgb.initialControl.setChromaDenoise(4) # range: 0..4, default: 1

Depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
Depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
Depth.setLeftRightCheck(True)
Depth.setExtendedDisparity(False)
Depth.setSubpixel(False)

# Linking
monoRight.out.link(xoutRight.input)
monoLeft.out.link(xoutLeft.input)
camRgb.preview.link(xoutRgb.input)

monoRight.out.link(Depth.right)
monoLeft.out.link(Depth.left)
Depth.disparity.link(xoutDepth.input)

# Device setup
device = dai.Device(pipeline)

qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)
qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
qRGB = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
qDepth = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)


def flushframes(n):
  for i in range(n):
    R, L, C, D = qRight.get(), qLeft.get(), qRGB.get(), qDepth.get()


def captureImage():
  t = str(int(time.time()))
  
  inRight = qRight.get()
  inLeft = qLeft.get()
  inRgb = qRGB.get()
  inDepth = qDepth.get()
  dframe = inDepth.getFrame()
  dframe = (dframe * (255 / Depth.initialConfig.getMaxDisparity())).astype(np.uint8)
  
  cv2.imwrite(f"{dirName}/Right_{t}.jpeg", inRight.getFrame())
  cv2.imwrite(f"{dirName}/Left_{t}.jpeg", inLeft.getFrame())
  cv2.imwrite(f"{dirName}/Depth_{t}.png", dframe)
  cv2.imwrite(f"{colordirName}/Rgb_{t}.jpeg", inRgb.getCvFrame())



# Capture Images
dirsetup()
flushframes(13)


for i in range(2):
  print('Clicking picture....')
  time.sleep(1)
  captureImage()
