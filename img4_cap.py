import sys
import argparse
import time
from pathlib import Path
import cv2
import depthai as dai
import numpy as np

dirName = "oak_images"
def dirsetup():
    Path(dirName).mkdir(parents=True, exist_ok=True)


# arguments
argn = argparse.ArgumentParser()
argn.add_argument('-n', type=int, default=100)
argn.add_argument('-iso', type=int, default=1000)
argn.add_argument('-fps', type=float, default=2)
argn.add_argument('-ss', type=float, default=0.5)
args = argn.parse_args()


# Create pipeline
pipeline = dai.Pipeline()
# total available RAM is 358 MiB Mebibyte (MiB = 1.048576 MB)

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
camRgb.setInterleaved(False)
camRgb.initialControl.setSharpness(0)     # range: 0..4, default: 1		
camRgb.initialControl.setLumaDenoise(0)   # range: 0..4, default: 1		
camRgb.initialControl.setChromaDenoise(4) # range: 0..4, default: 1

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
xoutRgb.input.setBlocking(False)
xoutRgb.input.setQueueSize(1)
camRgb.isp.link(xoutRgb.input)
# xoutRgb.setFpsLimit(5)


# control pipeline
xin = pipeline.create(dai.node.XLinkIn)
xin.setMaxDataSize(1)
xin.setStreamName("control")
xin.out.link(camRgb.inputControl)


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

# depth
Depth = pipeline.create(dai.node.StereoDepth)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("disparity")
Depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
Depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
Depth.setLeftRightCheck(True)
Depth.setExtendedDisparity(False)
Depth.setSubpixel(False)
monoRight.out.link(Depth.right)
monoLeft.out.link(Depth.left)
Depth.disparity.link(xoutDepth.input)


def manualExposure(expTimeMs, sensIso):
    expTimeUs = int(round(expTimeMs * 1000))
    ctrl = dai.CameraControl()
    ctrl.setManualExposure(expTimeUs, sensIso)
    qControl.send(ctrl)


def manualFocus(focus):
    ctrl = dai.CameraControl()
    ctrl.setManualFocus(focus)
    qControl.send(ctrl)


def set_fps_and_focus(fps):
    camRgb.setFps(fps)
    monoRight.setFps(fps)
    monoLeft.setFps(fps)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:	
    i = 0
    # image count
    n = args.n
    dirsetup()

    qRight = device.getOutputQueue(name="right", maxSize=2, blocking=False)
    qLeft = device.getOutputQueue(name="left", maxSize=2, blocking=False)
    qRGB = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    qDepth = device.getOutputQueue(name="disparity", maxSize=2, blocking=False)
    qControl = device.getInputQueue(name="control")


    # iso and shutter speed
    # manualExposure(args.ss, args.iso)
    # fps
    set_fps_and_focus(args.fps)

    for r in range(30):
        inRgb = qRGB.get()
        inRight = qRight.get()
        inLeft = qLeft.get()
        inDepth = qDepth.get()

    print("Started")
    start = time.time()
    last = start

    while(i<n):
        inRgb = qRGB.get()
        inRight = qRight.get()
        inLeft = qLeft.get()
        inDepth = qDepth.get()
        cv2.imwrite(f"{dirName}/{i+1}_Rgb.png", inRgb.getCvFrame())
        cv2.imwrite(f"{dirName}/{i+1}_Right.png", inRight.getFrame())
        cv2.imwrite(f"{dirName}/{i+1}_Left.png", inLeft.getFrame())
        dframe = inDepth.getFrame()
        dframe = (dframe * (255 / Depth.initialConfig.getMaxDisparity())).astype(np.uint8)
        cv2.imwrite(f"{dirName}/{i+1}_Depth.png", dframe)

        i += 1


print("Finished in", round(time.time()-start, 2))
