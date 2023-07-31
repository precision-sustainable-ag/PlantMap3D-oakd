import argparse
import time
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
from datetime import date
from PIL import Image
 
today = str(date.today())
dirName = "images_"+today
def dirsetup():
   Path(dirName).mkdir(parents=True, exist_ok=True)

 
## arguments
argn = argparse.ArgumentParser()
argn.add_argument('-n', type=int, default=5)      # number of images to collect
argn.add_argument('-fmode', type=int, default=0)    # focus mode of camera (0 for default setting)
argn.add_argument('-focus', type=int, default=-1)   # focus of color camera
argn.add_argument('-focusf', type=int, default=129)   # far
argn.add_argument('-focusn', type=int, default=145)   # near
argn.add_argument('-subpixel', type=int, default=0)   # Subpixel Disparity
args = argn.parse_args()

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
camRgb.initialControl.setSharpness(0)     # range: 0..4, default: 1    
camRgb.initialControl.setLumaDenoise(0)   # range: 0..4, default: 1    
camRgb.initialControl.setChromaDenoise(4) # range: 0..4, default: 1

script = pipeline.createScript()
camRgb.isp.link(script.inputs['isp'])

script.setScript("""
    while True:
        frame = node.io['isp'].get()
        num = frame.getSequenceNum()
        if (num%30) == 0:
            node.io['frame'].send(frame)
""")

# modifying isp frame and then feeding it to encoder
manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setCropRect(0.006, 0, 1, 1)
manip.setNumFramesPool(2)
manip.setMaxOutputFrameSize(18385920)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
script.outputs['frame'].link(manip.inputImage)

videoEnc = pipeline.create(dai.node.VideoEncoder)
videoEnc.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
manip.out.link(videoEnc.input)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
videoEnc.bitstream.link(xoutRgb.input)


monoRight = pipeline.create(dai.node.MonoCamera)
xoutRight = pipeline.create(dai.node.XLinkOut)
xoutRight.setStreamName("right")
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

monoLeft = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutLeft.setStreamName("left")
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

scriptr = pipeline.createScript()
monoRight.out.link(scriptr.inputs['inr'])
scriptr.setScript("""
    while True:
        frame = node.io['inr'].get()
        num = frame.getSequenceNum()
        if (num%30) == 0:
            node.io['framer'].send(frame)
""")
scriptr.outputs['framer'].link(xoutRight.input)

scriptl = pipeline.createScript()
monoLeft.out.link(scriptl.inputs['inl'])
scriptl.setScript("""
    while True:
        frame = node.io['inl'].get()
        num = frame.getSequenceNum()
        if (num%30) == 0:
            node.io['framel'].send(frame)
""")
scriptl.outputs['framel'].link(xoutLeft.input)

camRgb.setFps(30)
monoRight.setFps(30)
monoLeft.setFps(30)

# depth
Depth = pipeline.create(dai.node.StereoDepth)
Depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
Depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
Depth.setLeftRightCheck(True)
Depth.setExtendedDisparity(False)
if args.subpixel>0: Depth.setSubpixel(True)
else:   Depth.setSubpixel(False)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
monoRight.out.link(Depth.right)
monoLeft.out.link(Depth.left)

script_depth = pipeline.createScript()
Depth.depth.link(script_depth.inputs['ind'])
script_depth.setScript("""
    while True:
        frame = node.io['ind'].get()
        num = frame.getSequenceNum()
        if (num%30) == 0:
            node.io['framed'].send(frame)
""")
script_depth.outputs['framed'].link(xoutDepth.input)

xin1 = pipeline.create(dai.node.XLinkIn)
xin1.setNumFrames(1)   
xin1.setMaxDataSize(1) 
xin1.setStreamName("controlr") 
xin1.out.link(camRgb.inputControl)


def manualFocus(n, f):
    ctrl = dai.CameraControl()
    print("Set focus range", n, f)
    ctrl.setAutoFocusLensRange(n, f)
    qControl1.send(ctrl)
 

def setFocusMode(option):
    if option==0: return
    ctrl = dai.CameraControl()
    if option==1: ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
    else: ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_PICTURE)
    qControl1.send(ctrl)


# Connect to device with pipeline
with dai.Device(pipeline) as device:
    n = args.n      # image count
    dirsetup()

    qRGB = device.getOutputQueue("rgb", maxSize=2, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=2, blocking=False)
    qLeft = device.getOutputQueue(name="left", maxSize=2, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=2, blocking=False)
    qControl1 = device.getInputQueue(name="controlr")

    # focus mode
    setFocusMode(args.fmode)
    # focus
    if(args.focus > 0):
        manualFocus(args.focusf, args.focusn)
    
    for z in range(10):
        c, r, l, d = qRGB.get(), qRight.get(), qLeft.get(), qDepth.get()

    for i in range(n):
        
        t = str(time.time())
        inRgb, inLeft, inRight, inDepth = None, None, None, None

        while inRgb is None:    inRgb = qRGB.tryGet()
        while inRight is None:  inRight = qRight.tryGet()
        while inLeft is None:   inLeft = qLeft.tryGet()
        while inDepth is None:  inDepth = qDepth.tryGet()

        img = cv2.imdecode(inRgb.getData(), cv2.IMREAD_COLOR)
        cv2.imwrite(f"{dirName}/{t}_Rgb.jpg", img)
        cv2.imwrite(f"{dirName}/{t}_Right.png", inRight.getFrame())
        cv2.imwrite(f"{dirName}/{t}_Left.png", inLeft.getFrame())

        im = Image.fromarray(inDepth.getFrame())
        im.save(f"{dirName}/{t}_Depth.png")

