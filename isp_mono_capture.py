import argparse
import time
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
from datetime import date
 
today = str(date.today())
dirName = "images_"+today
def dirsetup():
   Path(dirName).mkdir(parents=True, exist_ok=True)
 
 
## arguments
argn = argparse.ArgumentParser()
argn.add_argument('-n', type=int, default=5)      # number of images to collect
argn.add_argument('-iso', type=int, default=400)   # iso of color camera
argn.add_argument('-fps', type=float, default=1)    # fps of all cameras
argn.add_argument('-ss', type=float, default=1)   # shutter speed of color camera in milliseconds
argn.add_argument('-miso', type=int, default=150)   # iso of mono camera
argn.add_argument('-focus', type=int, default=-1)   # focus of color camera (0 for far to 255 for near)
argn.add_argument('-fmode', type=int, default=0)
args = argn.parse_args()
mono_ss = 1000
 
## Create pipeline
pipeline = dai.Pipeline()
 
# color camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
camRgb.initialControl.setSharpness(0)     # range: 0..4, default: 1    
camRgb.initialControl.setLumaDenoise(0)   # range: 0..4, default: 1    
camRgb.initialControl.setChromaDenoise(4) # range: 0..4, default: 1
camRgb.setFps(args.fps)
 
# Mono cameras
monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setFps(args.fps)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setFps(args.fps)
 
# depth
Depth = pipeline.create(dai.node.StereoDepth)
Depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
Depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
Depth.setLeftRightCheck(True)
Depth.setExtendedDisparity(False)
Depth.setSubpixel(False)
Depth.setNumFramesPool(1)
 
## Linking
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.isp.link(xoutRgb.input)
 
xoutRight = pipeline.create(dai.node.XLinkOut)
xoutRight.setStreamName("right")
monoRight.out.link(xoutRight.input)
 
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutLeft.setStreamName("left")
monoLeft.out.link(xoutLeft.input)
 
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("disparity")
monoRight.out.link(Depth.right)
monoLeft.out.link(Depth.left)
Depth.disparity.link(xoutDepth.input)
 
xin1 = pipeline.create(dai.node.XLinkIn)
xin1.setNumFrames(1)   
xin1.setMaxDataSize(1) 
xin1.setStreamName("controlr") 
xin1.out.link(camRgb.inputControl) 
 
xin2 = pipeline.create(dai.node.XLinkIn)   
xin2.setNumFrames(1)
xin2.setMaxDataSize(1) 
xin2.setStreamName("controlm") 
xin2.out.link(monoLeft.inputControl)   
xin2.out.link(monoRight.inputControl)
 
 
def set_ss_iso(expTimeMs, sensIso):
   expTimeUs = int(round(expTimeMs * 1000))   
   ctr1 = dai.CameraControl() 
   ctr2 = dai.CameraControl() 
   ctr1.setManualExposure(expTimeUs, sensIso) 
   qControl1.send(ctr1)   
   ctr2.setManualExposure(mono_ss, args.miso) 
   qControl2.send(ctr2)
 
 
def manualFocus(focus):
   if focus < 0: return
   assert(focus >= 0 and focus <= 255)
   ctrl = dai.CameraControl()
   ctrl.setManualFocus(focus)
   qControl1.send(ctrl)
 
def setFocusMode(option):
   if option==0: return
   ctrl = dai.CameraControl()
   if option==1: ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
   else: ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_PICTURE)
   qControl1.send(ctrl)
 
 
# Connect to device and start pipeline
with dai.Device(pipeline) as device:   
   n = args.n      # image count
   dirsetup()
   init_dsp = (255 / Depth.initialConfig.getMaxDisparity())
 
   qRight = device.getOutputQueue(name="right", maxSize=5, blocking=False)
   qLeft = device.getOutputQueue(name="left", maxSize=5, blocking=False)
   qRGB = device.getOutputQueue(name="rgb", maxSize=5, blocking=False)
   qDepth = device.getOutputQueue(name="disparity", maxSize=5, blocking=False)
   qControl1 = device.getInputQueue(name="controlr")  
   qControl2 = device.getInputQueue(name="controlm")
              
   # iso and shutter speed
   set_ss_iso(args.ss, args.iso)
   # focus mode
   setFocusMode(args.fmode)
   # focus
   manualFocus(args.focus)
 
   for r in range(30):
       c, r, l, d = qRGB.get(), qRight.get(), qLeft.get(), qDepth.get()
 
   print("Started")
   start = time.time()
 
   for i in range(n):
       t = str(time.time())
 
       inRgb = qRGB.tryGet()
       inRight = qRight.tryGet()
       inLeft = qLeft.tryGet()
       inDepth = qDepth.tryGet()
       if inRgb is not None:
           cv2.imwrite(f"{dirName}/{t}_Rgb.png", inRgb.getCvFrame())
       if inRight is not None:
           cv2.imwrite(f"{dirName}/{t}_Right.png", inRight.getFrame())
       if inLeft is not None:
           cv2.imwrite(f"{dirName}/{t}_Left.png", inLeft.getFrame())
       if inDepth is not None:
           dframe = inDepth.getFrame()
           dframe = (dframe * init_dsp).astype(np.uint8)
           cv2.imwrite(f"{dirName}/{t}_Depth.png", dframe)
      
print("Finished in", round(time.time()-start, 2))
