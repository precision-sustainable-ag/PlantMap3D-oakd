import argparse
import time
import math
from pathlib import Path
import cv2
import depthai as dai
from csv import writer
import numpy as np
from datetime import date
import queue
 
today = str(date.today())
dirName = "images_"+today
def dirsetup():
   Path(dirName).mkdir(parents=True, exist_ok=True)

 
## arguments
argn = argparse.ArgumentParser()
argn.add_argument('-n', type=int, default=5)      # number of images to collect
argn.add_argument('-fmode', type=int, default=0)    # focus mode of camera (0 for default setting)
argn.add_argument('-focus', type=int, default=-1)   # focus of color camera (0 for far to 255 for near)
argn.add_argument('-focusf', type=int, default=129)   # far
argn.add_argument('-focusn', type=int, default=145)   # near
argn.add_argument('-top', type=int, default=128)
argn.add_argument('-bottom', type=int, default=104)
args = argn.parse_args()

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
# camRgb.setNumFramesPool(3,3,0,0,1)
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
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)

monoLeft = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutLeft.setStreamName("left")
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)

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
Depth.setSubpixel(False)
Depth.setNumFramesPool(2)

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


nISO = 19
ISO = np.logspace(1.25, 2, num=nISO+1, endpoint=True, base=40, dtype=int)
SS = [313, 400, 500, 625, 800, 1000]
T = 15
B = 12
dT = 1.99    #1.99
nQ = 4


def setisoss(fi, fs, col):
    ctr1 = dai.CameraControl()
    ctr1.setManualExposure(SS[fs], ISO[fi])
    if col:
        qControl1.send(ctr1)
    else:
        qControl2.send(ctr1)

def add_frame(img, col):
    if col:
        im = cv2.resize(img, (406,304), interpolation = cv2.INTER_AREA)
        Y = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([Y],[0],None,[32],[0,256])
        if( cor_grp.qsize()>=nQ ): cor_grp.get(0)
        cor_grp.put(hist)
    else:
        Y = cv2.resize(img, (320,180), interpolation = cv2.INTER_AREA)
        hist = cv2.calcHist([Y],[0],None,[32],[0,256])
        if( mono_grp.qsize()>=nQ ): mono_grp.get(0)
        mono_grp.put(hist)

def adjust_exposure(i, s, col):
    frm = []
    if col:
        for p in range(cor_grp.qsize()):    frm.append(cor_grp.queue[p])
    else:
        for p in range(mono_grp.qsize()):    frm.append(mono_grp.queue[p])
    arr = np.array(frm)
    
    summ = arr.sum(axis=0)
    peak = np.max(summ)
    pos = np.where(summ == peak)[0][0]

    if pos not in range(B, T): # 104 to 128 range by default
        # increase iso or ss
        if( pos < B):
            if s == len(SS)-1:
                i = min(i+1, nISO)
            else:
                s = min(s+1, 5)
        else:
            if i == 0:
                s = max(0, s-1)
            else:
                i = max(0, i-1)
    if col: setisoss(i, s, True)
    else:   setisoss(i, s, False)
    return i, s


# Connect to device with pipeline
with dai.Device(pipeline) as device:
    n = args.n      # image count
    dirsetup()

    qRGB = device.getOutputQueue("rgb", maxSize=2, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=2, blocking=False)
    qLeft = device.getOutputQueue(name="left", maxSize=2, blocking=False)
    qDepth = device.getOutputQueue(name="disparity", maxSize=2, blocking=False)
    qControl1 = device.getInputQueue(name="controlr")  
    qControl2 = device.getInputQueue(name="controlm")

    init_dsp = (255 / Depth.initialConfig.getMaxDisparity())
    iso, ss = 6, 5
    miso, mss = 0, 5
    cor_grp = queue.Queue(nQ)
    mono_grp = queue.Queue(nQ)
    diffs = np.array([])

    # focus mode
    setFocusMode(args.fmode)
    # focus
    if(args.focus > 0):
        manualFocus(args.focusf, args.focusn)

    stamp = time.time()
    start = time.time()
    
    for z in range(20):
        c, r, l, d = qRGB.get(), qRight.get(), qLeft.get(), qDepth.get()
    for z in range(10):
        c, r, l, d = qRGB.get(), qRight.get(), qLeft.get(), qDepth.get()
        img = cv2.imdecode(c.getData(), cv2.IMREAD_COLOR)
        add_frame(img, True)
        add_frame(r.getFrame(), False)
        iso, ss = adjust_exposure(iso, ss, True)
        miso, mss = adjust_exposure(miso, mss, False)


    #print(c.getSequenceNum())

    with open('data.csv', 'a', newline='') as f_object:
        writer_object = writer(f_object)
        for i in range(n):
            
            t = str(time.time())
            inRgb, inLeft, inRight, inDepth = None, None, None, None

            while inRgb is None:    inRgb = qRGB.tryGet()
            while inRight is None:  inRight = qRight.tryGet()
            while inLeft is None:   inLeft = qLeft.tryGet()
            while inDepth is None:  inDepth = qDepth.tryGet()

            img = cv2.imdecode(inRgb.getData(), cv2.IMREAD_COLOR)
            cv2.imwrite(f"{dirName}/{t}_Rgb_{ISO[iso]}.jpg", img)
            cv2.imwrite(f"{dirName}/{t}_Right_{ISO[miso]}.png", inRight.getFrame())
            cv2.imwrite(f"{dirName}/{t}_Left.png", inLeft.getFrame())
            dframe = inDepth.getFrame()
            dframe = (dframe * init_dsp).astype(np.uint8)
            cv2.imwrite(f"{dirName}/{t}_Depth.png", dframe)

            add_frame(img, True)
            add_frame(inRight.getFrame(), False)

            if( time.time()-stamp > dT ):
                iso, ss = adjust_exposure(iso, ss, True)
                miso, mss = adjust_exposure(miso, mss, False)
                stamp = time.time()
                writer_object.writerow( [ISO[iso], SS[ss], ISO[miso], SS[mss]] )
        
        f_object.close()
