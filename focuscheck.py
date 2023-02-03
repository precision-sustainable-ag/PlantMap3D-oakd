import depthai as dai
import cv2
import time
from itertools import cycle

# Step size ('W','A','S','D' controls)
STEP_SIZE = 8
# Manual exposure/focus/white-balance set step
EXP_STEP = 500  # us
ISO_STEP = 50
LENS_STEP = 3
WB_STEP = 200

def clamp(num, v0, v1):
    return max(v0, min(num, v1))

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
#camRgb.setIspScale(2,3) # 1080P -> 720P
stillEncoder = pipeline.create(dai.node.VideoEncoder)

controlIn = pipeline.create(dai.node.XLinkIn)
configIn = pipeline.create(dai.node.XLinkIn)
ispOut = pipeline.create(dai.node.XLinkOut)
videoOut = pipeline.create(dai.node.XLinkOut)
stillMjpegOut = pipeline.create(dai.node.XLinkOut)

controlIn.setStreamName('control')
configIn.setStreamName('config')
ispOut.setStreamName('isp')
videoOut.setStreamName('video')
stillMjpegOut.setStreamName('still')

# Properties
camRgb.setVideoSize(640,360)
stillEncoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)

# Linking
camRgb.isp.link(ispOut.input)
camRgb.still.link(stillEncoder.input)
camRgb.video.link(videoOut.input)
controlIn.out.link(camRgb.inputControl)
configIn.out.link(camRgb.inputConfig)
stillEncoder.bitstream.link(stillMjpegOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Get data queues
    controlQueue = device.getInputQueue('control')
    configQueue = device.getInputQueue('config')
    ispQueue = device.getOutputQueue('isp')
    videoQueue = device.getOutputQueue('video')
    stillQueue = device.getOutputQueue('still')

    # Defaults and limits for manual focus/exposure controls
    lensPos = 150

    awb_mode = cycle([item for name, item in vars(dai.CameraControl.AutoWhiteBalanceMode).items() if name.isupper()])
    anti_banding_mode = cycle([item for name, item in vars(dai.CameraControl.AntiBandingMode).items() if name.isupper()])
    effect_mode = cycle([item for name, item in vars(dai.CameraControl.EffectMode).items() if name.isupper()])

    while True:
        vidFrames = videoQueue.tryGetAll()
        # for vidFrame in vidFrames:
            # cv2.imshow('video', vidFrame.getCvFrame())

        ispFrames = ispQueue.tryGetAll()
        for ispFrame in ispFrames:
            #cv2.imshow('isp', ispFrame.getCvFrame())
            im = cv2.resize(ispFrame.getCvFrame(), (1280,720), interpolation = cv2.INTER_AREA)
            cv2.imshow('isp', im)



        stillFrames = stillQueue.tryGetAll()
        for stillFrame in stillFrames:
            # Decode JPEG
            frame = cv2.imdecode(stillFrame.getData(), cv2.IMREAD_UNCHANGED)
            t = time.time()
            cv2.imwrite(f"{t}_Still_{lensPos}.png", frame)
            # Display
            # cv2.imshow('still', frame)

        # Update screen (1ms pooling rate)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        elif key == ord('c'):
            ctrl = dai.CameraControl()
            ctrl.setCaptureStill(True)
            for i in range(3):
                controlQueue.send(ctrl)

        elif key in [ord(','), ord('.')]:
            if key == ord(','): lensPos -= LENS_STEP
            if key == ord('.'): lensPos += LENS_STEP
            lensPos = clamp(lensPos, 0, 255)
            print("Setting manual focus, lens position: ", lensPos)
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(lensPos)
            controlQueue.send(ctrl)
            
            
# click "," for going down, "." for going up and "c" for clicking a picture
