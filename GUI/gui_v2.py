from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import cv2, imutils
import time
import numpy as np
import numpy as np
import cv2
import pyshine as ps
import depthai as dai
import os


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(498, 522)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("images/H.png"))
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        
        #self.label_2 = QtWidgets.QLabel(self.centralwidget)
        #self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        #self.label_2.setObjectName("label_2")
        #self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        #self.label_3 = QtWidgets.QLabel(self.centralwidget)
        #self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        #self.label_3.setObjectName("label_3")
        #self.gridLayout.addWidget(self.label_3, 1, 1, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 2, 2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)
        
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.label_4.setText("Recording")
        self.label_4.setEnabled(False)
        self.gridLayout_2.addWidget(self.label_4, 2,1)
        
        
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.pushButton_2.clicked.connect(self.loadImage)
        self.pushButton.clicked.connect(self.savePhoto)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        

        self.started = False
        self.record_start = False
        self.pushButton.setEnabled(False)
        self.folder_path = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(self.folder_path)

    def videoCapture(self):
        self.pushButton.setEnabled(True)
        pipeline = dai.Pipeline()

        # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
        extended_disparity = False
        # Better accuracy for longer distance, fractional disparity 32-levels:
        subpixel = False
        # Better handling for occlusions:
        lr_check = True

        # Define sources and output
        camRgb = pipeline.create(dai.node.ColorCamera)
        videoEnc = pipeline.create(dai.node.VideoEncoder)
        xout = pipeline.create(dai.node.XLinkOut)
        xoutRgb = pipeline.create(dai.node.XLinkOut)

        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        depth = pipeline.create(dai.node.StereoDepth)
        xoutdepth = pipeline.create(dai.node.XLinkOut)

        videoEnc_depth = pipeline.create(dai.node.VideoEncoder)
        # Depth resolution/FPS will be the same as mono resolution/FPS
        videoEnc_depth.setDefaultProfilePreset(monoLeft.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
        depth.disparity.link(videoEnc_depth.input)
        xoutdepth_enc = pipeline.create(dai.node.XLinkOut)


        xout.setStreamName('h265')
        xoutRgb.setStreamName("rgb")
        xoutdepth.setStreamName("disparity")
        xoutdepth_enc.setStreamName("enc") 

        # Properties
        camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        videoEnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.H265_MAIN)
        camRgb.setPreviewSize(300, 300)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        depth.setLeftRightCheck(lr_check)
        depth.setExtendedDisparity(extended_disparity)
        depth.setSubpixel(subpixel)
        
        streams = ("rgb", "disparity")

        # Linking
        camRgb.video.link(videoEnc.input)
        camRgb.preview.link(xoutRgb.input)
        videoEnc.bitstream.link(xout.input)
        monoLeft.out.link(depth.left)
        monoRight.out.link(depth.right)
        depth.disparity.link(xoutdepth.input)
        videoEnc_depth.bitstream.link(xoutdepth_enc.input)

        self.pipeline = pipeline
        self.width,self.height=(640,480)
    
    def run(self):
        with dai.Device(self.pipeline) as device:
            '''rgbWindowName = "rgb"
            depthWindowName = "depth"
            blendedWindowName = "rgb-depth"
            cv2.namedWindow(rgbWindowName)
            cv2.namedWindow(depthWindowName)'''
            # Output queue will be used to get the encoded data from the output defined above
            q = device.getOutputQueue(name="h265", maxSize=30, blocking=True)
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            qdepth = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
            qdepth_enc = device.getOutputQueue(name="enc")
            qlist = [q,qRgb,qdepth,qdepth_enc]
            with open('video.h265', 'wb') as videoFile, open('disparity.mjpeg', 'wb') as depthFile:
                while True:
                    res = []
                    for q in qlist:
                        name = q.getName()
                        if name == 'h265':
                            h265Packet = q.get()  # Blocking call, will wait until a new data has arrived
                            if self.record_start:
                                h265Packet.getData().tofile(videoFile)  # Appends the packet data to the opened file
                            res.append(h265Packet)
                        elif name == "rgb":
                            inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
                            # Retrieve 'bgr' (opencv format) frame
                            #cv2.imshow("rgb", inRgb.getCvFrame())
                            res.append(inRgb.getCvFrame())
                        elif name == "disparity":
                            inDisparity = qdepth.get()  # blocking call, will wait until a new data has arrived
                            frame = inDisparity.getFrame()
                            # Normalization for better visualization
                            #frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

                            #cv2.imshow("depth", frame)
                            res.append(frame)
                        elif name == "enc":
                            #depthFile.write(q.get().getData())
                            depthPacket = q.get()
                            if self.record_start:
                                depthFile.write(depthPacket.getData())
                            res.append(depthPacket)
                    if cv2.waitKey(1) == ord('q'):
                        break
                    self.last_frame = res[1]
                    yield res

    def loadImage(self):
        """ This function will load the camera device, obtain the image
            and set it to label using the setPhoto function
        """
        if self.started:
            self.started=False
            rgb_path = self.folder_path+'/output_rgb.mp4'
            depth_path = self.folder_path+'/output_depth.mp4'
            os.system("ffmpeg -framerate 30 -i video.h265 -c copy " + rgb_path)
            os.system("ffmpeg -framerate 30 -i disparity.mjpeg -c copy " + depth_path)
            sys.exit( app.exec_() )
            self.pushButton_2.setText('Start')
        else:
            self.started=True
            self.pushButton_2.setText('Stop')
            self.pushButton_2.setEnabled(False)
            
        
        cam = True # True for webcam
        if cam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture('video.mp4')
        
        self.videoCapture()
        
        func = self.run()
        
        while True:
            QtWidgets.QApplication.processEvents()
            self.image = next(func)[1]
            img = self.image
            self.update()
            key = cv2.waitKey(1) & 0xFF
            if self.started==False:
                break
                print('Loop break')

    def setPhoto(self,image):
        """ This function will take image input and resize it 
            only for display purpose and convert it to QImage
            to set at the label.
        """
        self.tmp = image
        image = imutils.resize(image,width=640)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    def update(self):
        """ This function will update the photo according to the 
            current values of blur and brightness and set it to photo label.
        """
        img = self.image
        self.setPhoto(img)

    def savePhoto(self):
        """ This function will save the image"""
        self.record_start = True
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(True)
        self.label_4.setEnabled(True)
        self.label_4.setText('<font color="red">Recording</font>')


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "PyShine video process"))
        self.pushButton_2.setText(_translate("MainWindow", "Begin"))
        self.pushButton.setText(_translate("MainWindow", "Start Recording"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
