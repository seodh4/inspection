#------------------pyqt5 import start-------------------
from ossaudiodev import SNDCTL_DSP_GETBLKSIZE
import re
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui
from PyQt5.QtCore import *
from PyQt5 import *
from PyQt5.QtGui import *
#------------------pyqt5 import end---------------------

#------------------pylon import start-------------------
from tkinter import CENTER
import pypylon.pylon as pylon
#------------------pylon import end---------------------

#------------------common import start-------------------
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import pandas as pd
from collections import Counter
import copy
#------------------common import end-------------------

from canvas import Canvas
import json
from TrackingAPI8_0915 import *   


# sys.path.append('/home/ai/workspace/x86/maccel/')
# sys.path.append('.')
# import maccel
import torch
from utils.yolov5_post import post_processing_yolov5

# ID2LABEL = [
#   "AHS",
#   "BTN",
#   "CAL",
#   "CAP",
#   "CAS",
#   "CCA",
#   "CCL",
#   "CCS",
#   "CDS",
#   "CEL",
#   "CES",
#   "CPL",
#   "CPP",
#   "CPS",
#   "CTT",
#   "DDL",
#   "DDS",
#   "DTB",
#   "FES",
#   "FUS",
#   "ICD",
#   "ICT",
#   "LCL",
#   "LCS",
#   "LED",
#   "LLS",
#   "LNL",
#   "LNS",
#   "LPS",
#   "REL",
#   "RES",
#   "RLS",
#   "RPS",
#   "SWC",
#   "TAA",
#   "TAB",
#   "TAC",
#   "TAD",
#   "TAF",
#   "TAG",
#   "TAH",
#   "TAJ",
#   "TAK",
#   "TAL",
#   "TAM",
#   "TAN",
#   "TAQ",
#   "TAR",
#   "TAS",
#   "TAT",
#   "TAU",
#   "TAV",
#   "TAW",
#   "TAX",
#   "TAY",
#   "TAZ",
#   "TBA",
#   "TBB",
#   "TBC",
#   "TBD",
#   "TBE",
#   "TRL",
#   "TRS",
#   "cpp"]



class Webcam(QThread):
    
    sendimg_webcam = pyqtSignal(np.ndarray, int, int)
    # sendimg_noinfer = pyqtSignal(np.ndarray)

    def __init__(self, parent): 
        super().__init__(parent) 

        self.parent = parent

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    def run(self):

        while cv2.waitKey(33) < 0:
            ret, img = self.cap.read()
            # cv2.imshow("VideoFrame", frame)
            self.sendimg_webcam.emit(img, 1, 0)

        self.cap.release()





# definition of event handler class 
class TriggeredImage(pylon.ImageEventHandler):

    grap = pyqtSignal(np.ndarray, int)

    def __init__(self, parent):
        super().__init__() 
        self.parent = parent

        # self.grab_times = []
    def OnImageGrabbed(self, camera, grabResult):
        # self.grab_times.append(grabResult.TimeStamp)
        # print(grabResult.TimeStamp)
        # print("SizeX: ", grabResult.GetWidth())
        # print("SizeY: ", grabResult.GetHeight())

        # converter = pylon.ImageFormatConverter()

        # # converting to opencv bgr format
        # converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        # converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # image = converter.Convert(grabResult)
        # img = image.GetArray()

        img = grabResult.GetArray()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # open_cv_image = np.array(img) 
        # cv2.imwrite('asd.jpg',img)
        self.parent.grap(img, grabResult.TimeStamp)
        
        grabResult.Release()




 
 
class Pylon(QThread):
    
    sendimg = pyqtSignal(np.ndarray, int, int)
    # sendimg_noinfer = pyqtSignal(np.ndarray)

    def __init__(self, parent): 
        super().__init__(parent) 

        self.parent = parent

        # open the camera
        tlf = pylon.TlFactory.GetInstance()
        self.cam = pylon.InstantCamera(tlf.CreateFirstDevice())
        self.cam.Open()

        # get clean powerup state
        self.cam.UserSetSelector = "Default"
        self.cam.UserSetLoad.Execute()

        self.cam.LineSelector = "Line1"
        self.cam.LineMode = "Input"

        # self.cam.TriggerSelector = "FrameStart"
        self.cam.TriggerSelector = "FrameBurstStart"
     
        self.cam.TriggerSource = "Line1"
        self.cam.TriggerMode = "On"
        # self.cam.TriggerActivation.Value


        self.cam.TriggerActivation.SetValue('RisingEdge')
        print(self.cam.TriggerActivation.GetValue())

        # self.cam.TriggerActivation.SetValue('FallingEdge')
        # print(self.cam.TriggerActivation.GetValue())


        # self.cam.AcquisitionMode.SetValue('SingleFrame')
        self.cam.AcquisitionFrameRateEnable.SetValue(False)
        self.cam.TriggerDelay.SetValue(0) #us


        # cam.ExposureTime = 24500.0 #40
        self.cam.ExposureTime = 16300.0 #60
        # cam.LightSourcePreset.SetValue(py)
        self.cam.LightSourcePreset.SetValue("Daylight5000K")

        # self.cam.AcquisitionFrameRateEnable.SetValue(True)
        # self.cam.AcquisitionFrameRate.SetValue(5)


        print(self.cam.PixelFormat.Symbolics)
        self.cam.PixelFormat.SetValue("RGB8")

        self.cam.AcquisitionBurstFrameCount.SetValue(5)
        Burst=self.cam.AcquisitionBurstFrameCount.GetValue()
        print(Burst)

        print("Using device ", self.cam.GetDeviceInfo().GetModelName())
        self.cam.Width = 1280
        self.cam.Height = 720
        # cam.OffsetX = cam.CenterX.GetValue()
        # cam.OffsetY = cam.CenterY.GetValue()
        self.cam.CenterX.SetValue(True)
        self.cam.CenterY.SetValue(True)
        print(self.cam.OffsetX.GetValue())
        print(self.cam.OffsetY.GetValue())


    def grap(self, img, timestamp):
        self.sendimg.emit(img, timestamp, 1)
        

    def run(self):

        
        # create event handler instance
        triggeredImage = TriggeredImage(self)

        # register handler
        # remove all other handlers
        self.cam.RegisterImageEventHandler(triggeredImage, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_None)

        # start grabbing with background loop

        # self.cam.StartGrabbing(100,pylon.GrabStrategy_LatestImageOnly)
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImages, pylon.GrabLoop_ProvidedByInstantCamera)
        # wait ... or do something relevant

        # res = self.cam.GrabOne(pylon.waitForever)
        
        while self.cam.IsGrabbing():
            time.sleep(0.1)
        # stop grabbing

        self.cam.StopGrabbing()
        print('stop')

        self.cam.Close()





form_class = uic.loadUiType("windows.ui")[0]





class Yolo(QThread):

    def __init__(self, parent, model, classes): 
        super().__init__(parent) 

        self.model = model
        self.parent = parent

        self.input_img = np.zeros([768, 1280, 3], dtype=np.float32)
        self.conf_threshold = 0
        self.iou_threshold = 0
        self.inference_mode = False
        self.classes = classes


    def run(self):


            img=self.parent.current_img

            trainKP, trainDesc, trainImg, im_aspect_ratio, im_height, im_width, imCrop, im_area = search_feature(self.parent.fiducial_img)
            master_fiducial_center, fiducial_box, angle, check = fiducial_marker(img, trainKP, trainDesc, trainImg, im_aspect_ratio, im_height, im_width,im_area)

            if check:
                pass
                # print('Good')
                # print(master_fiducial_center, len(fiducial_box))
            else:
                pass
                # print('False')

            input_img = np.zeros([768, 1280, 3], dtype=np.float32)
                
            input_img[24:744, :, 0] = img[:, :, 2] * 0.003921569 # / 255
            input_img[24:744, :, 1] = img[:, :, 1] * 0.003921569 # / 255
            input_img[24:744, :, 2] = img[:, :, 0] * 0.003921569 # / 255
            input_img = np.ascontiguousarray(input_img)


            infer_t1 = time.time()
            npu_output = self.model.infer([input_img])
            infer_t2 = time.time()


            output = []
            output.append(torch.from_numpy(np.asarray(npu_output[0])).reshape(1, 96, 160, 207).permute(0, 3, 1, 2))
            output.append(torch.from_numpy(np.asarray(npu_output[1])).reshape(1, 48, 80, 207).permute(0, 3, 1, 2))
            output.append(torch.from_numpy(np.asarray(npu_output[2])).reshape(1, 24, 40, 207).permute(0, 3, 1, 2))

            result = post_processing_yolov5(output, self.conf_threshold, self.iou_threshold)[0]

            pr_boxs = []



            result[:, 0].clamp_(0, 1280)
            result[:, 1].clamp_(0, 768)
            result[:, 2].clamp_(0, 1280)
            result[:, 3].clamp_(0, 768)
            # split output
            boxes = result[:, :4].numpy()
            scores = result[:, 4].numpy()
            labels = result[:, 5].numpy()
            boxes[:, 1] -= 24
            boxes[:, 3] -= 24

        
            for i in range(len(scores)):
                Xmin, Ymin, Xmax, Ymax = boxes[i]
                Xmin, Ymin, Xmax, Ymax = int(Xmin), int(Ymin), int(Xmax), int(Ymax)
                description = self.classes[int(labels[i].item())]
                pr_boxs.append({'point': [Xmin,Ymin,Xmax,Ymax] , 'label': description})

            infer_time = infer_t2 - infer_t1
        
            self.parent.fiducial_box = fiducial_box
            self.parent.fiducial_center_point = master_fiducial_center
            self.parent.pr_boxs = pr_boxs
            self.parent.infer_time = infer_time
            # self.sendimg.emit(img,pr_boxs,infer_time)



            ##################################### yolov5 end ##############################################
            # else:
            #     self.sendimg_noinfer.emit(img)




    def stop(self):
        self.quit()
        self.wait(100)  # 3초 대기 (바로 안꺼질수도)
    

    def infer_one(self, img):
        
        input_img = np.zeros([768, 1280, 3], dtype=np.float32)

        input_img[24:744, :, 0] = img[:, :, 2] * 0.003921569 # / 255
        input_img[24:744, :, 1] = img[:, :, 1] * 0.003921569 # / 255
        input_img[24:744, :, 2] = img[:, :, 0] * 0.003921569 # / 255
        input_img = np.ascontiguousarray(input_img)

        infer_t1 = time.time()
        npu_output = self.model.infer([input_img])
        infer_t2 = time.time()

        output = []
        output.append(torch.from_numpy(np.asarray(npu_output[0])).reshape(1, 96, 160, 207).permute(0, 3, 1, 2))
        output.append(torch.from_numpy(np.asarray(npu_output[1])).reshape(1, 48, 80, 207).permute(0, 3, 1, 2))
        output.append(torch.from_numpy(np.asarray(npu_output[2])).reshape(1, 24, 40, 207).permute(0, 3, 1, 2))

        result = post_processing_yolov5(output, self.conf_threshold, self.iou_threshold)[0]

        pr_boxs = []

        result[:, 0].clamp_(0, 1280)
        result[:, 1].clamp_(0, 768)
        result[:, 2].clamp_(0, 1280)
        result[:, 3].clamp_(0, 768)
        # split output
        boxes = result[:, :4].numpy()
        scores = result[:, 4].numpy()
        labels = result[:, 5].numpy()
        boxes[:, 1] -= 24
        boxes[:, 3] -= 24

        for i in range(len(scores)):
            Xmin, Ymin, Xmax, Ymax = boxes[i]
            Xmin, Ymin, Xmax, Ymax = int(Xmin), int(Ymin), int(Xmax), int(Ymax)
            description = self.classes[int(labels[i].item())]
            pr_boxs.append({'point': [Xmin,Ymin,Xmax,Ymax] , 'label': description})

        infer_time = infer_t2 - infer_t1

        return pr_boxs, infer_time


    def infer_burst(self, img_array):
         
        infer_time = 0
        pr_boxs_array=[]
        fiducial_center_array = []

        trainKP, trainDesc, trainImg, im_aspect_ratio, im_height, im_width, imCrop, im_area = search_feature(self.parent.fiducial_img)

        for idx, img in enumerate(img_array):

            master_fiducial_center, fiducial_box, angle, check = fiducial_marker(img, trainKP, trainDesc, trainImg, im_aspect_ratio, im_height, im_width,im_area)

            if check:
                # print('Good')
                # print(master_fiducial_center)
                fiducial_center_array.append(master_fiducial_center)
            else:
                pass
                # print('False')



            input_img = np.zeros([768, 1280, 3], dtype=np.float32)
            
            input_img[24:744, :, 0] = img[:, :, 2] * 0.003921569 # / 255
            input_img[24:744, :, 1] = img[:, :, 1] * 0.003921569 # / 255
            input_img[24:744, :, 2] = img[:, :, 0] * 0.003921569 # / 255
            input_img = np.ascontiguousarray(input_img)

            infer_t1 = time.time()
            npu_output = self.model.infer([input_img])
            infer_t2 = time.time()

            output = []
            output.append(torch.from_numpy(np.asarray(npu_output[0])).reshape(1, 96, 160, 207).permute(0, 3, 1, 2))
            output.append(torch.from_numpy(np.asarray(npu_output[1])).reshape(1, 48, 80, 207).permute(0, 3, 1, 2))
            output.append(torch.from_numpy(np.asarray(npu_output[2])).reshape(1, 24, 40, 207).permute(0, 3, 1, 2))

            result = post_processing_yolov5(output, self.conf_threshold, self.iou_threshold)[0]

            pr_boxs = []

            result[:, 0].clamp_(0, 1280)
            result[:, 1].clamp_(0, 768)
            result[:, 2].clamp_(0, 1280)
            result[:, 3].clamp_(0, 768)
            # split output
            boxes = result[:, :4].numpy()
            scores = result[:, 4].numpy()
            labels = result[:, 5].numpy()
            boxes[:, 1] -= 24
            boxes[:, 3] -= 24

            for i in range(len(scores)):
                Xmin, Ymin, Xmax, Ymax = boxes[i]
                Xmin, Ymin, Xmax, Ymax = int(Xmin), int(Ymin), int(Xmax), int(Ymax)
                description = self.classes[int(labels[i].item())]
                pr_boxs.append({'point': [Xmin,Ymin,Xmax,Ymax] , 'label': description})

            infer_time += infer_t2 - infer_t1
            pr_boxs_array.append(pr_boxs)
        

        infer_time /=5

        return pr_boxs_array, fiducial_center_array, infer_time










        


class WindowClass(QMainWindow, form_class) :
    
    def __init__(self,model=None) :
        super().__init__()
        self.setupUi(self)


        self.gt_boxs = []
        self.pr_boxs = []

        self.label_classes = []

        f = open("classes.txt", 'r')
        lines = f.readlines()
        for line in lines:
            line=line.replace('\n','')
            self.label_classes.append(line)
        f.close()

        self.fiducial_center_point = (0,0)
        self.fiducial_box = []




        # maxCamerasToUse = 2
        
        # tlFactory = pylon.TlFactory.GetInstance()

        # # Get all attached devices and exit application if no device is found.
        # devices = tlFactory.EnumerateDevices()
        # if len(devices) == 0:
        #     raise pylon.RuntimeException("No camera present.")

        # # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
        # cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

        # l = cameras.GetSize()

        # print(l)

        # for i, cam in enumerate(cameras):
        #     cam.Attach(tlFactory.CreateDevice(devices[i]))
        
        
        # self.pyloncam = Pylon(self)
        # self.webcam = Webcam(self)

        self.cam_mode = 'swtrigger'

        self.yolo = Yolo(self,model,self.label_classes)
        
        self.yolo.iou_threshold = 0.1
        self.yolo.conf_threshold = 0.2

        self.inference = False

        self.cam_buff = []
        self.no = 0

        self.insp_iou = 0.5


        self.ft = 0
        self.rt = 0


        self.toolButton_savegtpath.clicked.connect(self.toolButton_savegtpath_fuction)
        self.toolButton_loadgtpath.clicked.connect(self.toolButton_loadgtpath_fuction)
        self.pushButton_savegt.clicked.connect(self.pushButton_savegt_fuction)
        self.pushButton_loadgt.clicked.connect(self.pushButton_loadgt_fuction)
        self.pushButton_set_trigger.clicked.connect(self.pushButton_set_trigger_fuction)

    


        self.pushButton_fiducial.clicked.connect(self.pushButton_fiducial_fuction)

        self.pushButton_swtrigger.clicked.connect(self.pushButton_swtrigger_fuction)

        self.pushButton_pylonopen.clicked.connect(self.pushButton_pylonopen_fuction)
        self.pushButton_webcamopen.clicked.connect(self.pushButton_webcamopen_fuction)


        self.checkBox_inference.stateChanged.connect(self.checkBox_inference_fuction)

        # self.radioButton_continuous.toggled.connect(self.radioButton_continuous_onClicked)
        # self.radioButton_hwtrigger.toggled.connect(self.radioButton_hwtrigger_onClicked)
        # self.radioButton_swtrigger.toggled.connect(self.radioButton_swtrigger_onClicked)

        self.doubleSpinBox_infer_iou.valueChanged.connect(self.doubleSpinBox_infer_iou_value_changed)
        self.doubleSpinBox_infer_conf.valueChanged.connect(self.doubleSpinBox_infer_conf_value_changed)
        self.doubleSpinBox_insp_iou.valueChanged.connect(self.doubleSpinBox_insp_iou_value_changed)

        self.pushButton_setgt.clicked.connect(self.pushButton_setgt_fuction)
        self.pushButton_addgt.clicked.connect(self.pushButton_addgt_fuction)
        self.pushButton_delgt.clicked.connect(self.pushButton_delgt_fuction)

        self.canvas.sum_signal.connect(self.process_sum)
        self.canvas.fiducial_signal.connect(self.fiducial_sum)
    
        self.listWidget_label_classes.itemClicked.connect(self.listWidget_label_classes_itemClicked)

        

        self.canvas.canvas_mode = 'default'

        
        self.savegtpath = './'
        self.loadgtpath = './'

        self.listWidget_label_classes.clear()
 
        for label_class in self.label_classes:
            label_class_item = QListWidgetItem(label_class)
            label_class_item.setBackground(QColor(255, 255, 255))
            self.listWidget_label_classes.addItem(label_class_item)



        self.d = 0

        # self.image = QImage(QSize(400, 400), QImage.Format_RGB32)
        # self.image.fill(Qt.white)
        # self.drawing = False
        # self.brush_size = 5
        # self.brush_color = Qt.black
        # self.last_point = QPoint()
    
    
    def process_sum(self, a):

        if self.canvas.canvas_mode == 'pattern':
            self.canvas.canvas_mode = 'default'
           

            # a = [i*2 for i in a]
            # self.gt_boxs.append({'point': a , 'label': ' '})
            self.gt_boxs.append({'point': [int(a[0]*2), int(a[1]*2), int(a[2]*2), int(a[3]*2)] , 'label': ' '})
            # self.listWidget_gts.add
            
            self.canvas.selbox = len(self.canvas.gt_boxs)-1
            self.update_classes(self.gt_boxs)
            self.canvas.gt_boxs = self.gt_boxs
            self.canvas.update_image(self.gt_img)
            
    

    def fiducial_sum(self, a):

        if self.canvas.canvas_mode == 'fiducial':
            self.canvas.canvas_mode = 'default'
            fiducial_points = a
    
            self.fiducial_img = self.gt_img[int(fiducial_points[1]*2):int(fiducial_points[3]*2), int(fiducial_points[0]*2):int(fiducial_points[2]*2)]

            # # a = [i*2 for i in a]
            # self.gt_boxs.append({'point': a , 'label': ' '})
            # print(a)
            # # self.listWidget_gts.add
            
            # self.canvas.selbox = len(self.canvas.gt_boxs)-1
            # self.update_classes(self.gt_boxs)
            # self.canvas.gt_boxs = self.gt_boxs
            # self.canvas.update_image(self.gt_img)

       
            # self.canvas.draw_gt(self.gt_boxs)



    def update_classes(self, gt_boxs):
        self.listWidget_gts.clear()
        for gt_box in gt_boxs:
            gt = QListWidgetItem(gt_box['label'])
            gt.setBackground(QColor(255, 128, 128))
            self.listWidget_gts.addItem(gt)



    def listWidget_label_classes_itemClicked(self):
        currentRow = self.listWidget_label_classes.currentRow()

        self.gt_boxs[self.canvas.selbox]['label'] = self.label_classes[currentRow].replace('\n','')

        
        self.update_classes(self.gt_boxs)
        self.canvas.gt_boxs = self.gt_boxs
        self.canvas.update_image(self.gt_img)


    def toolButton_savegtpath_fuction(self):
        self.savegtpath=QFileDialog.getExistingDirectory(self,"Choose GT Directory","./")
        self.lineEdit_savegtpath.setText(self.savegtpath)

    def toolButton_loadgtpath_fuction(self):
        self.loadgtpath = QFileDialog.getOpenFileName(self,"Choose Result File","./")
        self.lineEdit_loadgtpath.setText(self.loadgtpath[0])


    def pushButton_swtrigger_fuction(self):
        self.pyloncam.cam.ExecuteSoftwareTrigger()


    def pushButton_savegt_fuction(self):
        
        filename = self.lineEdit_pjname.text() + '.json'

        pjfile = {
            'pjname': self.lineEdit_pjname.text(),
            'gtdata': self.gt_boxs
        }

        print(self.savegtpath+'/'+filename)
        with open(self.savegtpath+'/'+filename, 'w') as outfile:
            json.dump(pjfile, outfile, indent=4)

        


    def pushButton_loadgt_fuction(self):

        try:
            with open(self.loadgtpath[0], 'r') as f:
                result_data = json.load(f)

                self.gt_boxs = result_data['gtdata']

                self.lineEdit_pjname.setText(result_data['pjname'])

                self.update_classes(self.gt_boxs)
                self.canvas.gt_boxs = self.gt_boxs
                self.canvas.update_image(self.gt_img)



        except:
            QMessageBox.information(self,'error','json file error')
            return





    def doubleSpinBox_infer_iou_value_changed(self):
        self.pyloncam.iou_threshold = self.doubleSpinBox_infer_iou.value()

    def doubleSpinBox_infer_conf_value_changed(self):
        self.pyloncam.conf_threshold = self.doubleSpinBox_infer_conf.value()

    def doubleSpinBox_insp_iou_value_changed(self):
        self.insp_iou = self.doubleSpinBox_insp_iou.value()


    def pushButton_fiducial_fuction(self):
        self.canvas.canvas_mode = 'fiducial'


    def pushButton_addgt_fuction(self):
        self.canvas.canvas_mode = 'pattern'
        
    def pushButton_delgt_fuction(self):
        
        if len(self.gt_boxs) > self.canvas.selbox:
            self.gt_boxs.pop(self.canvas.selbox)

            self.update_classes(self.gt_boxs)
            self.canvas.gt_boxs = self.gt_boxs
            self.canvas.update_image(self.gt_img)



    def pushButton_setgt_fuction(self):

        self.gt_img = self.current_img.copy()
        self.canvas.update_image(self.gt_img)
        # self.canvas.gt_img = self.gt_img 


    def pushButton_webcamopen_fuction(self):
        self.webcam = Webcam(self)

        self.pushButton_webcamopen.setEnabled(False)
        self.webcam.start()
        # self.pyloncam.sendimg.connect(self.receiveimg_noinfer)
        self.webcam.sendimg_webcam.connect(self.receiveimg)



    def pushButton_pylonopen_fuction(self):
        
        self.pyloncam = Pylon(self)

        self.pushButton_pylonopen.setEnabled(False)
        self.pyloncam.start()
        # self.pyloncam.sendimg.connect(self.receiveimg_noinfer)
        self.pyloncam.sendimg.connect(self.receiveimg)
        
        # self.yolo.start()
        # self.yolo.sendimg.connect(self.receiveimg)
        

    def pushButton_set_trigger_fuction(self):
        
        if self.radioButton_continuous.isChecked():
            self.cam_mode = 'continuous'
            self.pyloncam.cam.TriggerMode = "Off"
        elif self.radioButton_hwtrigger.isChecked():
            self.cam_mode = 'hwtrigger'
            self.pyloncam.cam.TriggerMode = "On"
            self.pyloncam.cam.TriggerSource = "Line1"
        elif self.radioButton_swtrigger.isChecked():
            self.cam_mode = 'swtrigger'
            self.pyloncam.cam.TriggerMode = "On"
            self.pyloncam.cam.TriggerSource = "Software"
        elif self.radioButton_test.isChecked():
            self.cam_mode = 'test'
            self.pyloncam.cam.TriggerMode = "On"
            self.pyloncam.cam.TriggerSource = "Software"
        

        self.d = 0
        self.cam_buff=[]
        self.pyloncam.cam.TriggerActivation.SetValue('RisingEdge')
        print('RisingEdge')
        self.start_trigger = True






            
          



    def checkBox_inference_fuction(self,state):
        if state == Qt.Checked:
            self.inference = True
            # self.yolo.inference_mode = True
            # self.yolo.start()
            # self.yolo.start()
            # self.yolo.sendimg.connect(self.receiveimg)
            print('infer start')
        else:
            self.inference = False
            # self.yolo.inference_mode = False
            print('infer stop')
            # self.yolo.stop()



    @pyqtSlot(np.ndarray)
    def receiveimg_noinfer(self, img):
        qt_img = self.convert_cv_qt(img)
        painter = QPainter(qt_img)

        for idx, gt_box in enumerate(self.gt_boxs):
            point = gt_box['point']
            x1 = point[0]
            y1 = point[1]
            x2 = point[2]
            y2 = point[3]
            painter.fillRect(x1,y1,x2-x1,y2-y1 ,QColor(255,255,20,128))

        painter.end()

        self.label_screen_cam.setPixmap(qt_img)



            # img=self.parent.current_img

            # trainKP, trainDesc, trainImg, im_aspect_ratio, im_height, im_width, imCrop, im_area = search_feature(self.parent.fiducial_img)
            # master_fiducial_center, fiducial_box, angle, check = fiducial_marker(img, trainKP, trainDesc, trainImg, im_aspect_ratio, im_height, im_width,im_area)

            # if check:
            #     print('Good')
            #     print(master_fiducial_center, len(fiducial_box))
            # else:
            #     print('False')



    def inspection_1(self, gt_boxs, pr_boxs_array, delta_fiducial_array = None):

        gt_boxs_array = []

        result_good = []
        result_false = []
        result_nothing = []

       
        for gt_idx, gt_box in enumerate(gt_boxs):
            
            good = 0
            false = 0
            nothing = 0

            

            for delta_idx, delta_fiducial in enumerate(delta_fiducial_array):
                
                
                gt_point = gt_box['point']
                gt_label = gt_box['label']

                # gt_x1, gt_y1, gt_x2, gt_y2 = gt_box['point'][0], gt_box['point'][1], gt_box['point'][2], gt_box['point'][3]

                gt_point[0] += delta_fiducial[0]
                gt_point[1] += delta_fiducial[1]
                gt_point[2] += delta_fiducial[0]
                gt_point[3] += delta_fiducial[1]
                
                gt_center = ((gt_point[2]-gt_point[0] / 2) , (gt_point[3]-gt_point[1] / 2))

                for pr_idx, pr_box in enumerate(pr_boxs_array[delta_idx]):

                    pr_point = pr_box['point']
                    pr_label = pr_box['label']

                    pr_center = ((pr_point[2]-pr_point[0] / 2) , (pr_point[3]-pr_point[1] / 2))
                    

                    if self.IoU(gt_point, pr_point) >= self.insp_iou:
                        if pr_label == gt_label:
                            recog = True
                            detect = True
                            break
                        else:  
                            recog = False
                            detect = True
                            break
                    else:
                        recog = False
                        detect = False
        
        
                if recog and detect:
                    good += 1
                    # painter.fillRect(gt_x1,gt_y1,gt_x2-gt_x1,gt_y2-gt_y1,QColor(0,255,0,128))
                elif ~recog and detect:
                    false += 1
                    # painter.fillRect(gt_x1,gt_y1,gt_x2-gt_x1,gt_y2-gt_y1 ,QColor(0,0,255,128))
                elif ~recog and ~detect:
                    nothing += 1
                    # painter.fillRect(gt_x1,gt_y1,gt_x2-gt_x1,gt_y2-gt_y1 ,QColor(255,0,0,128))

            
            # gt_boxs_array.append(gt_boxs)


            if good > 2:
                result_good.append(gt_box)
            if false > 2:
                result_false.append(gt_box)
            if nothing > 2:
                result_nothing.append(gt_box)

         
        return result_good, result_false, result_nothing, gt_boxs_array





    def inspection_2(self, gt_boxs, pr_boxs_array, delta_fiducial_array):


        detect_box = []
        e2e_box = []
        FN_box = []
        FP_box = []

        match_boxs = [[] for i in range(len(gt_boxs))]
        post_match_boxs = [{} for i in range(len(gt_boxs))]


        print('pr_boxs_array_before--------------------')
        print(pr_boxs_array)

        for gt_idx, gt_box in enumerate(gt_boxs):

            # print('------------------------------------------------------------------')
            # print('------------------------------------------------------------------')
            # print(gt_box)
            # print('------------------------------------------------------------------')
            # print('------------------------------------------------------------------')


            for delta_idx, delta_fiducial in enumerate(delta_fiducial_array):
            
                gt_point = gt_box['point']
                gt_label = gt_box['label']

                # print('------------------------')
                # print(delta_idx)
                # print('------------------------')


                # gt_center = ((gt_point[2]-gt_point[0] / 2) , (gt_point[3]-gt_point[1] / 2))

                # match_box_candinate = []
                iou_list = []

                # print('pr_boxs_array: ')
                # print(pr_boxs_array[delta_idx])
                # print('delta_fiducial: ')
                # print(delta_fiducial)

                for pr_idx, pr_box in enumerate(pr_boxs_array[delta_idx][:]):
                    
                    pr_point = pr_boxs_array[delta_idx][pr_idx]['point']
                    pr_label = pr_boxs_array[delta_idx][pr_idx]['label']

                    pr_point[0] -= delta_fiducial[0]
                    pr_point[1] -= delta_fiducial[1]
                    pr_point[2] -= delta_fiducial[0]
                    pr_point[3] -= delta_fiducial[1]

                    # pr_center = ((pr_point[2]-pr_point[0] / 2) , (pr_point[3]-pr_point[1] / 2))
                    
                    # if self.IoU(gt_point, pr_point) >= 0.2:
                    iou = self.IoU(gt_point, pr_point)
                    
                    iou_list.append(iou)
                

                # print('iou_list: ')
                # print(iou_list)
                #     # match_box_candinate.append(pr_box)
                

                
                if len(iou_list) > 0.1:
                    max_idx = iou_list.index(max(iou_list))
                    remove_box = pr_boxs_array[delta_idx][max_idx]

                    match_boxs[gt_idx].append(remove_box)
                    pr_boxs_array[delta_idx].remove(remove_box)
        


            if len(match_boxs[gt_idx]) >= 2:
                x1_sum , y1_sum, x2_sum, y2_sum = 0,0,0,0
                label_sum = []
                for match_box in match_boxs[gt_idx]:
                    x1_sum += match_box['point'][0]
                    y1_sum += match_box['point'][1]
                    x2_sum += match_box['point'][2]
                    y2_sum += match_box['point'][3]
                    label_sum.append(match_box['label'])
                
                x1_mean = x1_sum/len(match_boxs[gt_idx])
                y1_mean = y1_sum/len(match_boxs[gt_idx])
                x2_mean = x2_sum/len(match_boxs[gt_idx])
                y2_mean = y2_sum/len(match_boxs[gt_idx])

                if self.IoU(gt_point, [x1_mean, y1_mean, x2_mean, y2_mean]) > self.insp_iou:
                    count_label=Counter(label_sum)
                    max_label=count_label.most_common(n=1)
                    

                    if gt_box['label'] == max_label[0][0]:
                        # e2e_box.append({'point': [x1_mean, y1_mean, x2_mean, y2_mean], 'label': max_label[0][0]})
                        post_match_boxs[gt_idx] = {'point': [x1_mean, y1_mean, x2_mean, y2_mean], 'label': max_label[0][0], 'result':'e2e'}
                    else:
                        # detect_box.append({'point': [x1_mean, y1_mean, x2_mean, y2_mean], 'label': max_label[0][0]})
                        post_match_boxs[gt_idx] = {'point': [x1_mean, y1_mean, x2_mean, y2_mean], 'label': max_label[0][0], 'result':'detect'}
                
                else:
                    post_match_boxs[gt_idx] = {'point': [0, 0, 0, 0], 'label': None, 'result':'FN'}
            else:
                post_match_boxs[gt_idx] = {'point': [0, 0, 0, 0], 'label': None, 'result':'FN'}         
            
            

        print('pr_boxs_array_after--------------------')
        print(pr_boxs_array)

        print('result--------------------')
        print(gt_boxs)
        print(post_match_boxs)

        return post_match_boxs, pr_boxs_array



        
        # print(detect_box)
        # print(FN_box)
            
            


                # print('pr_boxs_array: ')
                # print(pr_boxs_array[delta_idx])

                # print('match_boxs: ')
                # for match_box in match_boxs:
                #     print(match_box)

        
        


    





    def gen_delta_fiducial_point(self,fiducial_center_array):
        delta_fiducial_array = []
        if len(fiducial_center_array) == 5:
            print(fiducial_center_array)
            delta_fiducial_array.append((0,0))
            delta_fiducial_array.append((fiducial_center_array[1][0] - fiducial_center_array[0][0], fiducial_center_array[1][1] - fiducial_center_array[0][1]))
            delta_fiducial_array.append((fiducial_center_array[2][0] - fiducial_center_array[1][0], fiducial_center_array[2][1] - fiducial_center_array[1][1]))
            delta_fiducial_array.append((fiducial_center_array[3][0] - fiducial_center_array[2][0], fiducial_center_array[3][1] - fiducial_center_array[2][1]))
            delta_fiducial_array.append((fiducial_center_array[4][0] - fiducial_center_array[3][0], fiducial_center_array[4][1] - fiducial_center_array[3][1]))
        
        return delta_fiducial_array




    @pyqtSlot(np.ndarray, int, int)
    def receiveimg(self, img, timestamp, camtype):

        


        # self.cam.TriggerActivation.SetValue('FallingEdge')
        # print(self.cam.TriggerActivation.GetValue())


        if camtype == 0:

            if self.cam_mode == 'continuous':
                self.current_img = img

                if self.inference:
                    self.yolo.start()
                
                qt_img = self.convert_cv_qt(img)
                painter = QPainter(qt_img)
                


                for idx, pr_box in enumerate(self.pr_boxs):

                    pr_point = pr_box['point']
                    pr_label = pr_box['label']
                    Xmin, Ymin, Xmax, Ymax = int(pr_point[0]/2), int(pr_point[1]/2), int(pr_point[2]/2), int(pr_point[3]/2)

                    painter.setPen(QPen(QColor(255, 160, 50), 3, Qt.SolidLine))
                    painter.drawRect(Xmin,Ymin,Xmax-Xmin,Ymax-Ymin)
                    painter.setPen(QPen(QColor(0, 0, 0), 1, Qt.SolidLine))
                    painter.drawRect(Xmin,Ymin,Xmax-Xmin,Ymax-Ymin)
                    painter.setPen(QPen(QColor(0, 0, 0), 2, Qt.SolidLine))
                    painter.setFont(QFont('Aria', 10))
                    painter.drawText(Xmin, Ymin, pr_label)

                if self.inference:
                    boxr = []
                    p=self.fiducial_center_point

                    painter.setPen(QPen(QColor(255, 50, 0), 3, Qt.SolidLine))
                    painter.drawPoint(int(p[0]/2),int(p[1]/2))
                    
                    # print(self.fiducial_box)
                    # cv2.waitKey(0)
                    # cv2.destroyWindow()
                    for box in self.fiducial_box:
                        box.tolist()
                        boxr.append(QPoint(int(box[0][0]/2), int(box[0][1]/2)))
                        boxr.append(QPoint(int(box[1][0]/2), int(box[1][1]/2)))
                        boxr.append(QPoint(int(box[2][0]/2), int(box[2][1]/2)))
                        boxr.append(QPoint(int(box[3][0]/2), int(box[3][1]/2)))


            
                    polygon1 = QPolygon(boxr)
                    painter.setPen(QPen(QColor(255, 50, 0), 3))
                    painter.drawPolygon(polygon1) 



                # painter.drawText(8, 12, str(int(1/self.infer_time)))
                painter.end()

                self.label_screen_cam.setPixmap(qt_img)

            elif self.cam_mode == 'swtrigger' or self.cam_mode == 'hwtrigger':
            
                self.current_img = img


                # for idx, img in enumerate(self.cam_buff):
                #     cv2.imwrite(str(idx)+'.jpg',img)
                
                qt_img = self.convert_cv_qt(img)
                painter = QPainter(qt_img)


                if self.inference:
                    pr_boxs_array,fiducial_center_array,self.infer_time=self.yolo.infer_burst(self.cam_buff)
                    delta_fiducial_array = self.gen_delta_fiducial_point(fiducial_center_array)


                    pr_boxs_array_copy = copy.deepcopy(pr_boxs_array)
                    


                    post_match_boxs, pr_boxs_array  = self.inspection_2(self.gt_boxs, pr_boxs_array, delta_fiducial_array)
                    
                    print(pr_boxs_array_copy)
                    
                    
                    for gt_idx, gt_box in enumerate(self.gt_boxs):

                        gt_point = gt_box['point']
                        gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_point[0]/2), int(gt_point[1]/2), int(gt_point[2]/2), int(gt_point[3]/2)

                        if post_match_boxs[gt_idx]['result'] == 'detect':
                            painter.fillRect(gt_x1,gt_y1,gt_x2-gt_x1,gt_y2-gt_y1 ,QColor(0,0,255,128))
                        elif post_match_boxs[gt_idx]['result'] == 'e2e':
                            painter.fillRect(gt_x1,gt_y1,gt_x2-gt_x1,gt_y2-gt_y1,QColor(0,255,0,128))
                        elif post_match_boxs[gt_idx]['result'] == 'FN':
                            painter.fillRect(gt_x1,gt_y1,gt_x2-gt_x1,gt_y2-gt_y1 ,QColor(255,0,0,128))


                    for idx, cam_img in enumerate(self.cam_buff):

                        qt_imgt = self.convert_cv_qt(cam_img)
                        paintert = QPainter(qt_imgt)

                        for gt_idx, gt_box in enumerate(self.gt_boxs):
                            gt_point = gt_box['point']
                            gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_point[0]/2), int(gt_point[1]/2), int(gt_point[2]/2), int(gt_point[3]/2)

                            gt_x1 += delta_fiducial_array[idx][0]
                            gt_y1 += delta_fiducial_array[idx][1]
                            gt_x2 += delta_fiducial_array[idx][0]
                            gt_y2 += delta_fiducial_array[idx][1]

                            paintert.fillRect(gt_x1,gt_y1,gt_x2-gt_x1,gt_y2-gt_y1 ,QColor(0,0,255,128))

                        for pr_box in pr_boxs_array_copy[idx]:

                            pr_point = pr_box['point']
                            pr_label = pr_box['label']
                            Xmin, Ymin, Xmax, Ymax = int(pr_point[0]/2), int(pr_point[1]/2), int(pr_point[2]/2), int(pr_point[3]/2)
                            paintert.setPen(QPen(QColor(255, 160, 50), 3, Qt.SolidLine))
                            paintert.drawRect(Xmin,Ymin,Xmax-Xmin,Ymax-Ymin)
                            paintert.setPen(QPen(QColor(0, 0, 0), 1, Qt.SolidLine))
                            paintert.drawRect(Xmin,Ymin,Xmax-Xmin,Ymax-Ymin)
                            paintert.setPen(QPen(QColor(0, 0, 0), 2, Qt.SolidLine))
                            paintert.setFont(QFont('Aria', 10))
                            paintert.drawText(Xmin, Ymin, pr_label)

                        qt_imgt.save('./c'+str(idx)+'.jpg')
                        paintert.end()



                    for idx, pr_boxs in enumerate(pr_boxs_array):
                        for pr_box in pr_boxs:

                            pr_point = pr_box['point']
                            pr_label = pr_box['label']
                            Xmin, Ymin, Xmax, Ymax = int(pr_point[0]/2), int(pr_point[1]/2), int(pr_point[2]/2), int(pr_point[3]/2)

                            painter.setPen(QPen(QColor(255, 160, 50), 3, Qt.SolidLine))
                            painter.drawRect(Xmin,Ymin,Xmax-Xmin,Ymax-Ymin)
                            painter.setPen(QPen(QColor(0, 0, 0), 1, Qt.SolidLine))
                            painter.drawRect(Xmin,Ymin,Xmax-Xmin,Ymax-Ymin)
                            painter.setPen(QPen(QColor(0, 0, 0), 2, Qt.SolidLine))
                            painter.setFont(QFont('Aria', 10))
                            painter.drawText(Xmin, Ymin, pr_label)

                # for imm in self.cam_buff:
                #     cv2.imwrite('./data2/'+str(time.time())+'.jpg',self.cam_buff[0])

                # painter.drawText(8, 12, str(int(1/self.infer_time)))
                painter.end()
                self.label_screen_cam.setPixmap(qt_img)
               

            elif self.cam_mode == 'test':
                pass
                
              
                

        elif camtype == 1:

            if self.cam_mode == 'continuous':
                self.current_img = img

                if self.inference:
                    self.yolo.start()
                
                qt_img = self.convert_cv_qt(img)
                painter = QPainter(qt_img)
                

                # recog = False
                # detect = False

            
                # for idx, gt_box in enumerate(self.gt_boxs):
                #     gt_point = gt_box['point']
                #     gt_x1, gt_y1, gt_x2, gt_y2 = gt_point[0], gt_point[1], gt_point[2], gt_point[3]
                #     gt_point = [i*2 for i in gt_point]
                #     gt_label = gt_box['label']

                #     for idx, pr_box in enumerate(self.pr_boxs):
                #         pr_point = pr_box['point']
                #         pr_label = pr_box['label']

                #         if self.IoU(gt_point, pr_point) >= self.insp_iou:
                #             if pr_label == gt_label:
                #                 recog = True
                #                 detect = True
                #                 break
                #             else:  
                #                 recog = False
                #                 detect = True
                #                 break
                #         else:
                #             recog = False
                #             detect = False
                
                
                #     if recog and detect:
                #         painter.fillRect(gt_x1,gt_y1,gt_x2-gt_x1,gt_y2-gt_y1,QColor(0,255,0,128))
                #     elif ~recog and detect:
                #         painter.fillRect(gt_x1,gt_y1,gt_x2-gt_x1,gt_y2-gt_y1 ,QColor(0,0,255,128))
                #     elif ~recog and ~detect:
                #         painter.fillRect(gt_x1,gt_y1,gt_x2-gt_x1,gt_y2-gt_y1 ,QColor(255,0,0,128))


                for idx, pr_box in enumerate(self.pr_boxs):

                    pr_point = pr_box['point']
                    pr_label = pr_box['label']
                    Xmin, Ymin, Xmax, Ymax = int(pr_point[0]/2), int(pr_point[1]/2), int(pr_point[2]/2), int(pr_point[3]/2)

                    painter.setPen(QPen(QColor(255, 160, 50), 3, Qt.SolidLine))
                    painter.drawRect(Xmin,Ymin,Xmax-Xmin,Ymax-Ymin)
                    painter.setPen(QPen(QColor(0, 0, 0), 1, Qt.SolidLine))
                    painter.drawRect(Xmin,Ymin,Xmax-Xmin,Ymax-Ymin)
                    painter.setPen(QPen(QColor(0, 0, 0), 2, Qt.SolidLine))
                    painter.setFont(QFont('Aria', 10))
                    painter.drawText(Xmin, Ymin, pr_label)

                if self.inference:
                    boxr = []
                    p=self.fiducial_center_point

                    painter.setPen(QPen(QColor(255, 50, 0), 3, Qt.SolidLine))
                    painter.drawPoint(int(p[0]/2),int(p[1]/2))
                    
                    # print(self.fiducial_box)
                    # cv2.waitKey(0)
                    # cv2.destroyWindow()
                    for box in self.fiducial_box:
                        box.tolist()
                        boxr.append(QPoint(int(box[0][0]/2), int(box[0][1]/2)))
                        boxr.append(QPoint(int(box[1][0]/2), int(box[1][1]/2)))
                        boxr.append(QPoint(int(box[2][0]/2), int(box[2][1]/2)))
                        boxr.append(QPoint(int(box[3][0]/2), int(box[3][1]/2)))


            
                    polygon1 = QPolygon(boxr)
                    painter.setPen(QPen(QColor(255, 50, 0), 3))
                    painter.drawPolygon(polygon1) 



                # painter.drawText(8, 12, str(int(1/self.infer_time)))
                painter.end()

                self.label_screen_cam.setPixmap(qt_img)

            elif self.cam_mode == 'swtrigger' or self.cam_mode == 'hwtrigger':
                


                if len(self.cam_buff) == 0:
                    self.cam_buff.append(img)
                elif len(self.cam_buff) < 4:
                    self.cam_buff.append(img)
                elif len(self.cam_buff) == 4:
  

                    if self.pyloncam.cam.TriggerActivation.GetValue() == 'RisingEdge':
                        self.rt = time.time()
                    
                        if self.start_trigger == False and self.rt - self.ft < 1:
                            self.cam_buff=[]
                            return 0
                        else:
                            self.pyloncam.cam.TriggerActivation.SetValue('FallingEdge')
                            print('FallingEdge')
                        

                    if self.pyloncam.cam.TriggerActivation.GetValue() == 'FallingEdge':
                        self.ft = time.time()

                        self.pyloncam.cam.TriggerActivation.SetValue('RisingEdge')
                        print('RisingEdge')
                        if self.start_trigger == True:
                            self.start_trigger = False




                    self.cam_buff.append(img)
                    img=self.cam_buff[0]
                    self.current_img = img


                    # for idx, img in enumerate(self.cam_buff):
                    #     cv2.imwrite(str(idx)+'.jpg',img)
                    
                    qt_img = self.convert_cv_qt(img)
                    painter = QPainter(qt_img)


                    
                    if self.inference:
                        pr_boxs_array,fiducial_center_array,self.infer_time=self.yolo.infer_burst(self.cam_buff)
                        delta_fiducial_array = self.gen_delta_fiducial_point(fiducial_center_array)


                        pr_boxs_array_copy = copy.deepcopy(pr_boxs_array)
                        


                        post_match_boxs, pr_boxs_array  = self.inspection_2(self.gt_boxs, pr_boxs_array, delta_fiducial_array)
                        
                        print(pr_boxs_array_copy)
                        
                        
                        for gt_idx, gt_box in enumerate(self.gt_boxs):

                            gt_point = gt_box['point']
                            gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_point[0]/2), int(gt_point[1]/2), int(gt_point[2]/2), int(gt_point[3]/2)

                            if post_match_boxs[gt_idx]['result'] == 'detect':
                                painter.fillRect(gt_x1,gt_y1,gt_x2-gt_x1,gt_y2-gt_y1 ,QColor(0,0,255,128))
                            elif post_match_boxs[gt_idx]['result'] == 'e2e':
                                painter.fillRect(gt_x1,gt_y1,gt_x2-gt_x1,gt_y2-gt_y1,QColor(0,255,0,128))
                            elif post_match_boxs[gt_idx]['result'] == 'FN':
                                painter.fillRect(gt_x1,gt_y1,gt_x2-gt_x1,gt_y2-gt_y1 ,QColor(255,0,0,128))


                        for idx, cam_img in enumerate(self.cam_buff):

                            qt_imgt = self.convert_cv_qt(cam_img)
                            paintert = QPainter(qt_imgt)

                            for gt_idx, gt_box in enumerate(self.gt_boxs):
                                gt_point = gt_box['point']
                                gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_point[0]/2), int(gt_point[1]/2), int(gt_point[2]/2), int(gt_point[3]/2)

                                gt_x1 += delta_fiducial_array[idx][0]
                                gt_y1 += delta_fiducial_array[idx][1]
                                gt_x2 += delta_fiducial_array[idx][0]
                                gt_y2 += delta_fiducial_array[idx][1]

                                paintert.fillRect(gt_x1,gt_y1,gt_x2-gt_x1,gt_y2-gt_y1 ,QColor(0,0,255,128))

                            for pr_box in pr_boxs_array_copy[idx]:

                                pr_point = pr_box['point']
                                pr_label = pr_box['label']
                                Xmin, Ymin, Xmax, Ymax = int(pr_point[0]/2), int(pr_point[1]/2), int(pr_point[2]/2), int(pr_point[3]/2)
                                paintert.setPen(QPen(QColor(255, 160, 50), 3, Qt.SolidLine))
                                paintert.drawRect(Xmin,Ymin,Xmax-Xmin,Ymax-Ymin)
                                paintert.setPen(QPen(QColor(0, 0, 0), 1, Qt.SolidLine))
                                paintert.drawRect(Xmin,Ymin,Xmax-Xmin,Ymax-Ymin)
                                paintert.setPen(QPen(QColor(0, 0, 0), 2, Qt.SolidLine))
                                paintert.setFont(QFont('Aria', 10))
                                paintert.drawText(Xmin, Ymin, pr_label)

                            qt_imgt.save('./c'+str(idx)+'.jpg')
                            paintert.end()



                        for idx, pr_boxs in enumerate(pr_boxs_array):
                            for pr_box in pr_boxs:

                                pr_point = pr_box['point']
                                pr_label = pr_box['label']
                                Xmin, Ymin, Xmax, Ymax = int(pr_point[0]/2), int(pr_point[1]/2), int(pr_point[2]/2), int(pr_point[3]/2)

                                painter.setPen(QPen(QColor(255, 160, 50), 3, Qt.SolidLine))
                                painter.drawRect(Xmin,Ymin,Xmax-Xmin,Ymax-Ymin)
                                painter.setPen(QPen(QColor(0, 0, 0), 1, Qt.SolidLine))
                                painter.drawRect(Xmin,Ymin,Xmax-Xmin,Ymax-Ymin)
                                painter.setPen(QPen(QColor(0, 0, 0), 2, Qt.SolidLine))
                                painter.setFont(QFont('Aria', 10))
                                painter.drawText(Xmin, Ymin, pr_label)

                    for imm in self.cam_buff:
                        cv2.imwrite('./data2/'+str(time.time())+'.jpg',self.cam_buff[0])


                    # painter.drawText(8, 12, str(int(1/self.infer_time)))
                    painter.end()
                    self.cam_buff=[]

                    self.label_screen_cam.setPixmap(qt_img)
               
            elif self.cam_mode == 'test':
                
                
                if self.d == 0:
                    self.start = time.time()
                    self.cam_buff.append(img)
                    self.d += 1
                    # print(start)
                elif self.d == 199:
                    self.cam_buff.append(img)
                    self.end = time.time()
                    self.d = 0
                    print(f"{self.end - self.start:.5f} sec")
                    print(len(self.cam_buff))

                    self.start2 = time.time()
                    for idx, saimg in enumerate(self.cam_buff):
                        cv2.imwrite('./a/'+str(idx)+'.jpg',saimg)
                    self.end2 = time.time()
                    print(f"img 200 save {self.end2 - self.start2:.5f} sec")
                    

                    


                else:
                    self.cam_buff.append(img)
                    self.d += 1
                


                

                qt_img = self.convert_cv_qt(img)
                self.label_screen_cam.setPixmap(qt_img)
                
    

    def IoU(self, box1, box2):
        # box = (x1, y1, x2, y2)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # obtain x1, y1, x2, y2 of the intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # compute the width and height of the intersection
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)

        inter = w * h
        iou = inter / (box1_area + box2_area - inter)
        return iou



    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 360, Qt.KeepAspectRatio, Qt.FastTransformation)
        return QPixmap.fromImage(p)



   




if __name__ == "__main__" :
    
    
    val1 = int(sys.argv[1])

    if val1 == 1 :

        sys.path.append('/home/ai/workspace/x86/maccel/')
        sys.path.append('.')
        import maccel

        acc = maccel.Accelerator()
        model = maccel.Model('./yolov5.mxq')

        model.launch(acc)


        app = QApplication(sys.argv) 
        myWindow = WindowClass(model) 
        myWindow.show()
        app.exec_()
    
    else:

        app = QApplication(sys.argv) 
        myWindow = WindowClass() 
        myWindow.show()
        app.exec_()


