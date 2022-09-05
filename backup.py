#------------------pyqt5 import start-------------------
from ossaudiodev import SNDCTL_DSP_GETBLKSIZE
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui
from PyQt5.QtCore import *
from PyQt5 import *
from PyQt5.QtGui import *
#------------------pyqt5 import end---------------------

#------------------pylon import start-------------------
from tkinter import CENTER
import pypylon.pylon as py
import pypylon.genicam as geni
#------------------pylon import end---------------------

#------------------common import start-------------------
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import pandas as pd
#------------------common import end-------------------

from canvas import Canvas
import json
# from TrackingAPI8_0915 import *   


sys.path.append('/home/mobilint/workspace/x86/maccel/')
sys.path.append('.')
import maccel
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


# definition of event handler class 
class TriggeredImage(py.ImageEventHandler):

    grap = pyqtSignal(np.ndarray)

    def __init__(self, parent):
        super().__init__() 
        self.parent = parent

        # self.grab_times = []
    def OnImageGrabbed(self, camera, grabResult):
        # self.grab_times.append(grabResult.TimeStamp)
        # print(grabResult.TimeStamp)
        # print("SizeX: ", grabResult.GetWidth())
        # print("SizeY: ", grabResult.GetHeight())

        # converter = py.ImageFormatConverter()

        # # converting to opencv bgr format
        # converter.OutputPixelFormat = py.PixelType_BGR8packed
        # converter.OutputBitAlignment = py.OutputBitAlignment_MsbAligned

        # image = converter.Convert(grabResult)
        # img = image.GetArray()

        img = grabResult.GetArray()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # open_cv_image = np.array(img) 
        # cv2.imwrite('asd.jpg',img)
        self.parent.grap(img)
        
        grabResult.Release()




 
 
class Pylon(QThread):
    
    sendimg = pyqtSignal(np.ndarray, list, float)
    sendimg_noinfer = pyqtSignal(np.ndarray)

    def __init__(self, parent, model, classes): 
        super().__init__(parent) 

        self.model = model


        self.parent = parent

        # open the camera
        tlf = py.TlFactory.GetInstance()
        self.cam = py.InstantCamera(tlf.CreateFirstDevice())
        self.cam.Open()

        # get clean powerup state
        self.cam.UserSetSelector = "Default"
        self.cam.UserSetLoad.Execute()

        self.cam.LineSelector = "Line1"
        self.cam.LineMode = "Input"

        # self.cam.TriggerSelector = "FrameStart"
        self.cam.TriggerSelector = "FrameBurstStart"
     
        self.cam.TriggerSource = "Line1"
        self.cam.TriggerMode = "Off"
        # self.cam.TriggerActivation.Value

        # self.cam.AcquisitionMode.SetValue('SingleFrame')
        self.cam.AcquisitionFrameRateEnable.SetValue(False)
        self.cam.TriggerDelay.SetValue(0) #us


        # cam.ExposureTime = 24500.0 #40
        self.cam.ExposureTime = 16300.0 #60
        # cam.LightSourcePreset.SetValue(py)
        self.cam.LightSourcePreset.SetValue("Daylight5000K")

        # self.cam.AcquisitionFrameRateEnable.SetValue(True)
        # self.cam.AcquisitionFrameRate.SetValue(100)


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


        self.input_img = np.zeros([768, 1280, 3], dtype=np.float32)
        self.conf_threshold = 0
        self.iou_threshold = 0

        self.inference_mode = False
        self.classes = classes

    def grap(self, img):
        
        self.parent.current_img = img
        # ##################################### yolov5 start ##############################################
        
        # if self.inference_mode == True:
           
        #     self.input_img[24:744, :, 0] = img[:, :, 2] * 0.003921569 # / 255
        #     self.input_img[24:744, :, 1] = img[:, :, 1] * 0.003921569 # / 255
        #     self.input_img[24:744, :, 2] = img[:, :, 0] * 0.003921569 # / 255
        #     self.input_img = np.ascontiguousarray(self.input_img)


        #     infer_t1 = time.time()
        #     npu_output = self.model.infer([self.input_img])
        #     infer_t2 = time.time()


        #     output = []
        #     output.append(torch.from_numpy(np.asarray(npu_output[0])).reshape(1, 96, 160, 207).permute(0, 3, 1, 2))
        #     output.append(torch.from_numpy(np.asarray(npu_output[1])).reshape(1, 48, 80, 207).permute(0, 3, 1, 2))
        #     output.append(torch.from_numpy(np.asarray(npu_output[2])).reshape(1, 24, 40, 207).permute(0, 3, 1, 2))

        #     result = post_processing_yolov5(output, self.conf_threshold, self.iou_threshold)[0]

        #     pr_boxs = []



        #     result[:, 0].clamp_(0, 1280)
        #     result[:, 1].clamp_(0, 768)
        #     result[:, 2].clamp_(0, 1280)
        #     result[:, 3].clamp_(0, 768)
        #     # split output
        #     boxes = result[:, :4].numpy()
        #     scores = result[:, 4].numpy()
        #     labels = result[:, 5].numpy()
        #     boxes[:, 1] -= 24
        #     boxes[:, 3] -= 24

          
        #     for i in range(len(scores)):
        #         Xmin, Ymin, Xmax, Ymax = boxes[i]
        #         Xmin, Ymin, Xmax, Ymax = int(Xmin), int(Ymin), int(Xmax), int(Ymax)
        #         description = self.classes[int(labels[i].item())]
        #         pr_boxs.append({'point': [Xmin,Ymin,Xmax,Ymax] , 'label': description})

        #     infer_time = infer_t2 - infer_t1
        
           
        #     self.sendimg.emit(img,pr_boxs,infer_time)



        # ##################################### yolov5 end ##############################################
        # else:
        #     self.sendimg_noinfer.emit(img)


    def run(self):

        
        # create event handler instance
        triggeredImage = TriggeredImage(self)

        # register handler
        # remove all other handlers
        self.cam.RegisterImageEventHandler(triggeredImage, py.RegistrationMode_ReplaceAll, py.Cleanup_None)

        # start grabbing with background loop

        # self.cam.StartGrabbing(100,py.GrabStrategy_LatestImageOnly)
        self.cam.StartGrabbing(py.GrabStrategy_LatestImages, py.GrabLoop_ProvidedByInstantCamera)
        # wait ... or do something relevant

        # res = self.cam.GrabOne(py.waitForever)
        
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
        
        if self.inference_mode == True:
           
            self.input_img[24:744, :, 0] = img[:, :, 2] * 0.003921569 # / 255
            self.input_img[24:744, :, 1] = img[:, :, 1] * 0.003921569 # / 255
            self.input_img[24:744, :, 2] = img[:, :, 0] * 0.003921569 # / 255
            self.input_img = np.ascontiguousarray(self.input_img)


            infer_t1 = time.time()
            npu_output = self.model.infer([self.input_img])
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
        
           
            self.sendimg.emit(img,pr_boxs,infer_time)



        ##################################### yolov5 end ##############################################
        else:
            self.sendimg_noinfer.emit(img)

            

    

    def infer_one(self, img):
         
        self.input_img[24:744, :, 0] = img[:, :, 2] * 0.003921569 # / 255
        self.input_img[24:744, :, 1] = img[:, :, 1] * 0.003921569 # / 255
        self.input_img[24:744, :, 2] = img[:, :, 0] * 0.003921569 # / 255
        self.input_img = np.ascontiguousarray(self.input_img)

        infer_t1 = time.time()
        npu_output = self.model.infer([self.input_img])
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
        for img in img_array:

            self.input_img[24:744, :, 0] = img[:, :, 2] * 0.003921569 # / 255
            self.input_img[24:744, :, 1] = img[:, :, 1] * 0.003921569 # / 255
            self.input_img[24:744, :, 2] = img[:, :, 0] * 0.003921569 # / 255
            self.input_img = np.ascontiguousarray(self.input_img)

            infer_t1 = time.time()
            npu_output = self.model.infer([self.input_img])
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

        return pr_boxs_array, infer_time










        


class WindowClass(QMainWindow, form_class) :
    
    def __init__(self,model) :
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


        self.pylon = Pylon(self,model,self.label_classes)
        
        self.pylon.iou_threshold = 0.1
        self.pylon.conf_threshold = 0.4
        self.insp_iou = 0.5

        self.toolButton_savegtpath.clicked.connect(self.toolButton_savegtpath_fuction)
        self.toolButton_loadgtpath.clicked.connect(self.toolButton_loadgtpath_fuction)
        self.pushButton_savegt.clicked.connect(self.pushButton_savegt_fuction)
        self.pushButton_loadgt.clicked.connect(self.pushButton_loadgt_fuction)

    



        self.pushButton_swtrigger.clicked.connect(self.pushButton_swtrigger_fuction)

        self.pushButton_camopen.clicked.connect(self.pushButton_camopen_fuction)


        self.checkBox_inference.stateChanged.connect(self.checkBox_inference_fuction)

        self.radioButton_continuous.toggled.connect(self.radioButton_continuous_onClicked)
        self.radioButton_hwtrigger.toggled.connect(self.radioButton_hwtrigger_onClicked)
        self.radioButton_swtrigger.toggled.connect(self.radioButton_swtrigger_onClicked)

        self.doubleSpinBox_infer_iou.valueChanged.connect(self.doubleSpinBox_infer_iou_value_changed)
        self.doubleSpinBox_infer_conf.valueChanged.connect(self.doubleSpinBox_infer_conf_value_changed)
        self.doubleSpinBox_insp_iou.valueChanged.connect(self.doubleSpinBox_insp_iou_value_changed)

        self.pushButton_setgt.clicked.connect(self.pushButton_setgt_fuction)
        self.pushButton_addgt.clicked.connect(self.pushButton_addgt_fuction)
        self.pushButton_delgt.clicked.connect(self.pushButton_delgt_fuction)

        self.canvas.sum_signal.connect(self.process_sum)
    
        self.listWidget_label_classes.itemClicked.connect(self.listWidget_label_classes_itemClicked)

        

        self.canvas.canvas_mode = 'default'

        
        self.savegtpath = './'
        self.loadgtpath = './'

        self.listWidget_label_classes.clear()
 
        for label_class in self.label_classes:
            label_class_item = QListWidgetItem(label_class)
            label_class_item.setBackground(QColor(255, 255, 255))
            self.listWidget_label_classes.addItem(label_class_item)



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
            self.gt_boxs.append({'point': a , 'label': ' '})
            print(a)
            # self.listWidget_gts.add
            
            self.canvas.selbox = len(self.canvas.gt_boxs)-1
            self.update_classes(self.gt_boxs)
            self.canvas.gt_boxs = self.gt_boxs
            self.canvas.update_image(self.gt_img)
            
    

       
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
        self.pylon.cam.ExecuteSoftwareTrigger()


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
        self.pylon.iou_threshold = self.doubleSpinBox_infer_iou.value()

    def doubleSpinBox_infer_conf_value_changed(self):
        self.pylon.conf_threshold = self.doubleSpinBox_infer_conf.value()

    def doubleSpinBox_insp_iou_value_changed(self):
        self.insp_iou = self.doubleSpinBox_insp_iou.value()



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


    def pushButton_camopen_fuction(self):

        self.pushButton_camopen.setEnabled(False)
        self.pylon.start()
        self.pylon.sendimg.connect(self.receiveimg)
        self.pylon.sendimg_noinfer.connect(self.receiveimg_noinfer)


    def radioButton_continuous_onClicked(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            print('continuous on')
        else:
            print('continuous off')

    def radioButton_hwtrigger_onClicked(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            print('hwtrigger on')
            self.pylon.cam.TriggerMode = "On"
            self.pylon.cam.TriggerSource = "Line1"
        else:
            print('hwtrigger off')
            self.pylon.cam.TriggerMode = "Off"
    
    def radioButton_swtrigger_onClicked(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            print('swtrigger on')
            self.pylon.cam.TriggerMode = "On"
            self.pylon.cam.TriggerSource = "Software"
        else:
            print('swtrigger off')
            self.pylon.cam.TriggerMode = "Off"
            
          



    def checkBox_inference_fuction(self,state):
        if state == Qt.Checked:
            self.pylon.inference_mode = True
        else:
            self.pylon.inference_mode = False


    @pyqtSlot(np.ndarray)
    def receiveimg_noinfer(self, img):
        self.current_img = img
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



    @pyqtSlot(np.ndarray, list, float)
    def receiveimg(self, img, pr_boxs, infer_time):


        self.current_img = img
        
        self.pr_boxs = pr_boxs
        qt_img = self.convert_cv_qt(img)
        painter = QPainter(qt_img)



        recog = False
        detect = False

      
        for idx, gt_box in enumerate(self.gt_boxs):
            gt_point = gt_box['point']
            gt_x1, gt_y1, gt_x2, gt_y2 = gt_point[0], gt_point[1], gt_point[2], gt_point[3]
            gt_point = [i*2 for i in gt_point]
            gt_label = gt_box['label']

            for idx, pr_box in enumerate(self.pr_boxs):
                pr_point = pr_box['point']
                pr_label = pr_box['label']

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
                painter.fillRect(gt_x1,gt_y1,gt_x2-gt_x1,gt_y2-gt_y1,QColor(0,255,0,128))
            elif ~recog and detect:
                painter.fillRect(gt_x1,gt_y1,gt_x2-gt_x1,gt_y2-gt_y1 ,QColor(0,0,255,128))
            elif ~recog and ~detect:
                painter.fillRect(gt_x1,gt_y1,gt_x2-gt_x1,gt_y2-gt_y1 ,QColor(255,0,0,128))


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


        painter.drawText(8, 12, str(int(1/infer_time)))
        painter.end()

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
    
    
    acc = maccel.Accelerator()
    model = maccel.Model('./yolov5.mxq')

    model.launch(acc)


    


    app = QApplication(sys.argv) 
    myWindow = WindowClass(model) 
    myWindow.show()
    app.exec_()

