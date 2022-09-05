from distutils.command.sdist import sdist
import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5 import uic, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import QPainter, QBrush, QPen, QPixmap
from PyQt5.QtCore import *
from cv2 import THRESH_OTSU
from canvas import Canvas
import cv2
import numpy as np
import math
import os
from drawpannel import Drawpannel
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from skimage.metrics import structural_similarity as ssim
import imutils
import glob
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtGui import QStandardItem
import json





        

def clickable(widget):
        class Filter(QObject):
            clicked = pyqtSignal()	#pyside2 사용자는 pyqtSignal() -> Signal()로 변경
            def eventFilter(self, obj, event):
                if obj == widget:
                    if event.type() == QEvent.MouseButtonRelease:
                        if obj.rect().contains(event.pos()):
                            self.clicked.emit()
                            # The developer can opt for .emit(obj) to get the object within the slot.
                            return True
                
                return False
        
        filter = Filter(widget)
        widget.installEventFilter(filter)
        return filter.clicked


form_class = uic.loadUiType("windows.ui")[0]



class WindowClass(QMainWindow, form_class) :
 
    change_pixmap_signal = pyqtSignal(np.ndarray)
    def __init__(self) :
        super().__init__()
       
        
        self._run_flag = True
        self.cam_mode = 'default'

        self.crop_img_on = False


        self.setupUi(self)

        self.pushButton_opencam.clicked.connect(self.opencam)

     
        self.pushButton_drawbox.clicked.connect(self.drawbox)
        
        self.pushButton_default.clicked.connect(self.default)
        self.pushButton_test.clicked.connect(self.pushButton_testFunction)

        self.pushButton_push.clicked.connect(self.pushButton_pushFunction)
        self.pushButton_pop.clicked.connect(self.pushButton_popFunction)

        self.pushButton_scaleup.clicked.connect(self.canvas.zoomIn)
        self.pushButton_scaledown.clicked.connect(self.canvas.zoomOut)

        self.canvas.sum_signal.connect(self.process_sum)



        self.pushButton_ssim.clicked.connect(self.pushButton_ssimFunction)
        self.pushButton_ahm.clicked.connect(self.pushButton_ahmFunction)


        self.listWidget_file.itemDoubleClicked.connect(self.listWidget_fileFunction)
        self.listWidget_font.itemDoubleClicked.connect(self.listWidget_fontFunction)
        


        clickable(self.drawpannel_ssim).connect(self.drawpannel_ssim_clicked)
        clickable(self.drawpannel_ssim_2).connect(self.drawpannel_ssim_2_clicked)
        clickable(self.drawpannel_ssim_3).connect(self.drawpannel_ssim_3_clicked)
        clickable(self.drawpannel_ahm).connect(self.drawpannel_ahm_clicked)
        clickable(self.drawpannel_ahm_2).connect(self.drawpannel_ahm_2_clicked)
        clickable(self.drawpannel_ahm_3).connect(self.drawpannel_ahm_3_clicked)



    
    def drawpannel_ssim_clicked(self):
        cam_crop_qt_img = self.convert_cv_qt_preview(self.textimg)
        
        self.preview.setPixmap(cam_crop_qt_img)
        self.current_font = self.label_ssim.text()
        self.current_font_size = self.label_ssim_size.text()
    def drawpannel_ssim_2_clicked(self):
        cam_crop_qt_img = self.convert_cv_qt_preview(self.textimg2)
        self.preview.setPixmap(cam_crop_qt_img)
        self.current_font = self.label_ssim_2.text()
        self.current_font_size = self.label_ssim_size_2.text()
    def drawpannel_ssim_3_clicked(self):
        cam_crop_qt_img = self.convert_cv_qt_preview(self.textimg3)
        self.preview.setPixmap(cam_crop_qt_img)
        self.current_font = self.label_ssim_3.text()
        self.current_font_size = self.label_ssim_size_3.text()
    def drawpannel_ahm_clicked(self):
        cam_crop_qt_img = self.convert_cv_qt_preview(self.textimg4)
        self.preview.setPixmap(cam_crop_qt_img)
        self.current_font = self.label_ahm.text()
        self.current_font_size = self.label_ahm_size.text()
    def drawpannel_ahm_2_clicked(self):
        cam_crop_qt_img = self.convert_cv_qt_preview(self.textimg5)
        self.preview.setPixmap(cam_crop_qt_img)
        self.current_font = self.label_ahm_2.text()
        self.current_font_size = self.label_ahm_size_2.text()
    def drawpannel_ahm_3_clicked(self):
        cam_crop_qt_img = self.convert_cv_qt_preview(self.textimg6)
        self.preview.setPixmap(cam_crop_qt_img)
        self.current_font = self.label_ahm_3.text()
        self.current_font_size = self.label_ahm_size_3.text()



    def pushButton_testFunction(self):
        # self.canvas.draw_rect(0,0,100,100)
        # print('asdasdasdasd')
        # cv2.imshow('asd',self.dst_crop)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        try:
            maskp = self.drawpannel.maskp

            cv2.grabCut(self.crop_img, maskp, self.rc, self.bgdModel, self.fgdModel, 1, cv2.GC_INIT_WITH_MASK)
            mask2 = np.where((maskp == 2) | (maskp == 0), 0, 1).astype('uint8')
            self.dst = self.crop_img * mask2[:, :, np.newaxis]

            # cv2.imshow('dst', dst)
            self.drawpannel.paint(self.dst,maskp, self.rc)
        except:
            pass
 
    
   

    def convert_cv_qt(self, cv_img, width, height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(width, height, Qt.KeepAspectRatio, Qt.FastTransformation)
        return QPixmap.fromImage(p)

    
    def convert_cv_qt_preview(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(w, h, Qt.KeepAspectRatio, Qt.FastTransformation)
        return QPixmap.fromImage(p)

    


    @pyqtSlot(list)
    def process_sum(self, a):
        self.lineEdit.setText(str(a))

        self.crop_img = self.img[a[1]:a[3],a[0]:a[2]]
        # self.crop_img = crop_img.copy()
        

        self.rc = (0,0,self.crop_img.shape[0],self.crop_img.shape[1])
        
        self.mask = np.zeros(self.crop_img.shape[:2], np.uint8)  # 마스크
        self.bgdModel = np.zeros((1, 65), np.float64)  # 배경 모델
        self.fgdModel = np.zeros((1, 65), np.float64)  # 전경 모델

        try:
            cv2.grabCut(self.crop_img, self.mask, self.rc, self.bgdModel, self.fgdModel, 1, cv2.GC_INIT_WITH_RECT)
        except:
            pass
        # 0: cv2.GC_BGD, 2: cv2.GC_PR_BGD
        mask2 = np.where((self.mask == 0) | (self.mask == 2), 0, 1).astype('uint8')
        self.dst = self.crop_img * mask2[:, :, np.newaxis]

        self.drawpannel.paint(self.dst , self.mask, self.rc)


        

        # cam_crop_qt_img = self.convert_cv_qt(crop_img)
        # self.drawpannel.setPixmap(cam_crop_qt_img)
        # self.drawpannel.setScaledContents(True)

        # # cv2.imshow('targetTh',cam_gray_lpf_canny_crop_cv_img)

        # cv2.imshow("Max-dddfd", cam_gray_lpf_canny_cv_img)

        # self.crop_img_on = True

    


    def img_pre(self, img):

        img_thresh = cv2.Canny(img, 50, 200)
        return img_thresh





    def listWidget_fontFunction(self):
        
        fontdsize=self.listWidget_font.currentItem().text()
        fontlabel = fontdsize.split(',')
   

        ws = int(fontlabel[1])
        wh = int(fontlabel[2])

        
        cimg=cv2.imread('./img/'+self.current_file)
        f_r = open('./gt/' + self.current_file[:-4]+'.txt', 'r')
        lines = f_r.readlines()

        for line in lines:
            line = line.strip()
            label = line.split(',')
            ftext = label[4][1:-1]

            x1 = int(label[0])
            y1 = int(label[1])
            x2 = int(label[2])
            y2 = int(label[3])

            ws = x2-x1
            wh = y2-y1
        
            font_path = './font/' + fontlabel[0]
            font_size = 200
            draw_text=ftext
            
            font = ImageFont.truetype(font_path, font_size)
            back_text_width = int(font_size*1.5)
            back_text_height = int(font_size*1.5)
            
            canvas = Image.new('RGB', (back_text_width, back_text_height), (0,0,0))
            draw = ImageDraw.Draw(canvas)
            w, h = font.getsize(draw_text)
            draw.text(((back_text_width - w) / 2.0, (back_text_height-h) / 2.0), draw_text, (255,255,255), font)
            # canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
            # canvas = canvas.rotate(rotation, expand=False, fillcolor="black")
            textimg=np.array(canvas)

            # cv2.imshow('sds2',textimg)

            ret, thresh = cv2.threshold(textimg[:, :,2], 1, 255, cv2.THRESH_BINARY)       
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(cnt) for cnt in contours]
            top_x = min([x for (x, y, w, h) in rects])
            top_y = min([y for (x, y, w, h) in rects])
            bottom_x = max([x + w for (x, y, w, h) in rects])
            bottom_y = max([y + h for (x, y, w, h) in rects])
            textimg=textimg[top_y:bottom_y, top_x:bottom_x]

            textimg = cv2.resize(textimg, dsize=(ws, wh), interpolation=cv2.INTER_AREA)

            cimg[y1:y2,x1:x2] = textimg


        cam_crop_qt_img = self.convert_cv_qt(cimg,640,480)
        self.label_pscreen.setPixmap(cam_crop_qt_img)




    def listWidget_fileFunction(self):
        
        path = './img/'
        self.current_file = self.file_list_jpg[self.listWidget_file.currentRow()]

        self.img=cv2.imread(path + self.current_file)
        self.canvas.update_image(self.img)


        self.listWidget_font.clear()
        for item in self.fontjson[self.current_file]:
            itemtext = str(item['font']) +','+ str(item['size'])
            self.listWidget_font.addItem(itemtext)



  

    def opencam(self):

        path = './img/'
        file_list = os.listdir(path)
        self.file_list_jpg = [file for file in file_list if file.endswith(".jpg")]


        if os.path.isfile('./font.json'):
            for idx, file_jpg in enumerate(self.file_list_jpg):
                self.listWidget_file.addItem(file_jpg)
            with open('./font.json', 'r') as f:
                self.fontjson = json.load(f)
        else:
            self.fontjson = {}
            for idx, file_jpg in enumerate(self.file_list_jpg):
                self.listWidget_file.addItem(file_jpg)
                self.fontjson[file_jpg]=[]

            with open('./font.json', 'w', encoding='utf-8') as make_file:
                json.dump(self.fontjson, make_file, indent="\t")

    

    def pushButton_pushFunction(self):
        self.fontjson[self.current_file].append({'font': self.current_font , 'size': self.current_font_size})

        with open('./font.json', 'w', encoding='utf-8') as make_file:
            json.dump(self.fontjson, make_file, indent="\t")

        self.listWidget_font.clear()
        for item in self.fontjson[self.current_file]:
            itemtext = str(item['font']) +','+ str(item['size'])
            self.listWidget_font.addItem(itemtext)



    def pushButton_popFunction(self):
        

        fontidx=self.listWidget_font.currentRow()
        self.fontjson[self.current_file].pop(fontidx)
  


        with open('./font.json', 'w', encoding='utf-8') as make_file:
            json.dump(self.fontjson, make_file, indent="\t")

        self.listWidget_font.clear()
        for item in self.fontjson[self.current_file]:
            itemtext = str(item['font']) +','+ str(item['size'])
            self.listWidget_font.addItem(itemtext)




    def closecam(self):
        self.stop()
        
  
    def drawbox(self):
        self.canvas.canvas_mode = 'pattern'

    def default(self):
        self.canvas.canvas_mode = 'default'

    
    def pushButton_ssimFunction(self):
        aa = self.dst[self.rc[1]:self.rc[1]+self.rc[3],self.rc[0]:self.rc[0]+self.rc[2]]
        # ret, dc = cv2.threshold(aa, 1, 255, cv2.THRESH_BINARY)
        # cv2.imshow('asdasdasdasd',dc)
        
        ret, thresh = cv2.threshold(aa[:, :,2], 1, 255, cv2.THRESH_BINARY)
        # cv2.imshow('thresh',thresh)
        
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(cnt) for cnt in contours]
        top_x = min([x for (x, y, w, h) in rects])
        top_y = min([y for (x, y, w, h) in rects])
        bottom_x = max([x + w for (x, y, w, h) in rects])
        bottom_y = max([y + h for (x, y, w, h) in rects])
        ccc=thresh[top_y:bottom_y, top_x:bottom_x]
        
        #cv2.imshow('ccc',ccc)
        
        
        fo_list = []
        file_list = os.listdir('./font/')
        
       
        for font_file in file_list:

            font_path = '/font/' + font_file
            font_size = 200
            draw_text=self.lineEdit_font.text()
            
            font = ImageFont.truetype(font_path, font_size)
            back_text_width = int(font_size*1.5)
            back_text_height = int(font_size*1.5)
            
            
            canvas = Image.new('RGB', (back_text_width, back_text_height), (0,0,0))
            draw = ImageDraw.Draw(canvas)
            w, h = font.getsize(draw_text)
            draw.text(((back_text_width - w) / 2.0, (back_text_height-h) / 2.0), draw_text, (255,255,255), font)
            # canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
            # canvas = canvas.rotate(rotation, expand=False, fillcolor="black")
            text=np.array(canvas)
            
            ret, thresh = cv2.threshold(text[:, :,2], 1, 255, cv2.THRESH_BINARY)
            # cv2.imshow('thresh',thresh)
            




            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(cnt) for cnt in contours]
            top_x = min([x for (x, y, w, h) in rects])
            top_y = min([y for (x, y, w, h) in rects])
            bottom_x = max([x + w for (x, y, w, h) in rects])
            bottom_y = max([y + h for (x, y, w, h) in rects])
            cccr=thresh[top_y:bottom_y, top_x:bottom_x]

            cccr_resize = cv2.resize(cccr, dsize=(ccc.shape[1], ccc.shape[0]), interpolation=cv2.INTER_AREA)

            # cv2.imshow('cccr',cccr_resize)
            

            

            (score, diff) = ssim(ccc, cccr_resize, full=True)
            diff = (diff * 255).astype("uint8")
            fo_list.append((score,font_file))





            # (score, diff) = ssim(ccc, cccr_resize, full = True)
            # diff = (diff * 255).astype("uint8")
            # print("SSIM: {}".format(score))

            # thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)[1]
            # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cnts = imutils.grab_contours(cnts)

            # cv2.imshow("Original", ccc)
            # cv2.imshow("Modified", cccr_resize)

            # cv2.imshow("Diff", diff)
            # cv2.imshow("Thresh", thresh)

        fo_list.sort(key = lambda x :-x[0])
        # print(fo_list)


        font_path = '/font/' + fo_list[0][1]
        font_size = 200
        draw_text=self.lineEdit_font.text()
        
        font = ImageFont.truetype(font_path, font_size)
        back_text_width = int(font_size*1.5)
        back_text_height = int(font_size*1.5)
        
        canvas = Image.new('RGB', (back_text_width, back_text_height), (0,0,0))
        draw = ImageDraw.Draw(canvas)
        w, h = font.getsize(draw_text)
        draw.text(((back_text_width - w) / 2.0, (back_text_height-h) / 2.0), draw_text, (255,255,255), font)
        # canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
        # canvas = canvas.rotate(rotation, expand=False, fillcolor="black")
        textimg=np.array(canvas)

        ret, thresh = cv2.threshold(textimg[:, :,2], 1, 255, cv2.THRESH_BINARY)       
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(cnt) for cnt in contours]
        top_x = min([x for (x, y, w, h) in rects])
        top_y = min([y for (x, y, w, h) in rects])
        bottom_x = max([x + w for (x, y, w, h) in rects])
        bottom_y = max([y + h for (x, y, w, h) in rects])
        self.textimg=textimg[top_y:bottom_y, top_x:bottom_x]

        self.textimg = cv2.resize(self.textimg, dsize=(ccc.shape[1], ccc.shape[0]), interpolation=cv2.INTER_AREA)
        self.textsize = (ccc.shape[1], ccc.shape[0])

        cam_crop_qt_img = self.convert_cv_qt(self.textimg,80,80)
        self.drawpannel_ssim.setPixmap(cam_crop_qt_img)
        self.label_ssim.setText(fo_list[0][1])
        self.label_ssim_size.setText(str(ccc.shape[1]) + ',' + str(ccc.shape[0]))



        font_path = '/font/' + fo_list[1][1]
        font_size = 200
        draw_text=self.lineEdit_font.text()
        
        font = ImageFont.truetype(font_path, font_size)
        back_text_width = int(font_size*1.5)
        back_text_height = int(font_size*1.5)
        
        canvas = Image.new('RGB', (back_text_width, back_text_height), (0,0,0))
        draw = ImageDraw.Draw(canvas)
        w, h = font.getsize(draw_text)
        draw.text(((back_text_width - w) / 2.0, (back_text_height-h) / 2.0), draw_text, (255,255,255), font)
        # canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
        # canvas = canvas.rotate(rotation, expand=False, fillcolor="black")
        textimg2=np.array(canvas)


        ret, thresh = cv2.threshold(textimg2[:, :,2], 1, 255, cv2.THRESH_BINARY)       
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(cnt) for cnt in contours]
        top_x = min([x for (x, y, w, h) in rects])
        top_y = min([y for (x, y, w, h) in rects])
        bottom_x = max([x + w for (x, y, w, h) in rects])
        bottom_y = max([y + h for (x, y, w, h) in rects])
        self.textimg2=textimg2[top_y:bottom_y, top_x:bottom_x]

        self.textimg2 = cv2.resize(self.textimg2, dsize=(ccc.shape[1], ccc.shape[0]), interpolation=cv2.INTER_AREA)
        self.textsize2 = (ccc.shape[1], ccc.shape[0])


        cam_crop_qt_img = self.convert_cv_qt(self.textimg2,80,80)
        self.drawpannel_ssim_2.setPixmap(cam_crop_qt_img)
        self.label_ssim_2.setText(fo_list[1][1])
        self.label_ssim_size_2.setText(str(ccc.shape[1]) + ',' + str(ccc.shape[0]))










        font_path = '/font/' + fo_list[2][1]
        font_size = 200
        draw_text=self.lineEdit_font.text()
        
        font = ImageFont.truetype(font_path, font_size)
        back_text_width = int(font_size*1.5)
        back_text_height = int(font_size*1.5)
        
        canvas = Image.new('RGB', (back_text_width, back_text_height), (0,0,0))
        draw = ImageDraw.Draw(canvas)
        w, h = font.getsize(draw_text)
        draw.text(((back_text_width - w) / 2.0, (back_text_height-h) / 2.0), draw_text, (255,255,255), font)
        # canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
        # canvas = canvas.rotate(rotation, expand=False, fillcolor="black")
        textimg3=np.array(canvas)

        ret, thresh = cv2.threshold(textimg3[:, :,2], 1, 255, cv2.THRESH_BINARY)       
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(cnt) for cnt in contours]
        top_x = min([x for (x, y, w, h) in rects])
        top_y = min([y for (x, y, w, h) in rects])
        bottom_x = max([x + w for (x, y, w, h) in rects])
        bottom_y = max([y + h for (x, y, w, h) in rects])
        self.textimg3=textimg3[top_y:bottom_y, top_x:bottom_x]
        
        self.textimg3 = cv2.resize(self.textimg3, dsize=(ccc.shape[1], ccc.shape[0]), interpolation=cv2.INTER_AREA)
        self.textsize3 = (ccc.shape[1], ccc.shape[0])
    
        cam_crop_qt_img = self.convert_cv_qt(self.textimg3,80,80)
        self.drawpannel_ssim_3.setPixmap(cam_crop_qt_img)
        self.label_ssim_3.setText(fo_list[2][1])
        self.label_ssim_size_3.setText(str(ccc.shape[1]) + ',' + str(ccc.shape[0]))






















        aa = self.dst[self.rc[1]:self.rc[1]+self.rc[3],self.rc[0]:self.rc[0]+self.rc[2]]
        # ret, dc = cv2.threshold(aa, 1, 255, cv2.THRESH_BINARY)
        # cv2.imshow('asdasdasdasd',dc)
        
        ret, thresh = cv2.threshold(aa[:, :,2], 1, 255, cv2.THRESH_BINARY)
        # cv2.imshow('thresh',thresh)
        
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(cnt) for cnt in contours]
        top_x = min([x for (x, y, w, h) in rects])
        top_y = min([y for (x, y, w, h) in rects])
        bottom_x = max([x + w for (x, y, w, h) in rects])
        bottom_y = max([y + h for (x, y, w, h) in rects])
        ccc=thresh[top_y:bottom_y, top_x:bottom_x]
        
     

        query_hash = self.img2hash(ccc)
        

        
        fo_list = []
        file_list = os.listdir('./font/')
        
       
        for font_file in file_list:

            font_path = '/font/' + font_file
            font_size = 200
            draw_text=self.lineEdit_font.text()
            
            font = ImageFont.truetype(font_path, font_size)
            back_text_width = int(font_size*1.5)
            back_text_height = int(font_size*1.5)
            
            
            canvas = Image.new('RGB', (back_text_width, back_text_height), (0,0,0))
            draw = ImageDraw.Draw(canvas)
            w, h = font.getsize(draw_text)
            draw.text(((back_text_width - w) / 2.0, (back_text_height-h) / 2.0), draw_text, (255,255,255), font)
            # canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
            # canvas = canvas.rotate(rotation, expand=False, fillcolor="black")
            text=np.array(canvas)
            
            ret, thresh = cv2.threshold(text[:, :,2], 1, 255, cv2.THRESH_BINARY)
            # cv2.imshow('thresh',thresh)
            




            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(cnt) for cnt in contours]
            top_x = min([x for (x, y, w, h) in rects])
            top_y = min([y for (x, y, w, h) in rects])
            bottom_x = max([x + w for (x, y, w, h) in rects])
            bottom_y = max([y + h for (x, y, w, h) in rects])
            cccr=thresh[top_y:bottom_y, top_x:bottom_x]

            cccr_resize = cv2.resize(cccr, dsize=(ccc.shape[1], ccc.shape[0]), interpolation=cv2.INTER_AREA)
    
            # cv2.imshow('cccr',cccr_resize)
            

            a_hash = self.img2hash(cccr_resize)
            dst = self.hamming_distance(query_hash, a_hash)
            fo_list.append((dst/256,font_file))


            # (score, diff) = ssim(ccc, cccr_resize, full=True)
            # diff = (diff * 255).astype("uint8")
            # fo_list.append(score)





            # (score, diff) = ssim(ccc, cccr_resize, full = True)
            # diff = (diff * 255).astype("uint8")
            # print("SSIM: {}".format(score))

            # thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)[1]
            # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cnts = imutils.grab_contours(cnts)

            # cv2.imshow("Original", ccc)
            # cv2.imshow("Modified", cccr_resize)

            # cv2.imshow("Diff", diff)
            # cv2.imshow("Thresh", thresh)

        fo_list.sort(key = lambda x :x[0])
        

        font_path = '/font/' + fo_list[0][1]
        font_size = 200
        draw_text=self.lineEdit_font.text()
        
        font = ImageFont.truetype(font_path, font_size)
        back_text_width = int(font_size*1.5)
        back_text_height = int(font_size*1.5)
        
        canvas = Image.new('RGB', (back_text_width, back_text_height), (0,0,0))
        draw = ImageDraw.Draw(canvas)
        w, h = font.getsize(draw_text)
        draw.text(((back_text_width - w) / 2.0, (back_text_height-h) / 2.0), draw_text, (255,255,255), font)
        # canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
        # canvas = canvas.rotate(rotation, expand=False, fillcolor="black")
        textimg4 =np.array(canvas)


        ret, thresh = cv2.threshold(textimg4[:, :,2], 1, 255, cv2.THRESH_BINARY)       
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(cnt) for cnt in contours]
        top_x = min([x for (x, y, w, h) in rects])
        top_y = min([y for (x, y, w, h) in rects])
        bottom_x = max([x + w for (x, y, w, h) in rects])
        bottom_y = max([y + h for (x, y, w, h) in rects])
        self.textimg4=textimg4[top_y:bottom_y, top_x:bottom_x]

        self.textimg4 = cv2.resize(self.textimg4, dsize=(ccc.shape[1], ccc.shape[0]), interpolation=cv2.INTER_AREA)
        self.textsize4 = (ccc.shape[1], ccc.shape[0])


        cam_crop_qt_img = self.convert_cv_qt(self.textimg4,80,80)
        self.drawpannel_ahm.setPixmap(cam_crop_qt_img)
        self.label_ahm.setText(fo_list[0][1])
        self.label_ahm_size.setText(str(ccc.shape[1]) + ',' + str(ccc.shape[0]))



        font_path = '/font/' + fo_list[1][1]
        font_size = 200
        draw_text=self.lineEdit_font.text()
        
        font = ImageFont.truetype(font_path, font_size)
        back_text_width = int(font_size*1.5)
        back_text_height = int(font_size*1.5)
        
        canvas = Image.new('RGB', (back_text_width, back_text_height), (0,0,0))
        draw = ImageDraw.Draw(canvas)
        w, h = font.getsize(draw_text)
        draw.text(((back_text_width - w) / 2.0, (back_text_height-h) / 2.0), draw_text, (255,255,255), font)
        # canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
        # canvas = canvas.rotate(rotation, expand=False, fillcolor="black")
        textimg5=np.array(canvas)


        ret, thresh = cv2.threshold(textimg5[:, :,2], 1, 255, cv2.THRESH_BINARY)       
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(cnt) for cnt in contours]
        top_x = min([x for (x, y, w, h) in rects])
        top_y = min([y for (x, y, w, h) in rects])
        bottom_x = max([x + w for (x, y, w, h) in rects])
        bottom_y = max([y + h for (x, y, w, h) in rects])
        self.textimg5=textimg5[top_y:bottom_y, top_x:bottom_x]

        self.textimg5 = cv2.resize(self.textimg5, dsize=(ccc.shape[1], ccc.shape[0]), interpolation=cv2.INTER_AREA)
        self.textsize5 = (ccc.shape[1], ccc.shape[0])

 
        cam_crop_qt_img = self.convert_cv_qt(self.textimg5,80,80)
        self.drawpannel_ahm_2.setPixmap(cam_crop_qt_img)
        self.label_ahm_2.setText(fo_list[1][1])
        self.label_ahm_size_2.setText(str(ccc.shape[1]) + ',' + str(ccc.shape[0]))




        font_path = '/font/' + fo_list[2][1]
        font_size = 200
        draw_text=self.lineEdit_font.text()
        
        font = ImageFont.truetype(font_path, font_size)
        back_text_width = int(font_size*1.5)
        back_text_height = int(font_size*1.5)
        
        canvas = Image.new('RGB', (back_text_width, back_text_height), (0,0,0))
        draw = ImageDraw.Draw(canvas)
        w, h = font.getsize(draw_text)
        draw.text(((back_text_width - w) / 2.0, (back_text_height-h) / 2.0), draw_text, (255,255,255), font)
        # canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
        # canvas = canvas.rotate(rotation, expand=False, fillcolor="black")
        textimg6=np.array(canvas)


        ret, thresh = cv2.threshold(textimg6[:, :,2], 1, 255, cv2.THRESH_BINARY)       
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(cnt) for cnt in contours]
        top_x = min([x for (x, y, w, h) in rects])
        top_y = min([y for (x, y, w, h) in rects])
        bottom_x = max([x + w for (x, y, w, h) in rects])
        bottom_y = max([y + h for (x, y, w, h) in rects])
        self.textimg6=textimg6[top_y:bottom_y, top_x:bottom_x]

        self.textimg6 = cv2.resize(self.textimg6, dsize=(ccc.shape[1], ccc.shape[0]), interpolation=cv2.INTER_AREA)
        self.textsize6 = (ccc.shape[1], ccc.shape[0])



        self.textimg6 = cv2.resize(self.textimg6, dsize=(ccc.shape[1], ccc.shape[0]), interpolation=cv2.INTER_AREA)

        cam_crop_qt_img = self.convert_cv_qt(self.textimg6,80,80)
        self.drawpannel_ahm_3.setPixmap(cam_crop_qt_img)
        self.label_ahm_3.setText(fo_list[2][1])
        self.label_ahm_size_3.setText(str(ccc.shape[1]) + ',' + str(ccc.shape[0]))



    def pushButton_ahmFunction(self):
        aa = self.dst[self.rc[1]:self.rc[1]+self.rc[3],self.rc[0]:self.rc[0]+self.rc[2]]
        # ret, dc = cv2.threshold(aa, 1, 255, cv2.THRESH_BINARY)
        # cv2.imshow('asdasdasdasd',dc)
        
        ret, thresh = cv2.threshold(aa[:, :,2], 1, 255, cv2.THRESH_BINARY)
        # cv2.imshow('thresh',thresh)
        
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(cnt) for cnt in contours]
        top_x = min([x for (x, y, w, h) in rects])
        top_y = min([y for (x, y, w, h) in rects])
        bottom_x = max([x + w for (x, y, w, h) in rects])
        bottom_y = max([y + h for (x, y, w, h) in rects])
        ccc=thresh[top_y:bottom_y, top_x:bottom_x]
        
        query_hash = self.img2hash(ccc)
        

        
        fo_list = []
        file_list = os.listdir('./font/')
        
       
        for font_file in file_list:

            font_path = '/font/' + font_file
            font_size = 200
            draw_text=self.lineEdit_font.text()
            
            font = ImageFont.truetype(font_path, font_size)
            back_text_width = int(font_size*1.5)
            back_text_height = int(font_size*1.5)
            
            
            canvas = Image.new('RGB', (back_text_width, back_text_height), (0,0,0))
            draw = ImageDraw.Draw(canvas)
            w, h = font.getsize(draw_text)
            draw.text(((back_text_width - w) / 2.0, (back_text_height-h) / 2.0), draw_text, (255,255,255), font)
            # canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
            # canvas = canvas.rotate(rotation, expand=False, fillcolor="black")
            text=np.array(canvas)
            
            ret, thresh = cv2.threshold(text[:, :,2], 1, 255, cv2.THRESH_BINARY)
            # cv2.imshow('thresh',thresh)
            




            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(cnt) for cnt in contours]
            top_x = min([x for (x, y, w, h) in rects])
            top_y = min([y for (x, y, w, h) in rects])
            bottom_x = max([x + w for (x, y, w, h) in rects])
            bottom_y = max([y + h for (x, y, w, h) in rects])
            cccr=thresh[top_y:bottom_y, top_x:bottom_x]

            cccr_resize = cv2.resize(cccr, dsize=(ccc.shape[1], ccc.shape[0]), interpolation=cv2.INTER_AREA)

            # cv2.imshow('cccr',cccr_resize)
            

            a_hash = self.img2hash(cccr_resize)
            dst = self.hamming_distance(query_hash, a_hash)
            fo_list.append((dst/256,font_file))


            # (score, diff) = ssim(ccc, cccr_resize, full=True)
            # diff = (diff * 255).astype("uint8")
            # fo_list.append(score)





            # (score, diff) = ssim(ccc, cccr_resize, full = True)
            # diff = (diff * 255).astype("uint8")
            # print("SSIM: {}".format(score))

            # thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)[1]
            # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cnts = imutils.grab_contours(cnts)

            # cv2.imshow("Original", ccc)
            # cv2.imshow("Modified", cccr_resize)

            # cv2.imshow("Diff", diff)
            # cv2.imshow("Thresh", thresh)

        fo_list.sort(key = lambda x :x[0])
        

        font_path = '/font/' + fo_list[0][1]
        font_size = 200
        draw_text=self.lineEdit_font.text()
        
        font = ImageFont.truetype(font_path, font_size)
        back_text_width = int(font_size*1.5)
        back_text_height = int(font_size*1.5)
        
        canvas = Image.new('RGB', (back_text_width, back_text_height), (0,0,0))
        draw = ImageDraw.Draw(canvas)
        w, h = font.getsize(draw_text)
        draw.text(((back_text_width - w) / 2.0, (back_text_height-h) / 2.0), draw_text, (255,255,255), font)
        # canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
        # canvas = canvas.rotate(rotation, expand=False, fillcolor="black")
        self.textimg4 =np.array(canvas)

        cam_crop_qt_img = self.convert_cv_qt(self.textimg4,80,80)
        self.drawpannel_ahm.setPixmap(cam_crop_qt_img)
        self.label_ahm.setText(fo_list[0][1])




        font_path = '/font/' + fo_list[1][1]
        font_size = 200
        draw_text=self.lineEdit_font.text()
        
        font = ImageFont.truetype(font_path, font_size)
        back_text_width = int(font_size*1.5)
        back_text_height = int(font_size*1.5)
        
        canvas = Image.new('RGB', (back_text_width, back_text_height), (0,0,0))
        draw = ImageDraw.Draw(canvas)
        w, h = font.getsize(draw_text)
        draw.text(((back_text_width - w) / 2.0, (back_text_height-h) / 2.0), draw_text, (255,255,255), font)
        # canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
        # canvas = canvas.rotate(rotation, expand=False, fillcolor="black")
        self.textimg5=np.array(canvas)

        cam_crop_qt_img = self.convert_cv_qt(self.textimg5,80,80)
        self.drawpannel_ahm_2.setPixmap(cam_crop_qt_img)
        self.label_ahm_2.setText(fo_list[1][1])




        font_path = '/font/' + fo_list[2][1]
        font_size = 200
        draw_text=self.lineEdit_font.text()
        
        font = ImageFont.truetype(font_path, font_size)
        back_text_width = int(font_size*1.5)
        back_text_height = int(font_size*1.5)
        
        canvas = Image.new('RGB', (back_text_width, back_text_height), (0,0,0))
        draw = ImageDraw.Draw(canvas)
        w, h = font.getsize(draw_text)
        draw.text(((back_text_width - w) / 2.0, (back_text_height-h) / 2.0), draw_text, (255,255,255), font)
        # canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
        # canvas = canvas.rotate(rotation, expand=False, fillcolor="black")
        self.textimg6=np.array(canvas)

        cam_crop_qt_img = self.convert_cv_qt(self.textimg6,80,80)
        self.drawpannel_ahm_3.setPixmap(cam_crop_qt_img)
        self.label_ahm_3.setText(fo_list[2][1])

        


    # def mousePressEvent(self, event):
    #     if self.crop_img_on == True:
    #         print(event.x(), event.y())


    #         cv2.circle(self.crop_img, (int((event.x()-680)*self.scaleFactor_w), int((event.y()-260)*self.scaleFactor_h)), 1, (255, 255, 0), -1, cv2.LINE_AA)

    #         cam_crop_qt_img = self.convert_cv_qt(self.crop_img)
    #         self.draw_pannel.setPixmap(cam_crop_qt_img)
    #         self.draw_pannel.setScaledContents(True)


    
    def img2hash(self, img):
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(img, (16, 16))
        avg = gray.mean()
        bi = 1 * (gray > avg)
        return bi

    # 해밍거리 측정 함수 ---③
    def hamming_distance(self, a, b):
        a = a.reshape(1,-1)
        b = b.reshape(1,-1)
        # 같은 자리의 값이 서로 다른 것들의 합
        distance = (a !=b).sum()
        return distance


   
if __name__ == "__main__" :
    #QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv) 

    #WindowClass의 인스턴스 생성
    myWindow = WindowClass() 

    #프로그램 화면을 보여주는 코드
    myWindow.show()

    #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()

    myWindow.stop()
    
    print('ddddd')