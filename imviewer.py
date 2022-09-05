import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import Qt
import cv2
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *


form_imageviwer = uic.loadUiType("imviewer.ui")[0] #두 번째창 ui

class imageviwer(QDialog,QWidget,form_imageviwer):
    def __init__(self, img, label,file):
        super(imageviwer,self).__init__()
        self.initUI(img,label,file)
        self.show() # 두번째창 실행

    def initUI(self, img, label,file):
        self.setupUi(self)
        qt_img = self.convert_cv_qt(img,1280,720)
        self.label_screen_2.setPixmap(qt_img)
        # self.label.setText(label)
        self.label_2.setText(file)
        # self.home.clicked.connect(self.Home)
        
    def Home(self):
        self.close() #창 닫기

    def mousePressEvent(self, e):  # e ; QMouseEvent
        # print('BUTTON PRESS')
        self.mouseButtonKind(e.buttons())

    def mouseButtonKind(self, buttons):

        if buttons & Qt.LeftButton:
            self.close() #창 닫기
        if buttons & Qt.MidButton:
            self.close() #창 닫기
        if buttons & Qt.RightButton:
            self.close() #창 닫기



    def convert_cv_qt(self, cv_img, disply_width, display_height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(disply_width, display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)