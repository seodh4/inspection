from typing import Any
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QPainter, QPen, QPixmap, QPalette, QFont
from PyQt5.QtCore import QSize, Qt, QLine, QPoint, pyqtSignal, reset, pyqtSlot
from PyQt5.QtWidgets import*
import cv2
import numpy as np


class Screen(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

    mousePressEvent_signal = pyqtSignal(object)
    mouseMoveEvent_signal = pyqtSignal(object)
    mouseReleaseEvent_signal = pyqtSignal(object)


    # def paintEvent(self, e):
    #     qp = QPainter()
    #     qp.begin(self)
    #     self.draw_line(qp)
    #     qp.end()

    # def draw_line(self, qp):
    #     qp.setPen(QPen(Qt.blue, 8))
    #     qp.drawLine(30, 230, 200, 50)
    #     qp.setPen(QPen(Qt.green, 12))
    #     qp.drawLine(140, 60, 320, 280)
    #     qp.setPen(QPen(Qt.red, 16))
    #     qp.drawLine(330, 250, 40, 190)

    # 마우스 클릭
    def mousePressEvent(self, event):
        self.mousePressEvent_signal.emit(event)

    # 마우스 MOVE
    def mouseMoveEvent(self,event):
        self.mouseMoveEvent_signal.emit(event)
        
    # 마우스 RELEASE
    def mouseReleaseEvent(self,event):
        self.mouseReleaseEvent_signal.emit(event)



class Canvas(QtWidgets.QLabel):
    
    sum_signal = pyqtSignal(list)
    fiducial_signal = pyqtSignal(list)



    def setupUi(self):
        self.screen = Screen(self)
        self.screen.setBackgroundRole(QPalette.Base)
        # self.screen.setScaledContents(True)
        self.screen.setObjectName("screen")
        self.screen.resize(650, 490)

        self.scrollArea = QScrollArea(self)
        # self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.screen)
        # self.scrollArea.setVisible(False)
        self.scrollArea.resize(650, 490)

        self.scrollArea.setVisible(True)
        

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()
        self.setAcceptDrops(True)

        self.disply_width = 640
        self.display_height = 480

        self.scaleFactor = 1.0

        self.state_draw_rec = False

        # 마우스 이벤트 시작값
        self.past_x = None
        self.past_y = None
        # 마우스 이벤트 종료값
        self.present_x = None
        self.present_y = None

        self.py = 0
        self.px = 0

        # canvas mode
        self.canvas_mode = 'pattern'


        self.gt_boxs = {}
        self.selbox = 0

        # create the video capture thread
        # connect its signal to the update_image slot
        # self.parent.change_pixmap_signal.connect(self.update_image)

        self.screen.mousePressEvent_signal.connect(self.s_mousePressEvent)
        self.screen.mouseMoveEvent_signal.connect(self.s_mouseMoveEvent)
        self.screen.mouseReleaseEvent_signal.connect(self.s_mouseReleaseEvent)


    def zoomIn(self):
        self.scaleFactor *= 1.25
        self.scaleImage()
        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), 1.25)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), 1.25)

    def zoomOut(self):
        self.scaleFactor *= 0.8
        self.scaleImage()
        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), 0.8)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), 0.8)

    def scaleImage(self):

        self.rimg=self.qt_img.scaled(self.scaleFactor * self.qt_img.size(),Qt.KeepAspectRatio, Qt.FastTransformation)
        self.screen.resize(self.rimg.size())
        self.screen.setPixmap(self.rimg)
        # self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        # self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value() + ((factor - 1) * scrollBar.pageStep() / 2)))


    # @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        # print(type(cv_img))
        self.qt_img = self.convert_cv_qt(cv_img)
        self.rimg=self.qt_img.scaled(self.scaleFactor * self.qt_img.size(),Qt.KeepAspectRatio, Qt.FastTransformation)
        painter = QPainter(self.rimg)
        # painter.scale(self.scaleFactor,self.scaleFactor)
        self.draw_gt(painter, self.gt_boxs, sel = self.selbox)
        # self.selbox = len(self.gt_boxs)-1
        self.screen.resize(self.rimg.size())
        self.screen.setPixmap(self.rimg)
    
   



    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio, Qt.FastTransformation)
        return QPixmap.fromImage(p)






    # 마우스 클릭
    @pyqtSlot(object)
    def s_mousePressEvent(self, event):
        
        x = event.x() * 2
        y = event.y() * 2

        self.past_x = int(event.x()/self.scaleFactor)
        self.past_y = int(event.y()/self.scaleFactor)

        if self.canvas_mode == 'fiducial':
            self.state_draw_rec = True


        if self.canvas_mode == 'pattern':
            self.state_draw_rec = True

        if self.canvas_mode == 'default':
            for idx, gt_box in enumerate(self.gt_boxs):
                point = gt_box['point']
                x1 = point[0]
                y1 = point[1]
                x2 = point[2]
                y2 = point[3]

                if x >= x1 and x <= x2 and y >= y1 and y <= y2:

                    self.rimg=self.qt_img.scaled(self.scaleFactor * self.qt_img.size(),Qt.KeepAspectRatio, Qt.FastTransformation)
                    p_rimg = self.rimg
                    painter = QPainter(p_rimg)
                    
                    self.draw_gt(painter, self.gt_boxs, sel=idx)

                    self.screen.setPixmap(QPixmap(p_rimg))
                    self.selbox = idx
                    break



    # 마우스 MOVE
    @pyqtSlot(object)
    def s_mouseMoveEvent(self,event):
        self.x =int(event.x()/self.scaleFactor)
        self.y =int(event.y()/self.scaleFactor)
        
        if self.canvas_mode == 'fiducial':
            if abs(self.x - self.px) > 0 or abs(self.y - self.py) > 0:
                # self.scaleImage()
                self.draw_line(self.x*self.scaleFactor,int(self.y*self.scaleFactor))
                if self.state_draw_rec == True:
                    self.draw_rect(int(self.past_x*self.scaleFactor),int(self.past_y*self.scaleFactor),int(self.x*self.scaleFactor),int(self.y*self.scaleFactor))
                self.py=self.y
                self.px=self.x


        if self.canvas_mode == 'pattern':
            if abs(self.x - self.px) > 0 or abs(self.y - self.py) > 0:
                # self.scaleImage()
                self.draw_line(self.x*self.scaleFactor,int(self.y*self.scaleFactor))
                if self.state_draw_rec == True:
                    self.draw_rect(int(self.past_x*self.scaleFactor),int(self.past_y*self.scaleFactor),int(self.x*self.scaleFactor),int(self.y*self.scaleFactor))
                self.py=self.y
                self.px=self.x

        # if self.canvas_mode == 'default':
        #     for idx, gt_box in enumerate(self.gt_boxs):
        #         point = gt_box['point']
        #         x1 = point[0]
        #         y1 = point[1]
        #         x2 = point[2]
        #         y2 = point[3]

        #         if event.x() > x1 and event.x() < x2 and event.y() > y1 and event.y() < y2:

        #             # self.rimg=self.qt_img.scaled(self.scaleFactor * self.qt_img.size(),Qt.KeepAspectRatio, Qt.FastTransformation)
        #             # p_rimg = self.rimg
        #             # painter = QPainter(p_rimg)
                    
        #             # self.draw_gt(painter, self.gt_boxs, hover=idx)

        #             # self.screen.setPixmap(QPixmap(p_rimg))
        #             print(idx)
        #             break




               
            


    # 마우스 RELEASE
    @pyqtSlot(object)
    def s_mouseReleaseEvent(self,event):

        if self.canvas_mode == 'fiducial':
            self.canvas_mode == 'dafault'

            if self.state_draw_rec == True:
                self.state_draw_rec = False
                self.draw_rect(int(self.past_x),int(self.past_y),int(event.x()),int(event.y()))

                # print(self.past_x,self.past_y)
                if int(self.past_x) > int(event.x()/self.scaleFactor):
                    x1 = int(event.x()/self.scaleFactor)
                    x2 = int(self.past_x)
                elif int(self.past_x) == int(event.x()/self.scaleFactor):
                    x1 = int(event.x()/self.scaleFactor)
                    x2 = int(event.x()/self.scaleFactor)+1
                else:
                    x1 = int(self.past_x)
                    x2 = int(event.x()/self.scaleFactor)

                if int(self.past_y) > int(event.y()/self.scaleFactor):
                    y2 = int(self.past_y)
                    y1 = int(event.y()/self.scaleFactor)
                elif int(self.past_y) == int(event.y()/self.scaleFactor):
                    y2 = int(self.past_y) + 1
                    y1 = int(event.y()/self.scaleFactor)
                else:
                    y1 = int(self.past_y)
                    y2 = int(event.y()/self.scaleFactor)

                # rect_point = [int(self.past_x/self.scaleFactor),int(self.past_y/self.scaleFactor),int(event.x()/self.scaleFactor),int(event.y()/self.scaleFactor)]
                rect_point= [x1,y1,x2,y2]
                self.fiducial_signal.emit(rect_point)

                # 마우스 이벤트 시작값 초기화
                self.state_draw_rec = False
                self.past_x = None
                self.past_y = None



        if self.canvas_mode == 'pattern':
            self.canvas_mode == 'dafault'

            if self.state_draw_rec == True:
                self.state_draw_rec = False
                self.draw_rect(int(self.past_x),int(self.past_y),int(event.x()),int(event.y()))

                # print(self.past_x,self.past_y)
                if int(self.past_x) > int(event.x()/self.scaleFactor):
                    x1 = int(event.x()/self.scaleFactor)
                    x2 = int(self.past_x)
                elif int(self.past_x) == int(event.x()/self.scaleFactor):
                    x1 = int(event.x()/self.scaleFactor)
                    x2 = int(event.x()/self.scaleFactor)+1
                else:
                    x1 = int(self.past_x)
                    x2 = int(event.x()/self.scaleFactor)

                if int(self.past_y) > int(event.y()/self.scaleFactor):
                    y2 = int(self.past_y)
                    y1 = int(event.y()/self.scaleFactor)
                elif int(self.past_y) == int(event.y()/self.scaleFactor):
                    y2 = int(self.past_y) + 1
                    y1 = int(event.y()/self.scaleFactor)
                else:
                    y1 = int(self.past_y)
                    y2 = int(event.y()/self.scaleFactor)

                # rect_point = [int(self.past_x/self.scaleFactor),int(self.past_y/self.scaleFactor),int(event.x()/self.scaleFactor),int(event.y()/self.scaleFactor)]
                rect_point= [x1,y1,x2,y2]
                self.sum_signal.emit(rect_point)

                # 마우스 이벤트 시작값 초기화
                self.state_draw_rec = False
                self.past_x = None
                self.past_y = None
            
                
                

   
    def draw_line(self, x, y):

        
        self.rimg=self.qt_img.scaled(self.scaleFactor * self.qt_img.size(),Qt.KeepAspectRatio, Qt.FastTransformation)
        p_rimg=self.rimg
        painter = QPainter(p_rimg)
        # painter.scale(self.scaleFactor,self.scaleFactor)

        self.draw_gt(painter, self.gt_boxs)

            
        # painter.setRenderHint(QPainter.Antialiasing)
        # painter.setRenderHint(QPainter.HighQualityAntialiasing)
        # painter.setRenderHint(QPainter.SmoothPixmapTransform)

        painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
        # print(self.rimg.size().x())
        painter.drawLine(0,int(y),p_rimg.width(),int(y))
        painter.drawLine(int(x),0,int(x),p_rimg.height())
        self.screen.setPixmap(QPixmap(p_rimg))

    def draw_rect(self,x1,y1,x2,y2):
        self.rimg=self.qt_img.scaled(self.scaleFactor * self.qt_img.size(),Qt.KeepAspectRatio, Qt.FastTransformation)
        p_rimg = self.rimg
        painter = QPainter(p_rimg)
        

        self.draw_gt(painter, self.gt_boxs)


        # painter.scale(self.scaleFactor,self.scaleFactor)

        # painter.setRenderHint(QPainter.Antialiasing)
        # painter.setRenderHint(QPainter.HighQualityAntialiasing)
        # painter.setRenderHint(QPainter.SmoothPixmapTransform)

        painter.setPen(QPen(Qt.yellow, 2, Qt.SolidLine))
        painter.drawRect(x1,y1,x2-x1,y2-y1)
        self.screen.setPixmap(QPixmap(p_rimg))


    def draw_gt(self,painter,gt_boxs, sel=None):
        
        for idx, gt_box in enumerate(gt_boxs):

            point = gt_box['point']
            x1 = int(point[0]/2)
            y1 = int(point[1]/2)
            x2 = int(point[2]/2)
            y2 = int(point[3]/2)
            label = gt_box['label']

            # painter.scale(self.scaleFactor,self.scaleFactor)

            # painter.setRenderHint(QPainter.Antialiasing)
            # painter.setRenderHint(QPainter.HighQualityAntialiasing)
            # painter.setRenderHint(QPainter.SmoothPixmapTransform)


            if idx == sel:
                painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            else:
                painter.setPen(QPen(Qt.yellow, 2, Qt.SolidLine))


            painter.drawRect(x1,y1,x2-x1,y2-y1)

            painter.setFont(QFont('Aria', 10))
            painter.drawText(x1, y1-3, label)
           


        



