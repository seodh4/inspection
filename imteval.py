# from re import I
# from shutil import register_unpack_format
import sys
from typing import overload
from PyQt5 import uic
from matplotlib.font_manager import json_dump
from eval import *
import numpy as np
import cv2
from datetime import datetime
import pandas as pd

import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns


from PyQt5.QtCore import pyqtSlot, Qt 
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtGui

from tqdm import tqdm


from PyQt5.QtCore import QThread, pyqtSignal
import time


import mplcursors
from matplotlib.backend_bases import MouseButton
import imviewer

from canvas import Canvas




# eval thread
class External(QThread):
    """
    Runs a counter thread.
    """
    countChanged = pyqtSignal(int,int)
    outevalresult = pyqtSignal(dict)
    outevalsummary = pyqtSignal(dict)
    errorout = pyqtSignal(str)

    def __init__(self, parent): 
        super().__init__(parent) 
        self.parent = parent 
       
    def run(self):
        self.eval()

    def eval(self):
        
        self.parent.label_9.setText('loading...')
        iou_th = self.parent.doubleSpinBox_iou.value()
        iou_th=round(iou_th,3)
        gt_path = self.parent.gt_path
        predict_path = self.parent.predict_path
        class_label= []
        data = {}
        total_len_tp = 0
        total_len_gt = 0
        total_len_pr = 0

        fileio = Fileio()
        eval = Eval()
 
        gtbox_list = []
        detbox_list = []

        try:
            gtfile_list =os.listdir(self.parent.gt_path)
        except:
            self.errorout.emit('error: gt path')
            self.parent.pushButton_evaluation.setEnabled(True)
            return

        gtfile_list = [file for file in gtfile_list if file.endswith(".json")]

        if len(gtfile_list) == 0:
            self.errorout.emit("error: 0 gt file")
            self.parent.pushButton_evaluation.setEnabled(True)
            self.parent.label_9.setText('')
            return

        for idx, gtfile in enumerate(gtfile_list):
            detfile = gtfile

            try:
                gtbox_list = fileio.read_file(self.parent.gt_path+'/'+ gtfile, gtfile)
                detbox_list = fileio.read_file(self.parent.predict_path+'/'+ detfile, gtfile)
            except:
                self.errorout.emit("error: read file")
                self.parent.pushButton_evaluation.setEnabled(True)
                self.parent.label_9.setText('')
                return

            
            # eval.evaluation(gtbox_list, detbox_list, iou_th)

            evallist, fp_list, fn_list , score, label= eval.evaluation(gtbox_list, detbox_list, iou_th)


            len_tp = 0
            len_false = 0
            len_overlab = 0
            len_around = 0

            for li in evallist:
                len_tp += len(li["TP"])
                len_false += len(li["FALSE"])
                len_overlab += len(li["overlab"])
                len_around += len(li["around"])
        
            
            total_len_gt += len(evallist)
            total_len_pr += len(fp_list) + len_overlab + len_around + len_false + len_tp
            total_len_tp += len_tp
    
        
            data[gtfile] = {
                "resultbox" : evallist, 
                "eval_score" : score,
                "fp_list" : fp_list,
                "fn_list" : fn_list
            }

            class_label += label
            self.countChanged.emit(idx,len(gtfile_list))


        e2e_precision = total_len_tp/total_len_pr
        e2e_recall =  total_len_tp/total_len_gt

        class_label_set = set(class_label)
        class_label = list(class_label_set)


        e2e_Hmean = round(2 * (e2e_precision * e2e_recall) / (e2e_precision + e2e_recall),4)

        result_data = {}
        now = datetime.now()
        result_data["project_name"] = self.parent.lineEdit_evalname.text()
        result_data["date"] = str(now)
        result_data["class_label"] = class_label
        result_data["gt_path"] = gt_path
        result_data["predict_path"] = predict_path
        result_data["num"] = len(gtfile_list)
        result_data["IOU"] = iou_th
        result_data["result"] = data
        result_data["summary"] = {
            "e2e_precision" : e2e_precision,
            "e2e_recall": e2e_recall,
            "e2e_Hmean" : e2e_Hmean
        }

        self.outevalsummary.emit(result_data)


        filename = self.parent.lineEdit_evalname.text() + '.json'
        json_dump(result_data,filename)
        # self.parent.print_result(result_data, class_label)
        self.parent.pushButton_evaluation.setEnabled(True)
        self.outevalresult.emit(result_data)














form_class = uic.loadUiType("windows.ui")[0]

class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        self.pushButton_evaluation.clicked.connect(self.pushButton_evaluation_fuction)
        self.listWidget_filelist.itemDoubleClicked.connect(self.listWidget_filelist_DoubleClicked)
        self.listWidget_filelist.currentItemChanged.connect(self.listWidget_filelist_currentItemChanged)
        self.tableWidget_filelist.doubleClicked.connect(self.tableWidget_filelist_doubleClicked)
        self.tableWidget_gt.doubleClicked.connect(self.tableWidget_gt_doubleClicked)

        self.toolButton_gtpath.clicked.connect(self.open_gtpath)
        self.toolButton_detpath.clicked.connect(self.open_detpath)
        self.toolButton_resultfile.clicked.connect(self.open_resultfile)
        self.pushButton_operresultfile.clicked.connect(self.oper_resultfile)

        self.canvas.sum_signal.connect(self.process_sum)
        self.canvas.rst.connect(self.rst_image)



        self.gt_path = './gt/'
        self.predict_path = './result/'


        self.current_filenum = 0
        self.past_filenum = 0


        self.imgfilelist = []
        self.listw_error = []



    def listWidget_filelist_currentItemChanged(self):
        self.current_filenum = self.listWidget_filelist.currentRow()

        if self.current_filenum == -1:
            pass
        else:
            self.open_image(self.current_filenum)


    #     self.shortcut_up = QShortcut(QKeySequence('Up'), self)
    #     self.shortcut_up.activated.connect(self.shortcut_up_press)


    #     self.shortcut_up = QShortcut(QKeySequence('Down'), self)
    #     self.shortcut_up.activated.connect(self.shortcut_down_press)


    # def shortcut_down_press(self):
    #     print('up')
    #     if len(self.imgfilelist) - 1 > self.current_filenum:
    #         self.past_filenum = self.current_filenum
    #         self.current_filenum += 1
    #         self.open_image(self.current_filenum)




    # def shortcut_up_press(self):
    #     print('down')
    #     if 0 < self.current_filenum:
    #         self.past_filenum = self.current_filenum
    #         self.current_filenum -= 1
    #         self.open_image(self.current_filenum)

    @pyqtSlot()
    def rst_image(self):
        self.current_filenum
        self.open_image(self.current_filenum)

    @pyqtSlot(list)
    def process_sum(self, a):
        # self.lineEdit.setText(str(a))
        # self.crop_img = self.img[a[1]:a[3],a[0]:a[2]]
        self.canvas.scaleFactor = 1
        self.canvas.mouseon = False

        if os.path.isfile(self.gt_path + '/' + self.imgfilelist[self.current_filenum][:-5]+'.jpg'):
            self.current_img = cv2.imread(self.gt_path + '/' + self.imgfilelist[self.current_filenum][:-5]+'.jpg')

            # result_data = self.result_datas[current_file]
            self.current_img=self.gen_result_img(self.current_img,self.imgfilelist[self.current_filenum],2,True)

            img=self.current_img[a[1]:a[3],a[0]:a[2]]
            # cv2.imwrite("./img/"+ file[:-5]+'.jpg'    ,img)

            # qt_img=self.convert_cv_qt(self.current_img,1280,720)
            # self.label_screen.setPixmap(qt_img)
            self.canvas.update_image(img)

            self.canvas.canvas_mode = 'pattern'

        else:
            QMessageBox.information(self,'error','gt file error1')



    
    def open_gtpath(self):
        self.gt_path=QFileDialog.getExistingDirectory(self,"Choose GT Directory","./")
        self.lineEdit_gtpath.setText(self.gt_path)
                    # QFileDialog.getOpenFileName(self,"Choose gtFile","./")
    def open_detpath(self):
        self.predict_path=QFileDialog.getExistingDirectory(self,"Choose prediction Directory","./")
        self.lineEdit_detpath.setText(self.predict_path)

    def open_resultfile(self):
        self.resultfile = QFileDialog.getOpenFileName(self,"Choose Result File","./")
        self.lineEdit_resultfile.setText(self.resultfile[0])
    
    def oper_resultfile(self):
        try:
            with open(self.resultfile[0], 'r') as f:
                result_data = json.load(f)
        except:
            QMessageBox.information(self,'error','json file error')
            return

  
        self.print_result(result_data)
        self.outevalsummary(result_data)


    def tableWidget_gt_doubleClicked(self):
        

        row = self.tableWidget_gt.currentIndex().row()
        column = self.tableWidget_gt.currentIndex().column()

        file=self.tableWidget_gt.item(row, 4).text()
        gt_label=self.tableWidget_gt.item(row, 0).text()
        pr_label=self.tableWidget_gt.item(row, 1).text()
        error=self.tableWidget_gt.item(row, 2).text()
        
        cordinate=[]
        cordinate_text=self.tableWidget_gt.item(row, 3).text()
        cordinate_text= cordinate_text.strip("]""[")
        cordinate_splits = cordinate_text.split(',')
        for cordinate_split in cordinate_splits:
            cordinate.append(int(cordinate_split))

        
        img = cv2.imread(self.gt_path + '/' + file[:-5]+'.jpg')
        # result_data = self.result_datas[current_file]
        img=self.gen_result_img(img,file,1)
        # cv2.rectangle(img, (cordinate[0],cordinate[1]), (cordinate[2],cordinate[3]), (0, 255, 255), 5)

        x = cordinate[0] + int((cordinate[2]-cordinate[0])/2)
        y = cordinate[1] + int((cordinate[3]-cordinate[1])/2)

        a = int(max(cordinate[2]-cordinate[0], cordinate[3]-cordinate[1]) * 1.3)

        cv2.circle(img, (x, y), a, (0,255,255), 5)

        self.imageviwer = imviewer.imageviwer(img, gt_label, file)



    def tableWidget_filelist_doubleClicked(self):
        row = self.tableWidget_filelist.currentIndex().row()
        column = self.tableWidget_filelist.currentIndex().column()
      
        gt_label=self.tableWidget_filelist.item(row, 2).text()
        
        cordinate=[]
        cordinate_text=self.tableWidget_filelist.item(row, 3).text()
        cordinate_text= cordinate_text.strip("]""[")
        cordinate_splits = cordinate_text.split(',')
        for cordinate_split in cordinate_splits:
            cordinate.append(int(cordinate_split))

        draw_img=self.current_img.copy()

        x = cordinate[0] + int((cordinate[2]-cordinate[0])/2)
        y = cordinate[1] + int((cordinate[3]-cordinate[1])/2)

        a = int(max(cordinate[2]-cordinate[0], cordinate[3]-cordinate[1]) * 1.3)

        cv2.circle(draw_img, (x, y), a, (0,255,255), 5)

        # cv2.rectangle(draw_img, (cordinate[0],cordinate[1]), (cordinate[2],cordinate[3]), (0, 255, 255), 5)
        cv2.putText(draw_img, gt_label, (cordinate[0],cordinate[1]), cv2.FONT_HERSHEY_PLAIN , 2, (0, 255, 255),2)
        # cv2.imwrite("./img/"+ file[:-5]+'.jpg' ,img)
        # qt_img=self.convert_cv_qt(draw_img,1280,720)
        # self.label_screen.setPixmap(qt_img)
        self.canvas.update_image(draw_img)



    def gen_result_img(self, img , current_file,thick,filelist_state=False):

        self.tableWidget_filelist.setRowCount(0)

        for i in self.result_datas[current_file]['resultbox']:

            if len(i['TP']) > 0: # 맞은거면
                img = self.pain_bbox(img, i['GT'], (0, 60, 20), -1) # gt

                if len(i['around']) > 0:
                    if filelist_state:
                        for j in i['around']:
                            self.tableWidget_filelist.insertRow(self.tableWidget_filelist.rowCount())
                            self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 0, QTableWidgetItem(str(i['GT'][0])))
                            self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 1, QTableWidgetItem(str(j[0])))
                            around_item=QTableWidgetItem('around')
                            around_item.setForeground(QBrush(QColor(255, 127, 0)))
                            self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 2, around_item)
                            self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 3, QTableWidgetItem(str(j[1:5])))
                            self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 4, QTableWidgetItem(str(j[5])))

            else: # 틀린거
                img = self.pain_bbox(img, i['GT'], (20, 0, 120), -1) # gt

                if len(i['FALSE']) > 0:
                    if filelist_state:
                        self.tableWidget_filelist.insertRow(self.tableWidget_filelist.rowCount())
                        self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 0, QTableWidgetItem(str(i['GT'][0])))
                        self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 1, QTableWidgetItem(str(i['FALSE'][0][0])))
                        false_item=QTableWidgetItem('False')
                        false_item.setForeground(QBrush(QColor(255, 0, 0)))
                        self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 2, false_item)
                        self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 3, QTableWidgetItem(str(i['FALSE'][0][1:5])))
                        self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 4, QTableWidgetItem(str(i['FALSE'][0][5])))



                if len(i['overlab']) > 0:
                    if filelist_state:
                        for j in i['overlab']:
                            self.tableWidget_filelist.insertRow(self.tableWidget_filelist.rowCount())
                            self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 0, QTableWidgetItem(str(i['GT'][0])))
                            self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 1, QTableWidgetItem(str(j[0])))
                            overlab_item=QTableWidgetItem('overlab')
                            overlab_item.setForeground(QBrush(QColor(255, 127, 0)))
                            self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 2, overlab_item)
                            self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 3, QTableWidgetItem(str(j[1:5])))
                            self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 4, QTableWidgetItem(str(j[5])))
                


            
            for bbox in i['TP']:
                cv2.rectangle(img, (bbox[1],bbox[2]), (bbox[3],bbox[4]), (0, 255, 0), thick)
                cv2.putText(img, bbox[0], (bbox[1],bbox[2]), cv2.FONT_HERSHEY_PLAIN , 1, (0, 255, 0),1)
            
            for bbox in i['overlab']:
                cv2.rectangle(img, (bbox[1],bbox[2]), (bbox[3],bbox[4]), (255, 0, 0), thick)
                cv2.putText(img, bbox[0], (bbox[1],bbox[2]), cv2.FONT_HERSHEY_PLAIN , 1, (255, 0, 0),1)
            
            for bbox in i['around']:
                cv2.rectangle(img, (bbox[1],bbox[2]), (bbox[3],bbox[4]), (0, 127, 255), thick)
                cv2.putText(img, bbox[0], (bbox[1],bbox[2]), cv2.FONT_HERSHEY_PLAIN , 1, (0, 127, 255),1)
            
            for bbox in i['FALSE']:
                cv2.rectangle(img, (bbox[1],bbox[2]), (bbox[3],bbox[4]), (0, 0, 255), thick)
                cv2.putText(img, bbox[0], (bbox[1],bbox[2]), cv2.FONT_HERSHEY_PLAIN , 1, (0, 0, 255),1)



            
        for i in self.result_datas[current_file]['fn_list']:
            img = self.pain_bbox(img, i, (20, 0, 120), -1) # fn_list
            if filelist_state:
                self.tableWidget_filelist.insertRow(self.tableWidget_filelist.rowCount())
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 0, QTableWidgetItem(str(i[0])))
                nan_item=QTableWidgetItem('NaN')
                nan_item.setForeground(QBrush(QColor(128, 128, 128)))
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 1, nan_item)
                fn_item=QTableWidgetItem('FN')
                fn_item.setForeground(QBrush(QColor(255, 0, 0)))
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 2, fn_item)
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 3, QTableWidgetItem(str(i[1:5])))
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 4, QTableWidgetItem(str(i[5])))



        for i in self.result_datas[current_file]['fp_list']:
            cv2.rectangle(img, (i[1],i[2]), (i[3],i[4]), (0, 0, 255), thick)
            cv2.putText(img, i[0], (i[1],i[2]), cv2.FONT_HERSHEY_PLAIN , 1, (0, 0, 255),1)
            if filelist_state:

                nan_item=QTableWidgetItem('NaN')
                nan_item.setForeground(QBrush(QColor(128, 128, 128)))
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 0, nan_item)

                self.tableWidget_filelist.insertRow(self.tableWidget_filelist.rowCount())
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 1, QTableWidgetItem(str(i[0])))
                fp_item=QTableWidgetItem('FP')
                fp_item.setForeground(QBrush(QColor(255, 0, 0)))
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 2, fp_item)
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 3, QTableWidgetItem(str(i[1:5])))
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 4, QTableWidgetItem(str(i[5])))





        table = self.tableWidget_filelist
        header = table.horizontalHeader()
        twidth = header.width()
        width = []
        for column in range(header.count()):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
            width.append(header.sectionSize(column))

        wfactor = twidth / sum(width)
        for column in range(header.count()):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
            header.resizeSection(column, width[column]*wfactor)



        if filelist_state:
            self.textEdit_file_result.clear()
            e2e_precision = str(self.result_datas[current_file]['eval_score']['e2e_precision'])
            e2e_recall = str(self.result_datas[current_file]['eval_score']['e2e_recall'])

            # self.textEdit_file_result.append("img: " + current_file[:-5])
            # self.textEdit_file_result.append("det_precision: " + det_precision)
            # self.textEdit_file_result.append("det_recall: " + det_recall)
            # self.textEdit_file_result.append("e2e_precision: " + e2e_precision)
            # self.textEdit_file_result.append("e2e_recall: " + e2e_recall)


            self.tableWidget_file_result.setRowCount(0)

            self.tableWidget_file_result.insertRow(0)
            self.tableWidget_file_result.setItem(0, 0, QTableWidgetItem('img'))
            qimg=QTableWidgetItem(current_file[:-5])
            self.tableWidget_file_result.setItem(0, 1, qimg)

            self.tableWidget_file_result.insertRow(1)
            self.tableWidget_file_result.setItem(1, 0, QTableWidgetItem('e2e_precision'))
            e2e_precision=QTableWidgetItem(str(e2e_precision))
            self.tableWidget_file_result.setItem(1, 1, e2e_precision)

            self.tableWidget_file_result.insertRow(2)
            self.tableWidget_file_result.setItem(2, 0, QTableWidgetItem('e2e_recall'))
            e2e_recall=QTableWidgetItem(str(e2e_recall))
            self.tableWidget_file_result.setItem(2, 1, e2e_recall)



        return img






    def listWidget_filelist_DoubleClicked(self):

        self.past_filenum = self.current_filenum
        self.current_filenum = self.listWidget_filelist.currentRow()
        self.open_image(self.current_filenum)





    def open_image(self, filenum):

        # self.imgfilelist[filenum]

        # self.listWidget_filelist.item(self.current_filenum).setBackground(QtGui.QColor(100,100,150))

        # if self.past_filenum in self.listw_error:
        #     self.listWidget_filelist.item(self.past_filenum).setBackground(QtGui.QColor(255, 128, 128))
        # else:
        #     self.listWidget_filelist.item(self.past_filenum).setBackground(QtGui.QColor(255, 255, 255))


        self.canvas.scaleFactor = 1
        self.canvas.mouseon = True


        print(filenum)
        print(len(self.imgfilelist))
        print(self.gt_path + '/' + self.imgfilelist[filenum][:-5]+'.jpg')


        if os.path.isfile(self.gt_path + '/' + self.imgfilelist[filenum][:-5]+'.jpg'):
            self.current_img = cv2.imread(self.gt_path + '/' + self.imgfilelist[filenum][:-5]+'.jpg')

            # result_data = self.result_datas[current_file]
            self.current_img=self.gen_result_img(self.current_img,self.imgfilelist[filenum],2,True)

            # cv2.imwrite("./img/"+ file[:-5]+'.jpg'    ,img)

            # qt_img=self.convert_cv_qt(self.current_img,1280,720)
            # self.label_screen.setPixmap(qt_img)
            self.canvas.update_image(self.current_img)

            self.canvas.canvas_mode = 'pattern'

        else:
            QMessageBox.information(self,'error','gt file error2')





    def print_result(self, data):
        
        self.current_filenum = 0
        self.imgfilelist = []

        self.tableWidget_gt.setRowCount(0)
        self.listWidget_filelist.clear()

        # self.textEdit_summary.clear()
        # pj_name=data["project_name"] 
        # iou=data["IOU"] 

        try:
            self.gt_path=data["gt_path"] 
            self.predict_path=data["predict_path"] 
            self.result_datas=data["result"]
            class_label = data['class_label']
        except:
            QMessageBox.information(self,'error','json file')
            return


        df = pd.DataFrame(columns = ['gt', 'pr', 'er', 'cord', 'file'])


        # print(self.result_datas)
        for idx, result_file_data in enumerate(self.result_datas):
            file = result_file_data

            result_data = self.result_datas[file]
            result_data["resultbox"]

            sw = 0

            for i in result_data["resultbox"]:


                if len(i['TP']) > 0: # 맞은거면
                    temp = { 'gt': i['GT'][0],
                            'pr': i['TP'][0][0],
                            'er': 'TP',
                            'cord': i['GT'][1:5],
                            'file': i['GT'][5]} 

                    tempdf = pd.DataFrame([temp])
                    df = pd.concat([df,tempdf],ignore_index=True)


                    if len(i['around']) > 0:
                        for j in i['around']:
                            temp = { 'gt': i['GT'][0],
                            'pr': j[0],
                            'er': 'around',
                            'cord': i['GT'][1:5],
                            'file': i['GT'][5]} 

                            tempdf = pd.DataFrame([temp])
                            df = pd.concat([df,tempdf],ignore_index=True)

                 
                else: # 틀린거

                    sw = sw + 1

                    if len(i['FALSE']) > 0:
                        temp = { 'gt': i['GT'][0],
                            'pr': i['FALSE'][0][0],
                            'er': 'FALSE',
                            'cord': i['GT'][1:5],
                            'file': i['GT'][5]} 

                        tempdf = pd.DataFrame([temp])
                        df = pd.concat([df,tempdf],ignore_index=True)

                    if len(i['overlab']) > 0:
                        for j in i['overlab']:
                            temp = { 'gt': i['GT'][0],
                            'pr': j[0],
                            'er': 'overlab',
                            'cord': i['GT'][1:5],
                            'file': i['GT'][5]} 

                            tempdf = pd.DataFrame([temp])
                            df = pd.concat([df,tempdf],ignore_index=True)


                    if len(i['around']) > 0:
                        for j in i['around']:
                            temp = { 'gt': i['GT'][0],
                            'pr': j[0],
                            'er': 'around',
                            'cord': i['GT'][1:5],
                            'file': i['GT'][5]} 

                            tempdf = pd.DataFrame([temp])
                            df = pd.concat([df,tempdf],ignore_index=True)

            if sw != 0:
                self.listw_error.append(idx)
                file_q = QListWidgetItem('%s' % file)
                file_q.setBackground(QColor(255, 128, 128))
                self.listWidget_filelist.addItem(file_q)
            else:
                file_q = QListWidgetItem('%s' % file)
                file_q.setBackground(QColor(255, 255, 255))
                self.listWidget_filelist.addItem(file_q)
                
            self.imgfilelist.append(file)

        # tp_heatmap_df = df.loc[df.gt_predict != 'gt']





        # gt_label_df_False = gt_label_df.loc[gt_label_df.true == 'False']


        # det_label_df = df.loc[df.gt_predict == 'predict']
        # det_label_df_False =  det_label_df.loc[det_label_df.box != 'OVER']
    
        
        # if len(df) > 0:
        #     self.fig = plt.figure()
        #     self.canvas = FigureCanvas(self.fig)
        #     df = df.sort_values(by='gt' ,ascending=True)
        #     plot3 = sns.countplot(data=df, y='gt', hue ='gt_predict', palette='Set2')
        #     plot3.set_title("GT / Prediction")
        #     self.clearlayout(self.gridLayout_gtpredict)
        #     self.gridLayout_gtpredict.addWidget(self.canvas)
        #     self.canvas.draw()

        # df2=pd.concat([det_label_df_False,gt_label_df])
        # df3=df2.loc[df2.box != 'TP'].loc[df2.box != 'GT']
        

        if len(df) > 0:
            # self.fig2 = plt.figure()

            self.fig2 = plt.figure()
            self.fig2.clf()
            self.canvas2 = FigureCanvas(self.fig2)
            df = df.sort_values(by='gt' ,ascending=True)
            plot2 = sns.countplot(data=df, y='gt', hue ='er', palette='Set3')
            plot2.set_title("Error Type : Detection")


            for p in plot2.patches:
                height = p.get_height()
                width= p.get_width()
                if width > 0:
                    plot2.text(p.get_x()+width, p.get_y()+(height/2), int(width), ha = 'center',va = 'center', size = 8, color = 'black')

            # self.clearlayout(self.gridLayout_errortype)
            self.gridLayout_errortype.addWidget(self.canvas2)
            self.canvas2.draw()


            # cursor = mplcursors.cursor(hover=mplcursors.HoverMode.Transient)
            # @cursor.connect("add")
            # def on_add(sel):
            #     x, y, width, height = sel.artist[sel.index].get_bbox().bounds
            #     sel.annotation.set(text=f"{x+width}: {self.gt_label[sel.index]}",position=(0, 20), anncoords="offset points")
            #     sel.annotation.xy = (x + width, y + height / 2)
            
            mplcursors.cursor().connect("add", lambda sel: self.on_click(sel)) 


        # if len(gt_det_label_df) > 0:
        #     self.fig3 = plt.figure()
        #     self.canvas3 = FigureCanvas(self.fig3)
        #     gt_det_label_df = gt_det_label_df.sort_values(by='gt_label' ,ascending=True)
        #     # sns.stripplot(data=gt_det_label_df, y='gt_label', hue ='iou', palette='Set2')
        #     plot5 = sns.stripplot(data=gt_det_label_df, x='iou', y='gt_label')
        #     plot5.set_title("IOU : Detection")
        #     self.clearlayout(self.gridLayout_iou)
        #     self.gridLayout_iou.addWidget(self.canvas3)
        #     self.canvas3.draw()


        # if len(gt_label_df_False) > 0:
        #     self.fig4 = plt.figure()
        #     self.canvas4 = FigureCanvas(self.fig4)
        #     gt_label_df_False = gt_label_df_False.sort_values(by='label' ,ascending=True)
        #     plot6 = sns.countplot(data=gt_label_df_False, y='label', palette='Set3')
        #     plot6.set_title("Recognition False")

        #     for p in plot6.patches:
        #         height = p.get_height()
        #         width= p.get_width()
        #         plot6.text(p.get_x()+width, p.get_y()+(height/2), width, ha = 'center',va = 'center', size = 8, color = 'black')

        #     self.clearlayout(self.gridLayout_recognitionfalse)
        #     self.gridLayout_recognitionfalse.addWidget(self.canvas4)
        #     self.canvas4.draw()



        class_label.sort()
        tp_heatmap_df = pd.DataFrame(0, index=class_label, columns=class_label)
        for idx,row in df.iterrows():
            tp_heatmap_df.at[row['pr'],row['gt']] += 1

        
        # tp_heatmap_df=df.groupby(['pr','gt']).size().reset_index(name='cnt')

        # tp_heatmap_df = tp_heatmap_df.pivot('pr','gt','cnt')

        # tp_heatmap_df = tp_heatmap_df.fillna(0)
        tp_heatmap_df = tp_heatmap_df.astype('int')

        # print(tp_heatmap_df)

        # for idx,i in enumerate(tp_heatmap_df):
        #     sum = tp_heatmap_df[i].sum()
        #     tp_heatmap_df[i] = tp_heatmap_df[i] * (1/sum)

        # # tp_heatmap_df = tp_heatmap_df.astype('int')

        # print(tp_heatmap_df)

        self.fig5 = plt.figure()
        self.canvas5 = FigureCanvas(self.fig5)
        

        plt.xlabel('GT', fontsize=14)
        plt.ylabel('PR', fontsize=14)

        plot7= sns.heatmap(data=tp_heatmap_df,  square=True, cmap='Blues', annot = True, fmt="d", cbar=True,linewidths=0.5,annot_kws={"size": 8})
        plot7.set_title("Recognition Heatmap")
        self.clearlayout(self.gridLayout_heatmap)
        self.gridLayout_heatmap.addWidget(self.canvas5)
        self.canvas5.draw()


        self.progressBar_eval.setValue(int(100))
        self.progressBar_eval.setValue(0)
        self.label_9.setText('')


        gt_label = []
        self.eval_df =df
        for gt in self.eval_df['gt']:
            gt_label.append(gt)

        gt_label_set = set(gt_label)
        self.gt_label = list(gt_label_set)
        self.gt_label.sort()



    def on_click(self, sel):
        index=sel.index
        tempdf = self.eval_df.loc[self.eval_df['gt'] == self.gt_label[index]]
        tempdf = tempdf.loc[tempdf['er'] != 'TP']
        self.tableWidget_gt.setRowCount(0)

        for index,df in tempdf.iterrows():
            self.tableWidget_gt.insertRow(self.tableWidget_gt.rowCount())
            self.tableWidget_gt.setItem(self.tableWidget_gt.rowCount()-1, 0, QTableWidgetItem(str(df['gt'])))
            self.tableWidget_gt.setItem(self.tableWidget_gt.rowCount()-1, 1, QTableWidgetItem(str(df['pr'])))
            # around_item=QTableWidgetItem('around')
            # around_item.setForeground(QBrush(QColor(255, 127, 0)))
            # self.tableWidget_gt.setItem(self.tableWidget_gt.rowCount()-1, 2, around_item)
            self.tableWidget_gt.setItem(self.tableWidget_gt.rowCount()-1, 2, QTableWidgetItem(str(df['er'])))
            self.tableWidget_gt.setItem(self.tableWidget_gt.rowCount()-1, 3, QTableWidgetItem(str(df['cord'])))
            self.tableWidget_gt.setItem(self.tableWidget_gt.rowCount()-1, 4, QTableWidgetItem(str(df['file'])))


        table = self.tableWidget_gt
        header = table.horizontalHeader()
        twidth = header.width()
        width = []
        for column in range(header.count()):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
            width.append(header.sectionSize(column))

        wfactor = twidth / sum(width)
        for column in range(header.count()):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
            header.resizeSection(column, width[column]*wfactor)




    def clearlayout(self,layout):
        for i in reversed(range(layout.count())):
            layout.removeItem(layout.itemAt(i))

    @pyqtSlot(dict)
    def outevalsummary(self, result_data):
        pj_name=result_data["project_name"] 
        iou=result_data["IOU"] 
        gt_path=result_data["gt_path"] 
        predict_path=result_data["predict_path"]
        num=result_data["num"]
        self.result_datas=result_data["result"]
        summary_data=result_data["summary"]
 

        self.tableWidget_summary.setRowCount(0)


        self.tableWidget_summary.insertRow(0)
        self.tableWidget_summary.setItem(0, 0, QTableWidgetItem('project'))
        qproject_name=QTableWidgetItem(pj_name)
        self.tableWidget_summary.setItem(0, 1, qproject_name)

        self.tableWidget_summary.insertRow(1)
        self.tableWidget_summary.setItem(1, 0, QTableWidgetItem('gt_path'))
        qgt_path=QTableWidgetItem(gt_path)
        self.tableWidget_summary.setItem(1, 1, qgt_path)

        self.tableWidget_summary.insertRow(2)
        self.tableWidget_summary.setItem(2, 0, QTableWidgetItem('predict_path'))
        qpredict_path=QTableWidgetItem(predict_path)
        self.tableWidget_summary.setItem(2, 1, qpredict_path)

        self.tableWidget_summary.insertRow(3)
        self.tableWidget_summary.setItem(3, 0, QTableWidgetItem('number of files'))
        qnum=QTableWidgetItem(str(num))
        self.tableWidget_summary.setItem(3, 1, qnum)


        self.tableWidget_summary.insertRow(4)
        self.tableWidget_summary.setItem(4, 0, QTableWidgetItem('IOU'))
        qiou=QTableWidgetItem(str(iou))
        self.tableWidget_summary.setItem(4, 1, qiou)


        self.tableWidget_summary.insertRow(5)
        self.tableWidget_summary.setItem(5, 0, QTableWidgetItem('e2e_precision'))
        total_e2e_precision=QTableWidgetItem(str(summary_data["e2e_precision"]))
        self.tableWidget_summary.setItem(5, 1, total_e2e_precision)

        self.tableWidget_summary.insertRow(6)
        self.tableWidget_summary.setItem(6, 0, QTableWidgetItem('e2e_recall'))
        total_e2e_recall=QTableWidgetItem(str(summary_data["e2e_recall"]))
        self.tableWidget_summary.setItem(6, 1, total_e2e_recall)

        self.tableWidget_summary.insertRow(7)
        self.tableWidget_summary.setItem(7, 0, QTableWidgetItem('e2e_Hmean'))
        e2e_Hmean=QTableWidgetItem(str(summary_data["e2e_Hmean"]))
        self.tableWidget_summary.setItem(7, 1, e2e_Hmean)

        # nan_item.setForeground(QBrush(QColor(128, 128, 128)))
       
        table = self.tableWidget_summary
        header = table.horizontalHeader()
        twidth = header.width()
        width = []
        for column in range(header.count()):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
            width.append(header.sectionSize(column))

        wfactor = twidth / sum(width)
        for column in range(header.count()):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
            header.resizeSection(column, width[column]*wfactor)


        self.textEdit_summary.clear()
        pj_name=result_data["project_name"] 
        iou=result_data["IOU"] 
        self.result_datas=result_data["result"]
        summary_data=result_data["summary"]
        
        evalname = self.lineEdit_evalname.text()
        self.textEdit_summary.clear()
        self.textEdit_summary.append("project: " + str(pj_name))
        self.textEdit_summary.append("IOU: " + str(iou))
        self.textEdit_summary.append("e2e_precision: " + str(summary_data["e2e_precision"]))
        self.textEdit_summary.append("e2e_recall: " + str(summary_data["e2e_recall"]))
        self.textEdit_summary.append("e2e_Hmean: " + str(summary_data["e2e_Hmean"]))



    @pyqtSlot(dict)
    def outevalresult(self, result_data):
        self.print_result(result_data)


    @pyqtSlot(int,int)
    def onCountChanged(self, value,value2):
        p = value
        e = value2
        self.progressBar_eval.setValue(int(p/e*100))
    
    @pyqtSlot(str)
    def errorout(self, msg):
        QMessageBox.information(self,'error',msg)


    def pushButton_evaluation_fuction(self):
        
        if self.lineEdit_evalname.text() == '':
            QMessageBox.information(self,'error','project name')
            self.label_9.setText('')
            return

        self.calc = External(self)
        self.calc.countChanged.connect(self.onCountChanged)
        self.calc.outevalresult.connect(self.outevalresult)
        self.calc.outevalsummary.connect(self.outevalsummary)
        self.calc.errorout.connect(self.errorout)
        self.calc.start()

        self.pushButton_evaluation.setEnabled(False)
  


    def convert_cv_qt(self, cv_img, disply_width, display_height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(disply_width, display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def pain_bbox(self, img, bbox, color, thickness):
        bbox_img = np.full((img.shape[0], img.shape[1], 3), (0, 0, 0), dtype=np.uint8)
        cv2.rectangle(bbox_img, (bbox[1],bbox[2]), (bbox[3],bbox[4]), color, thickness)
        if thickness == -1:
            bboxs_img = np.full((img.shape[0], img.shape[1], 3), (0, 0, 0), dtype=np.uint8)
            cv2.rectangle(bboxs_img,(bbox[1],bbox[2]), (bbox[3],bbox[4]), ((100-color[0])/2 ,(100-color[1])/2,(100-color[2])/2), thickness)
            img = cv2.subtract(img, bboxs_img)
        img = cv2.add(img, bbox_img)
        return img


if __name__ == "__main__":

    app = QApplication(sys.argv) 

    myWindow = WindowClass() 

    myWindow.show()

    app.exec_()