# -*- coding: utf-8 -*-
import sys
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QGraphicsPixmapItem, QApplication, QGraphicsScene
from Fit_Distribution_APP import *
from Stats.Stats import Fit_Density
import numpy as np
import pandas as pd

class MyMainWindow(QMainWindow,Ui_WalsinMainWindow):
    """
        super() 函数是用于调用父类(超类)的一个方法。
        super() 是用来解决多重继承问题的，直接用类名调用父类方法在使用单继承的时候没问题，但是如果使用多继承，会涉及到查找顺序（MRO）、重复调用（钻石继承）等种种问题。
        MRO 就是类的方法解析顺序表, 其实也就是继承父类方法时的顺序表。
    """
    def __init__(self, parent=None):
        self.default_data=None
        super(MyMainWindow, self).__init__(parent) # 將自身類別對象轉為父類對象，並執行其初始化函數(依據MRO順序，即先轉為QMainWindow類)
        self.setupUi(self)  # 使用父類的
        #%%
        # 获取显示器分辨率大小
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
        self.resize(self.width,self.height)
        self.setFixedSize(self.width,self.height)
        # event
        self.actionClose.triggered.connect(self.close) # 觸發繼承父類中的物件actionClose附帶的triggered.connect，執行關閉表單
        self.actionOpen.triggered.connect(self.openMSG)
        self.pushButton_test.clicked.connect(self.testButtonClick)
        self.comboBox_select_columns.currentIndexChanged.connect(self.selec_col_change) # 連結槽與訊號
        self.comboBox_test_distribution.currentIndexChanged.connect(self.select_distribution_change) # 連結槽與訊號
        self.pushButton_plot.clicked.connect(self.plotButtonClick) # 連結槽與訊號

    def initUI(self):
        self.distribution_combox = "all"
        self.default_data_select_col = list(self.default_data.columns)[0]
        # self.initUI()  # 視窗init

    def select_distribution_change(self):
        self.statusbar.showMessage("Select distribution:{}".format(self.comboBox_test_distribution.currentText()))

    def selec_col_change(self):
        self.statusbar.showMessage("Select column:{}".format(self.comboBox_select_columns.currentText()))

    def openMSG(self):
        file,ok=QFileDialog.getOpenFileName(self,"開啟","","Excel (*.xlsx);;Excel (*.xls);;CSV (*.csv);;Text Files (*.txt)")
        self.default_data=pd.read_excel(file)
        self.comboBox_select_columns.clear()
        self.comboBox_select_columns.addItems(list(self.default_data.columns))  # 加入多個下拉選單項目
        # for i,col_name in enumerate(list(self.default_data.columns)):
        #     self.comboBox_select_columns.setItemText(i,col_name)
        self.statusbar.showMessage("File load :{}".format(file))

    def testButtonClick(self):
        # self.default_data = pd.read_excel('Stats/直棒各站生產資料.xlsx')
        #%% 狀態欄位顯示測試欄位的資訊
        self.statusbar.showMessage("測試欄位:{}".format(self.comboBox_select_columns.currentText()))
        print("測試欄位:{}".format(self.comboBox_select_columns.currentText()))
        # %% 若沒選擇欄位則顯示"Need to Load data"
        if self.comboBox_select_columns.currentText()=='None':
            self.statusbar.showMessage("Need to Load data")
            return
        #%% 再狀態欄位顯示按下了測試按鈕
        sender = self.sender() # sender 是傳送訊號的物件
        self.statusbar.showMessage("Clicked {}".format(sender.text()))

        #%% 挑初指定欄位的Dataframe
        data = self.default_data[["WIP時間"]] # #self.comboBox_select_columns.currentText()
        data = data.dropna(inplace=False)
        data_= data.values
        #%% 設定信心水準
        self.sigma = float(self.lineEdit_significant.text())
        #%% 設定顯示訊息
        message="""Distribution Summary:\nDistribution:	= {}\nTest Statistic	= {}\nsigma	        = {}\np-value	        = {}\nMSE             = {}\nThe test result : {}\nExpression:{}\nData Summary\nNumber of Data Points	= {}\n"""
        message_=""
        if self.comboBox_test_distribution.currentText()=='all': #
            distributions=None
        else:
            distributions = [self.comboBox_test_distribution.currentText()]#
        print("以{}檢驗".format(self.comboBox_test_distribution.currentText()))
        self.fd = Fit_Density(data_, distributions=distributions)
        self.fd.fit()
        self.all_params = self.fd.fitted_param
        self.fd.get_simio_expression()
        self.best_params = self.fd.get_best(N=1)
        self.best_distname=list(self.best_params.keys())[0]
        self.test_score = self.fd.get_test_score(dist_names=distributions, vb=0)
        self.best_statistic=self.test_score[self.best_distname]['statistic']
        self.best_p_value=self.test_score[self.best_distname]['p-value']
        self.best_loss=self.test_score[self.best_distname]['mse']
        self.best_expression = self.fd.fitted_param_simio[self.best_distname]
        message_ += message.format(self.best_distname, self.best_statistic, self.sigma, self.best_p_value, self.best_loss,
                                   (self.best_p_value > self.sigma), self.best_expression, len(data_))
        message_ += "===========================================\n"
        del self.test_score[self.best_distname]
        for dist_name, score in self.test_score.items():
            statistic, p_value, loss = score["statistic"], score['p-value'], score['mse']
            expression = self.fd.fitted_param_simio[dist_name]
            message_ += message.format(dist_name, statistic, self.sigma, p_value,
                                       loss, (p_value > self.sigma), expression, len(data_))
            message_ += "========================\n"
        self.textBrowser_test_result.setText(message_)
    def plotButtonClick(self):
        if self.comboBox_select_columns.currentText() == 'None':
            self.statusbar.showMessage("Need to Load data")
        # sender 是傳送訊號的物件
        sender = self.sender()
        self.statusbar.showMessage("Clicked {}".format(sender.text()))
        self.fd.Density_Plot(self.default_data[[self.comboBox_select_columns.currentText()]],  vb=0, path="TEST.png")
        img = cv2.imread("TEST.png")  # 讀取影象
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換影象通道
        x = img.shape[1]  # 獲取影象大小
        y = img.shape[0]
        self.zoomscale = 1  # 圖片放縮尺度
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 建立畫素圖元
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  # 建立場景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    qstyle = """
        QPushButton[name="pushButton_test"] {color:red}
        """
    myWin.setStyleSheet(qstyle)
    # myWin.testButtonClick()
    myWin.show()
    sys.exit(app.exec_())