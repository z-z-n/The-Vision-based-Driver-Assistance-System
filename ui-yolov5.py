# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui-yolov5.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
import sys


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.pix_up = QPixmap('./icon/arrow-up.png')
        self.pix_cr = QPixmap('./icon/car.png')
        self.pix_c1 = QPixmap('./icon/car1.png')
        self.pix_ca = QPixmap('./icon/caution.png')
        self.pix_lf = QPixmap('./icon/corner-up-left.png')
        self.pix_rg = QPixmap('./icon/corner-up-right.png')
        self.pix_rt = QPixmap('./icon/result.png')
        self.setupUi(self)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1790, 1225)
        MainWindow.setMinimumSize(QtCore.QSize(1790, 1200))
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("background-color: rgb(202, 203, 224);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(1700, 1200))
        self.centralwidget.setMaximumSize(QtCore.QSize(2160, 1440))
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 2, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem2, 2, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem3, 0, 1, 1, 1)
        self.frame_background = QtWidgets.QFrame(self.centralwidget)
        self.frame_background.setMinimumSize(QtCore.QSize(1710, 1140))
        self.frame_background.setStyleSheet("background-color: rgb(202,203,224);")
        self.frame_background.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_background.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_background.setObjectName("frame_background")
        self.frame_left = QtWidgets.QFrame(self.frame_background)
        self.frame_left.setGeometry(QtCore.QRect(20, 60, 580, 1080))
        self.frame_left.setMinimumSize(QtCore.QSize(580, 1080))
        self.frame_left.setMaximumSize(QtCore.QSize(580, 1080))
        self.frame_left.setStyleSheet("background-color: rgb(226, 227, 248);border-radius:35px;")
        self.frame_left.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_left.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_left.setObjectName("frame_left")
        self.label_car = QtWidgets.QLabel(self.frame_left)
        self.label_car.setGeometry(QtCore.QRect(90, 30, 400, 550))
        self.label_car.setMinimumSize(QtCore.QSize(400, 500))
        self.label_car.setStyleSheet("background-color: rgb(170, 170, 255);")
        self.label_car.setObjectName("label_car")
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.label_car.setPixmap(self.pix_c1)
        self.label_car.setScaledContents(True)  # 自适应QLabel大小
        self.frame_obnum = QtWidgets.QFrame(self.frame_left)
        self.frame_obnum.setGeometry(QtCore.QRect(40, 640, 220, 220))
        self.frame_obnum.setStyleSheet("background-color: rgb(62,68,92);")
        self.frame_obnum.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_obnum.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_obnum.setObjectName("frame_obnum")
        self.frame_ob_icon = QtWidgets.QFrame(self.frame_obnum)
        self.frame_ob_icon.setGeometry(QtCore.QRect(30, 30, 71, 71))
        self.frame_ob_icon.setStyleSheet("background-color:rgb(41,122,255);")
        self.frame_ob_icon.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_ob_icon.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_ob_icon.setObjectName("frame_ob_icon")
        self.label_ob_icon = QtWidgets.QLabel(self.frame_ob_icon)
        self.label_ob_icon.setGeometry(QtCore.QRect(16, 15, 40, 40))
        self.label_ob_icon.setObjectName("label_ob_icon")
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.label_ob_icon.setPixmap(self.pix_cr)
        self.label_ob_icon.setScaledContents(True)  # 自适应QLabel大小
        self.label_ob_num = QtWidgets.QLabel(self.frame_obnum)
        self.label_ob_num.setGeometry(QtCore.QRect(110, 65, 91, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_ob_num.setFont(font)
        self.label_ob_num.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_ob_num.setObjectName("label_ob_num")
        self.label_ob_car = QtWidgets.QLabel(self.frame_obnum)
        self.label_ob_car.setGeometry(QtCore.QRect(30, 110, 170, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_ob_car.setFont(font)
        self.label_ob_car.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_ob_car.setObjectName("label_ob_car")
        self.label_ob_ped = QtWidgets.QLabel(self.frame_obnum)
        self.label_ob_ped.setGeometry(QtCore.QRect(30, 140, 170, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_ob_ped.setFont(font)
        self.label_ob_ped.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_ob_ped.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.label_ob_ped.setObjectName("label_ob_ped")
        self.label_ob_cyc = QtWidgets.QLabel(self.frame_obnum)
        self.label_ob_cyc.setGeometry(QtCore.QRect(30, 170, 170, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_ob_cyc.setFont(font)
        self.label_ob_cyc.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_ob_cyc.setObjectName("label_ob_cyc")
        self.frame_fcw = QtWidgets.QFrame(self.frame_left)
        self.frame_fcw.setGeometry(QtCore.QRect(310, 640, 220, 220))
        self.frame_fcw.setStyleSheet("background-color: rgb(62,68,92);")
        self.frame_fcw.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_fcw.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_fcw.setObjectName("frame_fcw")
        self.frame_fcw_icon = QtWidgets.QFrame(self.frame_fcw)
        self.frame_fcw_icon.setGeometry(QtCore.QRect(30, 30, 71, 71))
        self.frame_fcw_icon.setStyleSheet("background-color:rgb(41,122,255);")
        self.frame_fcw_icon.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_fcw_icon.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_fcw_icon.setObjectName("frame_fcw_icon")
        self.label_fcw_icon = QtWidgets.QLabel(self.frame_fcw_icon)
        self.label_fcw_icon.setGeometry(QtCore.QRect(16, 15, 40, 40))
        self.label_fcw_icon.setObjectName("label_fcw_icon")
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.label_fcw_icon.setPixmap(self.pix_ca)
        self.label_fcw_icon.setScaledContents(True)  # 自适应QLabel大小
        self.label_fcw_1 = QtWidgets.QLabel(self.frame_fcw)
        self.label_fcw_1.setGeometry(QtCore.QRect(120, 70, 91, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_fcw_1.setFont(font)
        self.label_fcw_1.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_fcw_1.setObjectName("label_fcw_1")
        self.label_fcw_state = QtWidgets.QLabel(self.frame_fcw)
        self.label_fcw_state.setGeometry(QtCore.QRect(20, 180, 170, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_fcw_state.setFont(font)
        self.label_fcw_state.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_fcw_state.setAlignment(QtCore.Qt.AlignCenter)
        self.label_fcw_state.setObjectName("label_fcw_state")
        self.label_fcw3 = QtWidgets.QLabel(self.frame_fcw)
        self.label_fcw3.setGeometry(QtCore.QRect(40, 140, 170, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_fcw3.setFont(font)
        self.label_fcw3.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_fcw3.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.label_fcw3.setObjectName("label_fcw3")
        self.label_fcw_2 = QtWidgets.QLabel(self.frame_fcw)
        self.label_fcw_2.setGeometry(QtCore.QRect(40, 110, 170, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_fcw_2.setFont(font)
        self.label_fcw_2.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_fcw_2.setObjectName("label_fcw_2")
        self.label_fcw_title = QtWidgets.QLabel(self.frame_fcw)
        self.label_fcw_title.setGeometry(QtCore.QRect(110, 30, 101, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_fcw_title.setFont(font)
        self.label_fcw_title.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_fcw_title.setAlignment(QtCore.Qt.AlignCenter)
        self.label_fcw_title.setObjectName("label_fcw_title")
        self.label_deviation = QtWidgets.QLabel(self.frame_left)
        self.label_deviation.setGeometry(QtCore.QRect(80, 550, 181, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_deviation.setFont(font)
        self.label_deviation.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_deviation.setStyleSheet("color: rgb(0, 0, 0);background: transparent;")
        self.label_deviation.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_deviation.setObjectName("label_deviation")
        self.label_dev_num = QtWidgets.QLabel(self.frame_left)
        self.label_dev_num.setGeometry(QtCore.QRect(80, 585, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(28)
        font.setBold(True)
        font.setWeight(75)
        self.label_dev_num.setFont(font)
        self.label_dev_num.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_dev_num.setStyleSheet("color: rgb(0, 0, 0);background: transparent；")
        self.label_dev_num.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_dev_num.setObjectName("label_dev_num")
        self.label_dev_m = QtWidgets.QLabel(self.frame_left)
        self.label_dev_m.setGeometry(QtCore.QRect(210, 595, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_dev_m.setFont(font)
        self.label_dev_m.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_dev_m.setStyleSheet("color: rgb(0, 0, 0);background: transparent；")
        self.label_dev_m.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_dev_m.setObjectName("label_dev_m")
        self.label_car_r = QtWidgets.QLabel(self.frame_left)
        self.label_car_r.setGeometry(QtCore.QRect(400, 30, 171, 501))
        self.label_car_r.setMinimumSize(QtCore.QSize(0, 0))
        self.label_car_r.setStyleSheet("background-color: rgba(197,239,253,200);")
        self.label_car_r.setText("")
        self.label_car_r.setObjectName("label_car_r")
        self.label_car_l = QtWidgets.QLabel(self.frame_left)
        self.label_car_l.setGeometry(QtCore.QRect(10, 30, 171, 501))
        self.label_car_l.setMinimumSize(QtCore.QSize(0, 0))
        self.label_car_l.setStyleSheet("background-color: rgba(232,87,77,200);")
        self.label_car_l.setText("")
        self.label_car_l.setObjectName("label_car_l")
        self.pushButton_4 = QtWidgets.QPushButton(self.frame_left)
        self.pushButton_4.setGeometry(QtCore.QRect(140, 990, 130, 70))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setStyleSheet("QPushButton{background-color: rgb(240,240,252);border-radius:35px;border:2px groove rgb(226,227,248);border-style:outset;}\n"
"QPushButton:checked { background-color: rgb(80,97,229); border-style: inset; color:white;}")
        self.pushButton_4.setCheckable(True)
        self.pushButton_4.setAutoExclusive(True)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.frame_left)
        self.pushButton_5.setGeometry(QtCore.QRect(290, 990, 130, 70))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setStyleSheet("QPushButton{background-color: rgb(240,240,252);border-radius:35px;border:2px groove rgb(226,227,248);border-style:outset;}\n"
"QPushButton:checked { background-color: rgb(80,97,229); border-style: inset; color:white;}")
        self.pushButton_5.setCheckable(True)
        self.pushButton_5.setAutoExclusive(True)
        self.pushButton_5.setObjectName("pushButton_5")
        self.frame = QtWidgets.QFrame(self.frame_left)
        self.frame.setGeometry(QtCore.QRect(50, 900, 471, 81))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(10, 0, 130, 70))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("QPushButton{background-color: rgb(240,240,252);border-radius:35px;border:2px groove rgb(226,227,248);border-style:outset;}\n"
"QPushButton:checked { background-color: rgb(80,97,229); border-style: inset; color:white;}")
        self.pushButton.setCheckable(True)
        self.pushButton.setAutoExclusive(True)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(170, 0, 130, 71))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("QPushButton{background-color: rgb(240,240,252);border-radius:35px;border:2px groove rgb(226,227,248);border-style:outset;}\n"
"QPushButton:checked { background-color: rgb(80,97,229); border-style: inset; color:white;}")
        self.pushButton_2.setCheckable(True)
        self.pushButton_2.setAutoExclusive(True)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.frame)
        self.pushButton_3.setGeometry(QtCore.QRect(330, 0, 130, 70))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setStyleSheet("QPushButton{background-color: rgb(240,240,252);border-radius:35px;border:2px groove rgb(226,227,248);border-style:outset;}\n"
"QPushButton:checked { background-color: rgb(80,97,229); border-style: inset; color:white;}")
        self.pushButton_3.setCheckable(True)
        self.pushButton_3.setChecked(True)
        self.pushButton_3.setObjectName("pushButton_3")
        self.frame_right = QtWidgets.QFrame(self.frame_background)
        self.frame_right.setGeometry(QtCore.QRect(710, 60, 980, 1080))
        self.frame_right.setMinimumSize(QtCore.QSize(980, 1080))
        self.frame_right.setMaximumSize(QtCore.QSize(2000, 2000))
        self.frame_right.setStyleSheet("background-color: rgb(226, 227, 248);border-radius:35px")
        self.frame_right.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_right.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_right.setObjectName("frame_right")
        self.label_result = QtWidgets.QLabel(self.frame_right)
        self.label_result.setGeometry(QtCore.QRect(40, 40, 900, 800))
        self.label_result.setMinimumSize(QtCore.QSize(900, 800))
        self.label_result.setStyleSheet("background-color: rgb(170, 170, 255);")
        self.label_result.setObjectName("label_result")
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.label_result.setPixmap(self.pix_rt)
        self.label_result.setScaledContents(True)  # 自适应QLabel大小
        self.frame_prompt = QtWidgets.QFrame(self.frame_right)
        self.frame_prompt.setGeometry(QtCore.QRect(80, 70, 281, 161))
        self.frame_prompt.setStyleSheet("background-color: rgba(41,122,254,150);")
        self.frame_prompt.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_prompt.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_prompt.setObjectName("frame_prompt")
        self.label_direction = QtWidgets.QLabel(self.frame_prompt)
        self.label_direction.setGeometry(QtCore.QRect(100, 60, 171, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_direction.setFont(font)
        self.label_direction.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_direction.setStyleSheet("color: rgb(255, 255, 255);background: transparent;")
        self.label_direction.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_direction.setObjectName("label_direction")
        self.label = QtWidgets.QLabel(self.frame_prompt)
        self.label.setGeometry(QtCore.QRect(20, 30, 71, 71))
        self.label.setStyleSheet("color: rgb(255, 255, 255);background: transparent;")
        self.label.setObjectName("label")
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.label.setPixmap(self.pix_up)
        self.label.setScaledContents(True)  # 自适应QLabel大小
        self.label_direction_2 = QtWidgets.QLabel(self.frame_prompt)
        self.label_direction_2.setGeometry(QtCore.QRect(30, 110, 221, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_direction_2.setFont(font)
        self.label_direction_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_direction_2.setStyleSheet("color: rgb(255, 255, 255);background: transparent;")
        self.label_direction_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_direction_2.setObjectName("label_direction_2")
        self.frame_conf = QtWidgets.QFrame(self.frame_right)
        self.frame_conf.setGeometry(QtCore.QRect(360, 870, 581, 80))
        self.frame_conf.setStyleSheet("background-color: rgb(240,240,252);border-radius:35px;border:2px groove rgb(226,227,248);border-style:outset;")
        self.frame_conf.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_conf.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_conf.setObjectName("frame_conf")
        self.Slider_conf = QtWidgets.QSlider(self.frame_conf)
        self.Slider_conf.setGeometry(QtCore.QRect(180, 20, 350, 40))
        self.Slider_conf.setTabletTracking(False)
        self.Slider_conf.setStyleSheet("QSlider{background-color: rgb(240,240,252);border-width:0px;}")
        self.Slider_conf.setMaximum(99)
        self.Slider_conf.setSliderPosition(10)
        self.Slider_conf.setOrientation(QtCore.Qt.Horizontal)
        self.Slider_conf.setInvertedAppearance(False)
        self.Slider_conf.setInvertedControls(False)
        self.Slider_conf.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.Slider_conf.setTickInterval(10)
        self.Slider_conf.setObjectName("Slider_conf")
        self.spinBox_conf = QtWidgets.QSpinBox(self.frame_conf)
        self.spinBox_conf.setGeometry(QtCore.QRect(100, 30, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.spinBox_conf.setFont(font)
        self.spinBox_conf.setStyleSheet("background-color: rgb(240,240,252);border-width:0px;")
        self.spinBox_conf.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.spinBox_conf.setKeyboardTracking(False)
        self.spinBox_conf.setObjectName("spinBox_conf")
        self.label_conf = QtWidgets.QLabel(self.frame_conf)
        self.label_conf.setGeometry(QtCore.QRect(20, 30, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_conf.setFont(font)
        self.label_conf.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_conf.setStyleSheet("border-width:0px;")
        self.label_conf.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_conf.setObjectName("label_conf")
        self.frame_IOU = QtWidgets.QFrame(self.frame_right)
        self.frame_IOU.setGeometry(QtCore.QRect(360, 970, 581, 80))
        self.frame_IOU.setStyleSheet("background-color: rgb(240,240,252);border-radius:35px;border:2px groove rgb(226,227,248);border-style:outset;")
        self.frame_IOU.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_IOU.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_IOU.setObjectName("frame_IOU")
        self.Slider_iou = QtWidgets.QSlider(self.frame_IOU)
        self.Slider_iou.setGeometry(QtCore.QRect(180, 30, 350, 40))
        self.Slider_iou.setTabletTracking(False)
        self.Slider_iou.setStyleSheet("background-color: rgb(240,240,252);border-width:0px;")
        self.Slider_iou.setMaximum(99)
        self.Slider_iou.setSliderPosition(10)
        self.Slider_iou.setOrientation(QtCore.Qt.Horizontal)
        self.Slider_iou.setInvertedAppearance(False)
        self.Slider_iou.setInvertedControls(False)
        self.Slider_iou.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.Slider_iou.setTickInterval(10)
        self.Slider_iou.setObjectName("Slider_iou")
        self.spinBox_iou = QtWidgets.QSpinBox(self.frame_IOU)
        self.spinBox_iou.setGeometry(QtCore.QRect(100, 30, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.spinBox_iou.setFont(font)
        self.spinBox_iou.setStyleSheet("background-color: rgb(240,240,252);border-width:0px;")
        self.spinBox_iou.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.spinBox_iou.setKeyboardTracking(False)
        self.spinBox_iou.setObjectName("spinBox_iou")
        self.label_iou = QtWidgets.QLabel(self.frame_IOU)
        self.label_iou.setGeometry(QtCore.QRect(30, 30, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_iou.setFont(font)
        self.label_iou.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_iou.setStyleSheet("border-width:0px;")
        self.label_iou.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_iou.setObjectName("label_iou")
        self.pushButton_6 = QtWidgets.QPushButton(self.frame_right)
        self.pushButton_6.setGeometry(QtCore.QRect(90, 880, 201, 70))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setStyleSheet("QPushButton{background-color: rgb(240,240,252);border-radius:35px;border:2px groove rgb(226,227,248);border-style:outset;}\n"
"QPushButton:checked { background-color: rgb(80,97,229); border-style: inset; color:white;}")
        self.pushButton_6.setCheckable(False)
        self.pushButton_6.setAutoExclusive(False)
        self.pushButton_6.setObjectName("pushButton_6")
        self.label_weather = QtWidgets.QLabel(self.frame_background)
        self.label_weather.setGeometry(QtCore.QRect(1460, 10, 91, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_weather.setFont(font)
        self.label_weather.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_weather.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_weather.setObjectName("label_weather")
        self.label_time = QtWidgets.QLabel(self.frame_background)
        self.label_time.setGeometry(QtCore.QRect(1590, 10, 91, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_time.setFont(font)
        self.label_time.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_time.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_time.setObjectName("label_time")
        self.label_weather_4 = QtWidgets.QLabel(self.frame_background)
        self.label_weather_4.setGeometry(QtCore.QRect(40, 10, 521, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_weather_4.setFont(font)
        self.label_weather_4.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_weather_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_weather_4.setObjectName("label_weather_4")
        self.gridLayout.addWidget(self.frame_background, 1, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ADAS视觉驾驶辅助系统"))
        # self.label_car.setText(_translate("MainWindow", "Image"))
        # self.label_ob_icon.setText(_translate("MainWindow", "ICON"))
        self.label_ob_num.setText(_translate("MainWindow", "Object: 10"))
        self.label_ob_car.setText(_translate("MainWindow", "Cars: 10"))
        self.label_ob_ped.setText(_translate("MainWindow", "Pedestrians: 10"))
        self.label_ob_cyc.setText(_translate("MainWindow", "Cyclists: 10"))
        # self.label_fcw_icon.setText(_translate("MainWindow", "ICON"))
        self.label_fcw_1.setText(_translate("MainWindow", "Dis: 10 m"))
        self.label_fcw_state.setText(_translate("MainWindow", "Activated"))
        self.label_fcw3.setText(_translate("MainWindow", "3rd: 10 m"))
        self.label_fcw_2.setText(_translate("MainWindow", "2nd: 10 m"))
        self.label_fcw_title.setText(_translate("MainWindow", "FCW"))
        self.label_deviation.setText(_translate("MainWindow", "Deviation"))
        self.label_dev_num.setText(_translate("MainWindow", "0.12"))
        self.label_dev_m.setText(_translate("MainWindow", "m"))
        self.pushButton_4.setText(_translate("MainWindow", "Start"))
        self.pushButton_5.setText(_translate("MainWindow", "End"))
        self.pushButton.setText(_translate("MainWindow", "Video"))
        self.pushButton_2.setText(_translate("MainWindow", "Image"))
        self.pushButton_3.setText(_translate("MainWindow", "AutoSave"))
        # self.label_result.setText(_translate("MainWindow", "Image"))
        self.label_direction.setText(_translate("MainWindow", "Turn Right"))
        # self.label.setText(_translate("MainWindow", "ICON"))
        self.label_direction_2.setText(_translate("MainWindow", "Radio of curvation: 71m"))
        self.label_conf.setText(_translate("MainWindow", "Conf:"))
        self.label_iou.setText(_translate("MainWindow", "IoU:"))
        self.pushButton_6.setText(_translate("MainWindow", "Model: YOLOv5"))
        self.label_weather.setText(_translate("MainWindow", "20℃"))
        self.label_time.setText(_translate("MainWindow", "08:40"))
        self.label_weather_4.setText(_translate("MainWindow", "ヾ(•ω•`)o：Hi, wish you a good mood today ~"))

if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv)
    ui=Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())