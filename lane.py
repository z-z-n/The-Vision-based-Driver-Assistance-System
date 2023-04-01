# Author: Zhining Zhang
# Task: Complete bend detection

import cv2
import numpy as np
import time
from collections import *


class LANE_FRAME:
    def __init__(self, average=5):
        # 帧数计数
        self.FID = 0
        # 平均的帧数，默认10帧平均
        self.frame = average
        # 最近车辆状态: 0直行； 1左转; 2有转；
        self.recent_st = deque(maxlen=self.frame * 3)
        # 最近曲率半径
        self.recent_radio = deque(maxlen=self.frame * 4)
        # 最近转向偏离度
        self.recent_curve = deque(maxlen=self.frame * 2)

        # 左车道线
        # 当前帧的多项式值
        self.l_current_fit = []
        # 最近的几帧
        self.l_recent_fit = deque(maxlen=self.frame)
        # 平均的多项式
        self.l_mean_fit = []
        # 是否有左车道线
        self.l_detected = True
        # 稀疏度
        self.l_recent_sp = deque(maxlen=self.frame * 3)

        # 右车道线
        # 当前帧的多项式值
        self.r_current_fit = []
        # 最近的几帧
        self.r_recent_fit = deque(maxlen=self.frame)
        # 平均的多项式
        self.r_mean_fit = []
        # 是否有左车道线
        self.r_detected = True
        # 稀疏度
        self.r_recent_sp = deque(maxlen=self.frame * 3)

    def update_info(self, l_x, l_y, r_x, r_y):
        self.FID = self.FID + 1
        # 判断当前是否检测到了左右2条车道线
        # 如果点的个数小于等于1300个，认为无效
        if len(l_x) <= 1300:
            self.l_detected = False
        if len(r_x) <= 1300:
            self.r_detected = False

        # 计算左右车道线 x均值
        l_x_mean = np.mean(l_x, axis=0)
        r_x_mean = np.mean(r_x, axis=0)
        lane_width = np.subtract(r_x_mean, l_x_mean)

        # 如果车道线过宽或过窄，认为无效
        # 调整窗口后需要调整***************************************
        if lane_width < 300 or lane_width > 700:
            self.l_detected = False
            self.r_detected = False

        # 分别处理左右车道线
        if self.l_detected:
            self.l_current_fit = np.polyfit(l_y, l_x, 2)
            self.l_recent_fit.append(self.l_current_fit)
            self.l_mean_fit = np.mean(self.l_recent_fit, axis=0)
        else:
            self.l_current_fit = None

        # 分别处理左右车道线
        if self.r_detected:
            self.r_current_fit = np.polyfit(r_y, r_x, 2)
            self.r_recent_fit.append(self.r_current_fit)
            self.r_mean_fit = np.mean(self.r_recent_fit, axis=0)
        else:
            self.r_current_fit = None

    def update_sparsity(self, l_sp, r_sp):
        self.l_recent_sp.append(l_sp)
        self.r_recent_sp.append(r_sp)
        l_mean = np.mean(self.l_recent_sp)
        r_mean = np.mean(self.r_recent_sp)
        # return min([l_mean, l_sp]), min([r_mean, r_sp])
        return l_mean, r_mean

    def update_laneInfo(self, radio, curve):
        self.recent_radio.append(radio.astype(int))
        self.recent_curve.append(curve.astype(float))
        return np.mean(self.recent_radio), np.mean(self.recent_curve)


class LANE_DETECTION:
    def __init__(self):
        self.frame_img = LANE_FRAME()

    '''
    图像预处理：1.HLS颜色过滤；2.sobel算子过滤 （2者利用阈值保留并结合）
    '''

    # HLS 阈值过滤，保留黄色和白色区域
    def HLS_filter(self,img):
        # convert to HLS to mask based on HLS
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # 白色min和max阈值
        lower = np.array([0, 190, 0])
        upper = np.array([255, 255, 255])
        # 黄色min阈值和max阈值
        yellower = np.array([10, 0, 90])
        yelupper = np.array([200, 255, 255])
        yellowmask = cv2.inRange(hls, yellower, yelupper)
        whitemask = cv2.inRange(hls, lower, upper)
        HLS_bin = cv2.bitwise_or(yellowmask, whitemask)  # black=0, white=255
        # HLS_bin[HLS_bin == 255] = 1  # turn to binary
        # print(mask)
        return HLS_bin

    # x方向上的sobel算子阈值过滤
    def sobel_filter(self,img, thresh_min=20, thresh_max=100):
        # 原图， 阈值范围20-100
        # 转为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # print(gray)
        # 计算x方向上梯度 Take the derivative in x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        # 取绝对值 Absolute x derivative
        abs_sobelx = np.absolute(sobelx)
        max_sobelx = np.max(abs_sobelx)
        # 图像转为阈值范围，unit8将大于255的截断（图像均为unit8）
        scaled_sobel = np.uint8(255 * abs_sobelx / max_sobelx)
        # 创建x方向上的二值图0,255
        sobel_bin = np.zeros_like(scaled_sobel)
        sobel_bin[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255
        return sobel_bin

    def HLS_Sfilter(self, img, thresh_min=130, thresh_max=255):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 1]
        s_bin = np.zeros_like(s_channel)
        s_bin[(s_channel >= thresh_min) & (s_channel <= thresh_max)] = 255
        return s_bin

    # 预处理函数
    def preprocess(self,img):
        HLS_bin = self.HLS_filter(img)
        sobel_bin = self.sobel_filter(img)
        Schannel_bin = self.HLS_Sfilter(img)
        # HLS过滤和sobel的x过滤结合
        combined = np.zeros_like(sobel_bin)
        # combined[(HLS_bin == 255) | (sobel_bin == 255)] = 255
        combined[((HLS_bin == 255) & (Schannel_bin == 255)) | (sobel_bin == 255)] = 255
        return combined

    # 不感兴趣区域
    def ROU(self, img, mask):
        cv2.fillPoly(img, mask, 0)
        return img

    '''
    透视变换：将图片转化为俯视图，保留感兴趣区域
    '''

    def perspective_img(self, img, img0, draw=True):
        # 二值图，原彩色图，是否画图的flag
        # 图片大小，opencv读取，[1]是宽度[0]是高度
        img_size = (img.shape[1], img.shape[0])
        dst_size = (800, 450)
        # src：源图像中待测矩形的四点坐标
        # dst：目标图像中矩形的四点坐标
        # src0 = np.array([(0.46, 0.625), (0.54, 0.625), (0.1, 1), (1, 1)], dtype="float32")
        dst0 = np.array([(0.2, 0), (0.8, 0), (0.2, 1), (0.8, 1)], dtype="float32")
        # dst0 = np.array([(0.3, 0), (0.7, 0), (0.3, 1), (0.7, 1)], dtype="float32")
        src0 = np.array([(0.44, 0.65), (0.57, 0.65), (0.1, 1), (1, 1)], dtype="float32")
        # dst0 = np.array([(0.1, 0), (0.9, 0), (0.1, 1), (0.9, 1)], dtype="float32")
        src = np.float32(img_size) * src0
        dst = np.float32(dst_size) * dst0
        R = cv2.getPerspectiveTransform(src, dst)
        warped_bin = cv2.warpPerspective(img, R, dst_size)
        warped_color = cv2.warpPerspective(img0, R, dst_size)

        # 去除不感兴趣区域
        # 中间
        mask0 = np.array([(0.4, 1), (0.6, 1), (0.4, 0.5), (0.6, 0.5)], dtype="float32")
        mask0 = np.int32(dst_size * mask0)
        mask0[[0, 1], :] = mask0[[1, 0], :]
        # 右
        mask1 = np.array([(0.9, 1), (1, 1), (1, 0), (0.9, 0)], dtype="float32")
        # 左
        mask2 = np.array([(0, 1), (0.1, 1), (0.1, 0), (0, 0)], dtype="float32")
        mask1 = np.int32(dst_size * mask1)
        mask2 = np.int32(dst_size * mask2)
        warped_bin = self.ROU(warped_bin, [mask0])
        warped_bin = self.ROU(warped_bin, [mask1])
        warped_bin = self.ROU(warped_bin, [mask2])

        # 是否画出感兴趣区域
        if draw:
            # 感兴趣区域——画多边形
            img_ROI = np.zeros_like(img0)
            mask = dst.astype(np.int32)
            mask[[0, 1], :] = mask[[1, 0], :]
            mask3 = np.array([(0.1, 1), (0.1, 0), (0.9, 0), (0.9, 1)], dtype="float32")
            mask3 = np.int32(dst_size * mask3)
            # cv2.fillPoly(img_ROI, [mask3], (106, 90, 205))
            cv2.fillPoly(img_ROI, [mask3], (0, 255, 0))
            cv2.fillPoly(img_ROI, [mask0], (0, 0, 0))
            cv2.polylines(img_ROI, [mask], True, (0, 0, 255), 5)
            area = self.re_perspective_img(img_ROI)
            img_add = cv2.addWeighted(img0, 1, area, 0.7, 0)
            ROI = cv2.resize(img_add, (800, 450), interpolation=cv2.INTER_AREA)
            # cv2.imshow("ROI", ROI)
            # cv2.waitKey(0)

        return warped_bin, warped_color

    # 逆透视变换（将俯视图转为前视图）
    def re_perspective_img(self,img):
        # 原图
        # 图片大小，opencv读取，[1]是宽度[0]是高度
        img_size = (800, 450)
        dst_size = (img.shape[1], img.shape[0])
        # src：源图像中待测矩形的四点坐标
        # dst：目标图像中矩形的四点坐标
        src0 = np.array([(0.1, 0), (0.9, 0), (0.1, 1), (0.9, 1)], dtype="float32")
        src0 = np.array([(0.2, 0), (0.8, 0), (0.2, 1), (0.8, 1)], dtype="float32")
        # src0 = np.array([(0.3, 0), (0.7, 0), (0.3, 1), (0.7, 1)], dtype="float32")
        dst0 = np.array([(0.44, 0.65), (0.57, 0.65), (0.1, 1), (1, 1)], dtype="float32")
        src = np.float32(img_size) * src0
        dst = np.float32(dst_size) * dst0
        R = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, R, dst_size)

        return warped

    '''
    滑动窗口
    '''

    # 5窗口方法：根据上一次窗口内有效像素的x均值和窗口的高度，一次划分左，右，左上，中上和右上 共5个窗口
    # 该方法有助于当出现急转弯时，左右两条车道线水平延申的情况；也有助于当转弯时左右2车道进入同一侧
    def update_win5(self, img_RGB, win_h, win_w, win_y, x_current, valid_x, valid_y, min_p):
        # 3通道彩图，窗口高度，窗口宽度，之前窗口y高度下标，之前窗口有效像素x下标均值，有效像素x下标list，有效像素y下标list，窗口内最少点的个数
        # 更新当前窗口的xy坐标（每个窗口需要左下和右上2个像素
        # 3种y值
        win_y_low = win_y
        win_y_up = win_y - win_h
        win_y_upper = win_y_up - win_h
        # 左下像素的元组list：（x，y）
        win_low = [(x_current - win_w, win_y_low), (x_current, win_y_low), (int(x_current - win_w * 1.5), win_y_up),
                   (int(x_current - win_w * 0.5), win_y_up), (int(x_current + win_w * 0.5), win_y_up)]
        # 右上像素的元组list：（x，y）
        win_up = [(x_current, win_y_up), (x_current + win_w, win_y_up), (int(x_current - win_w * 0.5), win_y_upper),
                  (int(x_current + win_w * 0.5), win_y_upper), (int(x_current + win_w * 1.5), win_y_upper)]

        # 初始化有效像素的index和包含最多像素的窗口下标
        inside_idx = []
        num = 0
        for i in range(5):
            # cv2.rectangle(img_RGB, win_low[i], win_up[i], (0, 255, 0), 2) #显示5个窗口
            # 获得窗口内像素的index：获得True和False组成的矩阵，利用非零点获得其在list中的下标
            inside_idx_t = ((valid_y <= win_low[i][1]) & (valid_y > win_up[i][1]) & (valid_x >= win_low[i][0]) & (
                    valid_x < win_up[i][0])).nonzero()[0]
            # 更新最多点的窗口
            if len(inside_idx_t) > len(inside_idx):
                num = i
                inside_idx = inside_idx_t

        # 如果窗口内的值大于最少值，则画出选的窗口并更新当前窗口y值；否则不画窗口，也不更新窗口y值
        if len(inside_idx) > min_p:
            cv2.rectangle(img_RGB, win_low[num], win_up[num], (0, 255, 0), 2)
            win_y = win_low[num][1]
        # 返回窗口y值，3通道彩图和非零像素在valid_x中的下标
        return win_y, img_RGB, inside_idx

    # 先进的滑动窗口函数，使用上述5窗口函数
    def adv_slide_window(self, img, win_num=9, win_wr=0.07, min_p=150):
        # time1 = time.time()
        # 二值图，滑动窗口数量，窗口宽度与图片宽度比，窗口包含最少像素点的个数

        '''根据直方图获得初始窗口位置'''
        # 将二值图只保留一半高度用于获得初始直方图
        hist0 = np.sum(img[int(img.shape[0] / 3):, :], axis=0)
        # 分别找左右2部分峰值的x坐标
        midx = int(hist0.shape[0] / 2)
        # 取最大值下标
        # 此处应增加try except解决没有的情况！！！！
        l_x_base = np.argmax(hist0[:midx])
        r_x_base = midx + np.argmax(hist0[midx:])
        # 将灰度图转为RGB 3通道图像
        img_t = np.copy(img)
        img_RGB = np.dstack((img_t, img_t, img_t))

        # 窗口的高度和窗口的宽度
        win_h = int(img.shape[0] / win_num)
        win_w = int(img.shape[1] * win_wr)

        # 获得图片矩阵中非0值的下标
        nonzero = img.nonzero()
        valid_y = np.array(nonzero[0])
        valid_x = np.array(nonzero[1])

        # 将当前左右x下标初始化为 基础x下标
        l_x_current = l_x_base
        r_x_current = r_x_base
        # 初始化起始窗口的高度
        win_l_y = img.shape[0] + win_h
        win_r_y = img.shape[0] + win_h
        # 创建左右2线的xy下标list，最终保存的滑动窗口检测到的像素下标
        left_lane_indx = []
        left_lane_indy = []
        right_lane_indx = []
        right_lane_indy = []

        # 滑动窗口，执行win_num次
        for window in range(win_num):
            # 更新当前窗口位置：左右2个
            win_l_y, img_RGB, l_inside_idx = self.update_win5(img_RGB, win_h, win_w, win_l_y, l_x_current, valid_x, valid_y,
                                                         min_p)
            win_r_y, img_RGB, r_inside_idx = self.update_win5(img_RGB, win_h, win_w, win_r_y, r_x_current, valid_x, valid_y,
                                                         min_p)
            # cv2.imshow('1', img_RGB)
            # cv2.waitKey(0)
            # 获得当前有效像素的xy下标
            l_x_inside = valid_x[l_inside_idx]
            l_y_inside = valid_y[l_inside_idx]
            r_x_inside = valid_x[r_inside_idx]
            r_y_inside = valid_y[r_inside_idx]

            # 更新有效像素的xy下标：因为如果不将之前窗口包含的像素从有效像素（有效像素指的是二值为255的像素）中删除，遇到间隙窗口将持续保持在同一区域附近
            # 将二值图中找到的有效像素设置为0，再取非零值获得新的有效像素
            img_t[l_y_inside, l_x_inside] = 0
            img_t[r_y_inside, r_x_inside] = 0
            nonzero = img_t.nonzero()
            valid_y = np.array(nonzero[0])
            valid_x = np.array(nonzero[1])

            # 合并左右两侧有效点x，y下标list
            left_lane_indx.append(l_x_inside)
            left_lane_indy.append(l_y_inside)
            right_lane_indx.append(r_x_inside)
            right_lane_indy.append(r_y_inside)

            # 如果左侧窗口内找到大于最少点，则取平均并更新下标；如果少于保留当前x下标，并向上平移窗口
            if len(l_x_inside) > min_p:
                l_x_current = int(np.mean(l_x_inside))
            else:
                l_x_current = l_x_current
                win_l_y = win_l_y - win_h
            # 如果右侧窗口内找到大于最少点，则取平均并更新下标；如果少于保留当前x下标，并向上平移窗口
            if len(r_x_inside) > min_p:
                r_x_current = int(np.mean(r_x_inside))
            else:
                r_x_current = r_x_current
                win_r_y = win_r_y - win_h

        # time2 = time.time()
        # print("耗时", time2 - time1, "s")
        # 连接索引数组
        left_lane_indx = np.concatenate(left_lane_indx)
        left_lane_indy = np.concatenate(left_lane_indy)
        right_lane_indx = np.concatenate(right_lane_indx)
        right_lane_indy = np.concatenate(right_lane_indy)

        # 更新信息，同时判断是否识别到左右车道线
        self.frame_img.update_info(left_lane_indx, left_lane_indy, right_lane_indx, right_lane_indy)

        # 多项式拟合左右2条曲线，获得多项式值
        left_fit = self.frame_img.l_mean_fit
        right_fit = self.frame_img.r_mean_fit

        # 产生x和y值
        ploty = np.linspace(0, img.shape[0] - 1, img_RGB.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # 画出窗口内的像素
        img_RGB[left_lane_indy, left_lane_indx] = [255, 0, 0]
        img_RGB[right_lane_indy, right_lane_indx] = [0, 0, 255]

        # 画出曲线
        right = np.array(tuple(zip(right_fitx, ploty)), np.int32)
        left = np.array(tuple(zip(left_fitx, ploty)), np.int32)
        cv2.polylines(img_RGB, [right], False, (0, 255, 255), thickness=5)
        cv2.polylines(img_RGB, [left], False, (0, 255, 255), thickness=5)

        return left_fitx, right_fitx, ploty, img_RGB

    def sparsity(self, img, l_x, r_x, ploty):
        # 二值图，左拟合x值，右拟合x值，y值
        # 用拟合到的曲线x+-2的区域作为范围，共5像素
        # 判断是否识别到车道线
        if self.frame_img.l_detected:
            img_l_line = np.zeros_like(img)
            l1 = np.array([np.transpose(np.array([l_x - 2, ploty], dtype="int"))])
            l2 = np.transpose(np.array([l_x + 2, ploty], dtype="int"))
            l2 = np.array([np.flipud(l2)])
            l_pix = np.hstack((l1, l2))
            cv2.fillPoly(img_l_line, l_pix, 255)

            combine1 = np.zeros_like(img)
            combine1[(img == 255) & (img_l_line == 255)] = 255
            # cv2.imshow("test",img)
            # cv2.imshow("test1", combine1)
            # cv2.waitKey(0)
            # 计算稀疏度
            l_sparsity = 1.0 - np.count_nonzero(combine1) / (5 * len(ploty))
            # True虚线 False实线
            # l_sp = (True, l_sparsity) if l_sparsity > max_sp else (False, l_sparsity)
        else:
            # False不稀疏-实线
            # l_sp = (False, 0)
            l_sparsity = 0

        if self.frame_img.r_detected:
            img_r_line = np.zeros_like(img)
            r1 = np.array([np.transpose(np.array([r_x - 2, ploty], dtype="int"))])
            r2 = np.transpose(np.array([r_x + 2, ploty], dtype="int"))
            r2 = np.array([np.flipud(r2)])
            r_pix = np.hstack((r1, r2))
            cv2.fillPoly(img_r_line, r_pix, 255)

            combine2 = np.zeros_like(img)
            combine2[(img == 255) & (img_r_line == 255)] = 255
            # 计算稀疏度
            r_sparsity = 1.0 - np.count_nonzero(combine2) / (5 * len(ploty))
            # r_sp = (True, r_sparsity) if r_sparsity > max_sp else (False, r_sparsity)
        else:
            # False不稀疏-实线
            # r_sp = (False, 0)
            r_sparsity = 0

        return l_sparsity, r_sparsity

    def cal_curve(self, l_indx, r_indx, y, img_bin):
        # 俯视图，左车道线x坐标，右车道线x坐标 和 多项式y坐标
        # x方向上1个像素多少米；y方向上1个像素多少米；
        xm_for_1pix = 3.7 / 720
        ym_for_1pix = 30.5 / 720
        # 最高点
        y_max = np.max(y)

        # 拟合实际情况的曲线
        left_fit = np.polyfit(y * ym_for_1pix, l_indx * xm_for_1pix, 2)
        right_fit = np.polyfit(y * ym_for_1pix, r_indx * xm_for_1pix, 2)
        # 计算曲率半径
        left_cur = ((1 + (2 * left_fit[0] * y_max * ym_for_1pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit[0])
        right_cur = ((1 + (2 * right_fit[0] * y_max * ym_for_1pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit[0])

        # 车辆位置
        car_pos = img_bin.shape[1] / 2
        lane_center_pos = (l_indx[-1] + r_indx[-1]) / 2
        dev = car_pos - lane_center_pos
        deviation = dev * xm_for_1pix

        # 返回平均曲率半径，平均参数a和偏离程度
        return (np.mean([left_cur, right_cur]), np.mean([left_fit[0], right_fit[0]])), deviation

    # 显示信息
    def show_info(self, img, img_bin, l_x, r_x, ploty, max_sp=0.6):
        # 原图像，二值图，左车道线x坐标，右车道线x坐标 和 多项式y坐标

        # 初始化图像
        img_RGB = np.zeros_like(img)
        img_RGB2 = np.copy(img)
        # 将x和y合并后，矩阵转置，使得每行是【x,y】
        l1 = np.transpose(np.array([l_x, ploty], dtype="int"))
        r1 = np.transpose(np.array([r_x, ploty], dtype="int"))
        left_pix = np.array([l1])
        # 将矩阵反转1st行→n行，2nd行→n-1行；这样画图
        right_pix = np.array([np.flipud(r1)])
        # 矩阵合并：2个(1,450,2)变为(1,900,2)
        pix = np.hstack((left_pix, right_pix))

        # 填充区域
        cv2.fillPoly(img_RGB, pix, (106, 90, 205))
        lane_area = self.re_perspective_img(img_RGB)
        img_add = cv2.addWeighted(img, 1, lane_area, 0.7, 0)
        # img_add = cv2.resize(img_add, (1200, 750)) #图像大小不能固定在yolo检测中

        # 计算曲率半径和车辆偏离度
        radius_param, deviation = self.cal_curve(l_x, r_x, ploty, img_bin)
        # 计算偏离度
        x_final = np.mean([l_x[0], r_x[0]])
        x_init = np.mean([l_x[-1], r_x[-1]])
        degree = (x_final - x_init) / x_init
        # 更新信息
        radio_mean, degree_mean = self.frame_img.update_laneInfo(radius_param[0], degree)
        # 透视变化调整，此处需要调整*************************************************************
        if abs(degree_mean) <= 0.075:
            cv2.putText(img_add, 'Go Straight!', (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 0), 2)
        elif degree_mean < 0:
            cv2.putText(img_add, 'Turn Left!', (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0),
                        2)
        else:
            cv2.putText(img_add, 'Turn Right!', (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 0), 2)

        # 显示信息
        cv2.putText(img_add, 'Radius of curvature: {0:5.0f} m'.format(radio_mean), (450, 650),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        cv2.putText(img_add, 'Deviation of car: {0:.4f} m'.format(deviation), (470, 700),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        # 计算稀疏度
        # 平均曲线
        l_sp_r, r_sp_r = self.sparsity(img_bin, l_x, r_x, ploty)
        # 当前曲线
        l_fit = self.frame_img.l_current_fit
        r_fit = self.frame_img.r_current_fit
        l_fitx = l_fit[0] * ploty ** 2 + l_fit[1] * ploty + l_fit[2]
        r_fitx = r_fit[0] * ploty ** 2 + r_fit[1] * ploty + r_fit[2]
        l_sp_c, r_sp_c = self.sparsity(img_bin, l_fitx, r_fitx, ploty)
        l_sp = np.min([l_sp_r, l_sp_c])
        r_sp = np.min([r_sp_r, r_sp_c])
        # 更新frame的稀疏度
        l_sp_m, r_sp_m = self.frame_img.update_sparsity(l_sp, r_sp)
        l_sp = (True, l_sp_m) if l_sp_m > max_sp else (False, l_sp_m)
        r_sp = (True, r_sp_m) if r_sp_m > max_sp else (False, r_sp_m)
        # 显示是否可以变道
        if l_sp[0]:
            cv2.putText(img_add, 'Can change left!', (450, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            print(l_sp)
        if r_sp[0]:
            cv2.putText(img_add, 'Can change right!', (650, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            # print(r_sp)

        return img_add

    # 图像合并
    def merge_img(self, img_add, img_per, img_filter, img_slide):
        # 合并后的车道识别图像，透视变换彩图，透视变换二值图，滑动窗口处理图
        # 图像分为左侧(1200，750)车道区域与信息，右侧3部分小图（400，250），分别显示剩余3图
        final_img = np.zeros((750, 1600, 3), dtype=np.uint8)
        final_img[0:750, 0:1200, :] = cv2.resize(img_add, (1200, 750))

        # 右侧图片
        final_img[0:250, 1200:1600, :] = cv2.resize(img_per, (400, 250))
        # 图片，内容，位置，字体，大小，颜色和粗细
        cv2.putText(final_img, 'Perspective img', (1300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        gray_image = cv2.cvtColor(img_filter, cv2.COLOR_GRAY2RGB)
        final_img[250:500, 1200:1600, :] = cv2.resize(gray_image, (400, 250))
        cv2.putText(final_img, 'Perspective & Filter', (1280, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        final_img[500:750, 1200:1600, :] = cv2.resize(img_slide, (400, 250))
        cv2.putText(final_img, 'Detected Lanes', (1300, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        return final_img

    # 最终图像处理函数：将上述函数合并
    def process_img(self, img0):
        # 原图
        time1 = time.time()
        pre = self.preprocess(img0)
        time2 = time.time()
        img_warped_bin, img_warped_color = self.perspective_img(pre, img0)
        time3 = time.time()
        # l_x, r_x, ploty, img_slide, l_sp, r_lsp=slide_window(img_warped_bin)
        l_x, r_x, ploty, img_slide = self.adv_slide_window(img_warped_bin)
        time4 = time.time()
        img_processed = self.show_info(img0, img_warped_bin, l_x, r_x, ploty)
        time5 = time.time()
        # cv2.imshow('frame', img_processed)
        result = self.merge_img(img_processed, img_warped_color, img_warped_bin, img_slide)
        time6 = time.time()
        # print("耗时"," 1: ",time2-time1," 2: ",time3-time2," 3: ",time4-time3," 4: ",time5-time4," 5: ",time6-time5)
        return result

    def detection(self, img0):
        pre = self.preprocess(img0)
        img_warped_bin, img_warped_color = self.perspective_img(pre, img0)
        l_x, r_x, ploty, img_slide = self.adv_slide_window(img_warped_bin)
        img_processed = self.show_info(img0, img_warped_bin, l_x, r_x, ploty)
        return img_processed

# 滑动窗口一般方法
# 更新窗口函数
def update_win(img, window, win_h, win_w, l_x_current, r_x_current):
    # 二值图，当前窗口数，窗口高度，窗口宽度，当前左侧车道x下标，当前右侧车道x下标
    # 更新当前窗口位置
    win_y_low = img.shape[0] - (window + 1) * win_h
    win_y_up = img.shape[0] - window * win_h
    win_left_low = (l_x_current - win_w, win_y_low)
    win_left_up = (l_x_current + win_w, win_y_up)
    win_right_low = (r_x_current - win_w, win_y_low)
    win_right_up = (r_x_current + win_w, win_y_up)
    return win_y_low, win_y_up, win_left_low, win_left_up, win_right_low, win_right_up


# 滑动窗口函数
def slide_window(img, win_num=9, win_wr=0.06, min_p=50):
    # 二值图，滑动窗口数量，窗口宽度与图片宽度比，窗口包含最少像素点的个数

    '''根据直方图获得初始窗口位置'''
    # 将二值图只保留一半高度用于获得初始直方图
    hist0 = np.sum(img[int(img.shape[0] / 3):, :], axis=0)
    # 分别找左右2部分峰值的x坐标
    midx = int(hist0.shape[0] / 2)
    # 取最大值下标
    # 此处应增加try except解决没有的情况！！！！
    l_x_base = np.argmax(hist0[:midx])
    r_x_base = midx + np.argmax(hist0[midx:])
    # 将灰度图转为RGB 3通道图像
    img_t = np.copy(img)
    img_RGB = np.dstack((img_t, img_t, img_t))

    # 窗口的高度
    win_h = int(img.shape[0] / win_num)
    # 窗口的宽度
    win_w = int(img.shape[1] * win_wr)

    # 获得图片矩阵中非0值的下标
    nonzero = img.nonzero()
    valid_y = np.array(nonzero[0])
    valid_x = np.array(nonzero[1])
    # print(valid_x)
    valid_pix = np.dstack((valid_x, valid_y))
    # 将当前左右下标初始化为基础
    l_x_current = l_x_base
    r_x_current = r_x_base
    # 创建左右2线最终保存的点下标
    left_lane_indx = []
    left_lane_indy = []
    right_lane_indx = []
    right_lane_indy = []

    # 滑动窗口，执行win_num次
    for window in range(win_num):
        # 更新当前窗口位置
        win_y_low, win_y_up, win_left_low, win_left_up, win_right_low, win_right_up = update_win(img, window, win_h,
                                                                                                 win_w, l_x_current,
                                                                                                 r_x_current)
        # 获得在窗口内的像素值x，y坐标
        mask = (valid_y >= win_y_low) & (valid_y < win_y_up) & (valid_x >= win_left_low[0]) & (valid_x < win_left_up[0])
        mask_inside = np.zeros_like(mask, dtype=np.uint8)
        mask_inside[mask] = 1
        l_x_inside = valid_x[mask_inside.nonzero()[0]]
        l_y_inside = valid_y[mask_inside.nonzero()[0]]

        mask = ((valid_y >= win_y_low) & (valid_y < win_y_up) & (valid_x >= win_right_low[0]) & (
                valid_x < win_right_up[0]))
        mask_inside = np.zeros_like(mask, dtype=np.uint8)
        mask_inside[mask] = 1
        r_x_inside = valid_x[mask_inside.nonzero()[0]]
        r_y_inside = valid_y[mask_inside.nonzero()[0]]

        # 合并
        left_lane_indx.append(l_x_inside)
        left_lane_indy.append(l_y_inside)
        right_lane_indx.append(r_x_inside)
        right_lane_indy.append(r_y_inside)

        cv2.rectangle(img_RGB, win_left_low, win_left_up, (0, 255, 0), 2)
        cv2.rectangle(img_RGB, win_right_low, win_right_up, (0, 255, 0), 2)

        # 如果窗口内找到大于最少点，则取平均并更新下标
        if len(l_x_inside) > min_p:
            l_x_current = int(np.mean(l_x_inside))
        if len(r_x_inside) > min_p:
            r_x_current = int(np.mean(r_x_inside))

    # 连接索引数组
    left_lane_indx = np.concatenate(left_lane_indx)
    left_lane_indy = np.concatenate(left_lane_indy)
    right_lane_indx = np.concatenate(right_lane_indx)
    right_lane_indy = np.concatenate(right_lane_indy)

    # 多项式拟合左右2条曲线
    left_fit = np.polyfit(left_lane_indy, left_lane_indx, 2)
    right_fit = np.polyfit(right_lane_indy, right_lane_indx, 2)

    # 产生x和y值
    ploty = np.linspace(0, img.shape[0] - 1, img_RGB.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # 画出点
    img_RGB[left_lane_indy, left_lane_indx] = [255, 0, 0]
    img_RGB[right_lane_indy, right_lane_indx] = [0, 0, 255]

    # 画出曲线
    right = np.array(tuple(zip(right_fitx, ploty)), np.int32)
    left = np.array(tuple(zip(left_fitx, ploty)), np.int32)
    cv2.polylines(img_RGB, [right], False, (0, 255, 255), thickness=5)
    cv2.polylines(img_RGB, [left], False, (0, 255, 255), thickness=5)

    return left_fitx, right_fitx, ploty, img_RGB, (False, 0), (False, 0)


if __name__ == "__main__":
    capture = cv2.VideoCapture('./data/video/Input_Video.mp4')
    # print("帧率",capture.get(5))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_cat = cv2.VideoWriter("./data/video/save1.mp4", fourcc, 25, (1600, 750))  # 保存位置/格式
    # out1 = cv2.VideoWriter("gray.mp4", fourcc, 15, (800, 450), 0)
    # frame_img = Frame()
    lane_detect=LANE_DETECTION()
    while True:
        ret, frame = capture.read()
        c = cv2.waitKey(5)
        if c == 27 or not ret:
            break
        time1 = time.time()
        frame1 = lane_detect.process_img(frame)
        out_cat.write(frame1)  # 保存视频
        # out1.write(frame2)
        cv2.imshow('frame', frame1)
        time_now = time.time()
        print("耗时1",time_now-time1,"s")
    out_cat.release()
    # out1.release()