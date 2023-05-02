# Author: Zhining Zhang
# Task: Complete bend detection

import cv2
import numpy as np

'''
图像预处理：1.HLS颜色过滤；2.sobel算子过滤 （2者利用阈值保留并结合）
'''


# HLS 阈值过滤，保留黄色和白色区域
def HLS_filter(img):
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
def sobel_filter(img, thresh_min=20, thresh_max=100):
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


# 预处理函数
def preprocess(img, sthresh_min=20, sthresh_max=100):
    HLS_bin = HLS_filter(img)
    sobel_bin = sobel_filter(img)
    # HLS过滤和sobel的x过滤结合
    combined = np.zeros_like(sobel_bin)
    combined[(HLS_bin == 255) | (sobel_bin == 255)] = 255
    return combined


'''
透视变换：将图片转化为俯视图，保留感兴趣区域
'''


def perspective_img(img, img0, draw=True):
    # 二值图，原彩色图，是否画图的flag
    # 图片大小，opencv读取，[1]是宽度[0]是高度
    img_size = (img.shape[1], img.shape[0])
    dst_size = (800, 450)
    # src：源图像中待测矩形的四点坐标
    # dst：目标图像中矩形的四点坐标
    # src0 = np.array([(0.46, 0.625), (0.54, 0.625), (0.1, 1), (1, 1)], dtype="float32")
    # dst0 = np.array([(0.2, 0), (0.8, 0), (0.2, 1), (0.8, 1)], dtype="float32")
    # dst0 = np.array([(0.3, 0), (0.7, 0), (0.3, 1), (0.7, 1)], dtype="float32")
    src0 = np.array([(0.44, 0.65), (0.57, 0.65), (0.1, 1), (1, 1)], dtype="float32")
    dst0 = np.array([(0.1, 0), (0.9, 0), (0.1, 1), (0.9, 1)], dtype="float32")
    src = np.float32(img_size) * src0
    dst = np.float32(dst_size) * dst0
    R = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, R, dst_size)

    # 是否画出感兴趣区域
    if draw:
        # 感兴趣区域——画多边形
        img1 = np.copy(img0)
        ROI_M = src.astype(np.int32)
        # 交换矩阵第1，2行
        ROI_M[[0, 1], :] = ROI_M[[1, 0], :]
        # print(ROI_M)
        cv2.polylines(img1, [ROI_M], True, (100, 87, 249), 5)
        # 图像大小重置
        img0 = cv2.resize(img1, (800, 450), interpolation=cv2.INTER_AREA)

    return warped, img0

# 逆透视变换（将俯视图转为前视图）
def re_perspective_img(img):
    # 原图
    # 图片大小，opencv读取，[1]是宽度[0]是高度
    img_size = (800, 450)
    dst_size = (img.shape[1], img.shape[0])
    # src：源图像中待测矩形的四点坐标
    # dst：目标图像中矩形的四点坐标
    src0 = np.array([(0.1, 0), (0.9, 0), (0.1, 1), (0.9, 1)], dtype="float32")
    # src0 = np.array([(0.2, 0), (0.8, 0), (0.2, 1), (0.8, 1)], dtype="float32")
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
def slide_window(img, win_num=9, win_wr=0.05, min_p=50):
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
        win_y_low, win_y_up, win_left_low, win_left_up, win_right_low, win_right_up = update_win(img, window, win_h, win_w, l_x_current, r_x_current)
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

    return left_lane_indx, right_lane_indx, img_RGB

# 5窗口方法：根据上一次窗口内有效像素的x均值和窗口的高度，一次划分左，右，左上，中上和右上 共5个窗口
# 该方法有助于当出现急转弯时，左右两条车道线水平延申的情况；也有助于当转弯时左右2车道进入同一侧
def update_win5(img_RGB, win_h, win_w, win_y, x_current, valid_x, valid_y, min_p):
    # 3通道彩图，窗口高度，窗口宽度，之前窗口y高度下标，之前窗口有效像素x下标均值，有效像素x下标list，有效像素y下标list，窗口内最少点的个数
    # 更新当前窗口的xy坐标（每个窗口需要左下和右上2个像素
    # 3种y值
    '''
    # 画5窗口用
    win_y_low = win_y - win_h
    win_y_up = win_y_low - win_h
    win_y_upper = win_y_up - win_h
    '''
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

    # cv2.imshow("5windows", img_RGB)
    # cv2.waitKey(0)
    # 如果窗口内的值大于最少值，则画出选的窗口并更新当前窗口y值；否则不画窗口，也不更新窗口y值
    if len(inside_idx) > min_p:
        cv2.rectangle(img_RGB, win_low[num], win_up[num], (0, 255, 0), 2)
        win_y = win_low[num][1]
    # 返回窗口y值，3通道彩图和非零像素在valid_x中的下标
    return win_y, img_RGB, inside_idx


# 先进的滑动窗口函数，使用上述5窗口函数
def adv_slide_window(img, win_num=9, win_wr=0.07, min_p=150, max_sp=0.2):
    # 二值图，滑动窗口数量，窗口宽度与图片宽度比，窗口包含最少像素点的个数和最大稀疏程度

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
        win_l_y, img_RGB, l_inside_idx = update_win5(img_RGB, win_h, win_w, win_l_y, l_x_current, valid_x, valid_y,
                                                     min_p)
        win_r_y, img_RGB, r_inside_idx = update_win5(img_RGB, win_h, win_w, win_r_y, r_x_current, valid_x, valid_y,
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

    # 连接索引数组
    left_lane_indx = np.concatenate(left_lane_indx)
    left_lane_indy = np.concatenate(left_lane_indy)
    right_lane_indx = np.concatenate(right_lane_indx)
    right_lane_indy = np.concatenate(right_lane_indy)

    # 注意：若有一边没有检测到点，处理****************************************************************
    # 多项式拟合左右2条曲线
    left_fit = np.polyfit(left_lane_indy, left_lane_indx, 2)
    right_fit = np.polyfit(right_lane_indy, right_lane_indx, 2)

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

    # 稠密度分析
    try:
        com1 = np.zeros_like(img)
        com2 = np.zeros_like(img)
        img_l = np.zeros_like(img)
        img_r = np.zeros_like(img)
        y = ploty.astype(int)
        img_l[y, left_fitx.astype(int)] = 255
        img_r[y, right_fitx.astype(int)] = 1
        com1[(img == 255) & (img_l == 255)] = 255
        com2[(img == 255) & (img_r == 1)] = 255
        l_sparsity = 1.0 - np.count_nonzero(com1) / len(y)
        r_sparsity = 1.0 - np.count_nonzero(com2) / len(y)
        # True虚线 False实线
        l_sp = (True, l_sparsity) if l_sparsity > max_sp else (False, l_sparsity)
        r_sp = (True, r_sparsity) if r_sparsity > max_sp else (False, r_sparsity)
        # print("左侧稀疏度", l_sp, "\n右侧稀疏度", r_sp)
    except:
        l_sp = (False, 0)
        r_sp = (False, 0)

    return left_fitx, right_fitx, ploty, img_RGB, l_sp, r_sp

def cal_curve(l_indx, r_indx, y, img_bin):
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
def show_info(img, img_bin, l_x, r_x, ploty, l_sp, r_sp, max_sp=0.6):
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
    cv2.fillPoly(img_RGB, pix, (41,122,255))
    lane_area = re_perspective_img(img_RGB)
    img_add = cv2.addWeighted(img, 1, lane_area, 0.7, 0)
    # img_add = cv2.resize(img_add, (1200, 750))

    # 计算曲率半径和车辆偏离度
    radius_param, deviation = cal_curve(l_x, r_x, ploty, img_bin)
    # 透视变化调整，此处需要调整*************************************************************
    if radius_param[0] > 2400:
        cv2.putText(img_add, 'Go Straight!', (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    elif radius_param[1] < 0:
        cv2.putText(img_add, 'Turn Left!', (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    else:
        cv2.putText(img_add, 'Turn Right!', (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    # 显示信息
    cv2.putText(img_add, 'Radius of curvature: {0:5.0f} m'.format(radius_param[0]), (450, 650),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(img_add, 'Deviation of car: {0:.4f} m'.format(deviation), (470, 700),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    # 显示是否可以变道
    if l_sp[0]:
        cv2.putText(img_add, 'Can change left!', (450, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    if r_sp[0]:
        cv2.putText(img_add, 'Can change right!', (650, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    return img_add


# 最终图像处理函数：将上述函数合并
def process_img(img0):
    # 原图
    pre = preprocess(img0)
    img_warped_bin, img_warped_color = perspective_img(pre, img0)
    # l_x, r_x, ploty, img_slide, l_sp, r_lsp=slide_window(img_warped_bin)
    l_x, r_x, ploty, img_slide, l_sp, r_sp = adv_slide_window(img_warped_bin)
    img_processed = show_info(img0, img_warped_bin, l_x, r_x, ploty, l_sp, r_sp)
    # cv2.imshow('frame', img_processed)
    return img_processed

if __name__ == "__main__":
    capture = cv2.VideoCapture('Input_Video.mp4')
    # print("帧率",capture.get(5))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_cat = cv2.VideoWriter("result1.mp4", fourcc, 25, (1200, 750))  # 保存位置/格式
    while True:
        ret, frame = capture.read()
        c = cv2.waitKey(5)
        if c == 27 or not ret:
            break
        frame1 = process_img(frame)
        out_cat.write(frame1)  # 保存视频
        cv2.imshow('frame', frame1)
    out_cat.release()