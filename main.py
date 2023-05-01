import cv2
import numpy as np
import time

from lane import LANE_DETECTION
from Yolo import Yolov5

if __name__ == '__main__':
    capture = cv2.VideoCapture('./data/video/Input_Video.mp4')
    # print("帧率",capture.get(5))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_cat = cv2.VideoWriter("./runs/merge/r1.mp4", fourcc, 25, (1280, 720))  # 保存位置/格式
    lane_detect = LANE_DETECTION()
    model = Yolov5(weights='yolov5_c3ca.pt')
    while True:
        ret, frame = capture.read()
        c = cv2.waitKey(5)
        if c == 27 or not ret:
            break
        time1 = time.time()
        img_bin, lx, rx, pty = lane_detect.detection(frame)
        frame1, xyxy, conf, cls=model.detect_object(frame)
        frame2=lane_detect.show_info(frame1, img_bin, lx, rx, pty)
        out_cat.write(frame2)  # 保存视频
        # out1.write(frame2)
        cv2.imshow('frame', frame2)
        time_now = time.time()
        # print("耗时1", time_now - time1, "s")
    out_cat.release()