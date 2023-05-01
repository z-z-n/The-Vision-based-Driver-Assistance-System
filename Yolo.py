import argparse
import os
import platform
import sys
import torch
import numpy as np

from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import letterbox
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


class Yolov5:
    def __init__(self,
                 device='',
                 weights='yolov5s.pt',
                 data='data/coco128.yaml'
                 ):
        '''
        self.FILE = Path(__file__).resolve()
        self.ROOT = self.FILE.parents[0]  # YOLOv5 root directory
        if str(self.ROOT) not in sys.path:
            sys.path.append(str(self.ROOT))  # add ROOT to PATH
        self.ROOT = Path(os.path.relpath(self.ROOT, Path.cwd()))  # relative
        '''
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        imgsz = (640, 640)  # inference size (height, width)
        # 模型初始化
        device = select_device(device)
        print('打开ing')
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Run inference
        bs = 1  # batch_size
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

        self.img_size = 640
        self.stride = stride
        self.names = names
        self.auto = pt
        self.model = model
        self.conf_thres = 0.7  # 置信度默认0.25
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # 最多检测目标
        self.classes = None  # 是否只保留特定类别: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # 进行nms是否也去除不同类别间的框class-agnostic NMS
        self.line_thickness = 3  # 框线条宽度


    def detect_object(self, im0,view_result=False):
        # 图像预处理1-缩放
        im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        # 图像预处理2
        im = torch.from_numpy(im).to(self.model.device)     # 图像格式转换
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0归一化
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # 模型推理
        # 图片进行前向推理
        pred = self.model(im, augment=False, visualize=False) # augment和visualize都是参数，预测是否采用数据增强、虚拟化特征？
        # nms除去多余的框
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        # 后续处理
        det=pred[0]  # 预测结果
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
        xyxy_list = []
        conf_list = []
        class_id_list = []
        if len(det):
            # 将坐标信息恢复到原始图像的尺寸
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):  # 框位置，置信度和类别id
                c = int(cls)  # integer class
                # 保存当前信息
                xyxy_list.append(xyxy)
                conf_list.append(conf)
                class_id_list.append(c)
                # 标签内容
                label = f'{self.names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
                '''
                dist = lane_detect.distance(xyxy)
                annotator.box_label2(xyxy, dist, color=colors(c, True))
                '''
        im0 = annotator.result()
        if view_result:
            # Stream results
            # 视频显示
            cv2.imshow('frame', im0)
            cv2.waitKey(1)  # 1 millisecond

        return im0, xyxy_list, conf_list, class_id_list

if __name__ == '__main__':
    model=Yolov5()