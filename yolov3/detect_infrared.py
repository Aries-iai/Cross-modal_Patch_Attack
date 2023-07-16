
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from yolov3.models.common import DetectMultiBackend
from yolov3.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov3.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov3.utils.plots import Annotator, colors, save_one_box
from yolov3.utils.torch_utils import select_device, time_sync
from yolov3.utils.augmentations import letterbox
import numpy as np
import torch.nn as nn
import PIL.Image as Image
from torchvision import transforms

device = torch.device("cuda")
inputsize = [416,416]
trans = transforms.Compose([
    transforms.ToTensor(),
])

def load_infrared_model():
    weights = "/workspace/spline_de_attack/last.pt"
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model

def detect_infrared(model,img):
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    H = len(img[0][0])
    W = len(img[0][0][0])
    # img = nn.functional.interpolate(img, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)
    img = img.cuda()
    pred = model(img)
    conf_thres=0.0001 # confidence threshold
    iou_thres=0.45
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)
    if len(pred[0]) == 0:
        return None,0
    left = max(int(pred[0][0][0].item()),0)
    up = max(int(pred[0][0][1].item()),0)
    right = min(int(pred[0][0][2].item()),inputsize[0])
    below = min(int(pred[0][0][3].item()),inputsize[1])
    left = int(left*W/inputsize[0])
    up = int(up*H/inputsize[1])
    right = int(right*W/inputsize[0])
    below = int(below*H/inputsize[1])
    return [left,up,right,below],pred[0][0][4].clone().detach()

