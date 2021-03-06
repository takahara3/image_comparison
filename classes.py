import numpy as np
import cv2
from matplotlib import pyplot as plt

class Shape:
    def __init__(self, image):
        self.h, self.w, self.ch = image.shape


class Rect:
    def __init__(self, org_img, com_img):
        self.org_height = org_img.shape[0]
        self.org_width = org_img.shape[1]
        self.com_height = com_img.shape[0]
        self.com_width = com_img.shape[1]
        self.sep_num = 4
        self.slice_num = 10
        self.c_h = self.org_height // self.sep_num
        self.c_w = self.org_width // self.sep_num
        self.dh = self.c_h // self.slice_num
        self.dw = self.c_w // self.slice_num
        self.start_h = 0
        self.start_w = 0
