import numpy as np
import cv2
from matplotlib import pyplot as plt


class Param:
    def __init__(self, org_img, com_img):
        self.org_height = org_img.shape[0]
        self.org_width = org_img.shape[1]
        self.com_height = com_img.shape[0]
        self.com_width = com_img.shape[1]
        self.sep_num = 3
        self.slice_num = 3
        self.c_h = self.org_height // self.sep_num
        self.c_w = self.org_width // self.sep_num
        self.dh = self.c_h // self.slice_num
        self.dw = self.c_w // self.slice_num
        self.start_h = 0
        self.start_w = 0
