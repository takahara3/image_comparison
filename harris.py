import numpy as np
import cv2
from matplotlib import pyplot as plt

from functions import *

if __name__ == "__main__":
    org_img_path = "img/origin_image.png"
    com_img_path = "img/comparison_image.png"
    org_img, com_img = Read_img(org_img_path, com_img_path)

    org_show_img = cv2.imread('img/test3.png')
    org_show_img= cv2.cvtColor(org_show_img, cv2.COLOR_RGB2BGR)

    color = [255,0,0]
    diff = 40

    point_org_img, point_com_img = HarrisFeaturePointdetection(org_img, com_img)

    shape = Comparison(point_org_img, point_com_img, color, diff)
    img_conversion(com_img, shape)
