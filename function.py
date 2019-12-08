import numpy as np
import cv2
from matplotlib import pyplot as plt

#比較画像の読み込み
def Read_img():
    org_img = cv2.imread('img/warped_img.png')
    com_img = cv2.imread('img/test5.png')
    #色変換
    org_img= cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)
    com_img= cv2.cvtColor(com_img, cv2.COLOR_RGB2BGR)
    return org_img, com_img
