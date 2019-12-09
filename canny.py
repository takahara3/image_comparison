import numpy as np
import cv2
from matplotlib import pyplot as plt

from functions import *

if __name__ == "__main__":
    org_img_path = 'img/warped_img.png'
    com_img_path = 'img/test5.png'
    org_img, com_img = Read_img(org_img_path, com_img_path)

    gray_org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    gray_com_img = cv2.cvtColor(com_img, cv2.COLOR_BGR2GRAY)

    min_val = 100
    max_val = 200
    org_edge_img = cv2.Canny(gray_org_img, min_val, max_val)
    com_edge_img = cv2.Canny(gray_com_img, min_val, max_val)

    #特徴点の画素のRGB値
    color = [255,255,255]
    #判定に使う特徴点の差
    diff = 300

    #特徴点の比較，矩形ごとの特徴点の取得
    shape, org_point_num, com_point_num = Comparison(org_edge_img, com_edge_img, color, diff, True)

    #矩形を描画
    img_conversion(com_img, shape)
    #グラフの作成
    draw_graph(org_point_num, com_point_num)
