import numpy as np
import cv2
from matplotlib import pyplot as plt

from functions import *

if __name__ == "__main__":
    org_img_path = 'img/test3.png'
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

    #矩形ごとの特徴点を比較，差分の大きい矩形の座標＆各矩形の登頂点数を取得
    shape, org_point_num, com_point_num = Comparison(org_edge_img, com_edge_img, color, diff, True)
    print(len(shape))

    #全ての矩形の共通部分の矩形を取得
    x = seek_intersect(shape)
    print(x)

    img_1 = com_img.copy()
    img_2 = com_img.copy()
    print(type(img_1))

    #矩形を描画
    cover_img = draw_common_rect(img_1, x)
    after_img = draw_rect(img_2, shape)

    two_images_show(after_img, cover_img)

    #グラフの作成
    #draw_graph(org_point_num, com_point_num)
