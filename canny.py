import numpy as np
import cv2
from matplotlib import pyplot as plt

from functions import *

def has_intersect(rect1,rect2):

    return max(rect1[1], rect2[1]) <= min(rect1[3], rect2[3])\
            and max(rect1[0], rect2[0]) <= min(rect1[2], rect2[2])

def seek_intersect(shape):
    '''
    if has_intersect(rect1, rect2) == True:
        print('rect1 = {}'.format(rect1))
        print('rect2 = {}'.format(rect2))
        x1 = max(rect1[1], rect2[1])
        y1 = max(rect1[0], rect2[0])
        x2 = min(rect1[3], rect2[3])
        y2 = min(rect1[2], rect2[2])
    '''
    rect1, rect2 = shape[0],shape[1]
    for i in range(len(shape)-2):
        if has_intersect(rect1,rect2) == True:
            x1 = max(rect1[1], rect2[1])
            y1 = max(rect1[0], rect2[0])
            x2 = min(rect1[3], rect2[3])
            y2 = min(rect1[2], rect2[2])

            rect1 = [y1,x1,y2,x2]
            rect2 = shape[i+2]

        else:
            return rect1

    return rect1

def drawrect(img,shape):
    draw_img = cv2.rectangle(img,(shape[1], shape[0]), (shape[3], shape[2]), (255,0,0))
    return draw_img

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

    #矩形ごとの特徴点を比較，差分の大きい矩形の座標＆各矩形の登頂点数を取得
    shape, org_point_num, com_point_num = Comparison(org_edge_img, com_edge_img, color, diff, True)
    print(len(shape))

    #全ての矩形の共通部分の矩形を取得
    x = seek_intersect(shape)
    print(x)

    img_1 = com_img.copy()
    img_2 = com_img.copy()

    #矩形を描画
    cover_img = drawrect(img_1, x)
    after_img = draw_rect(img_2, shape)

    '''
    for i,rect in enumerate(shape):
        img = drawrect(com_img,rect)
        image_show(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./result/result_{}.png'.format(i), img)
    '''

    some_images_show(after_img, cover_img)

    #グラフの作成
    #draw_graph(org_point_num, com_point_num)
