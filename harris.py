import numpy as np
import cv2
from matplotlib import pyplot as plt

from function import *

def FeaturePointdetection(org_img, com_img):
    block_size = 2
    kernel_size = 5
    k = 0.02

    #特徴点検出用のグレースケール画像
    gray_org_img = cv2.cvtColor(org_img, cv2.COLOR_RGB2GRAY)
    gray_com_img = cv2.cvtColor(com_img, cv2.COLOR_RGB2GRAY)

    #harris法による特徴点検出
    org_res = cv2.cornerHarris(gray_org_img, block_size, kernel_size, k)
    com_res = cv2.cornerHarris(gray_com_img, block_size, kernel_size, k)

    #特徴点の画素の色を赤にする
    org_img[org_res>0.01*org_res.max()] = (255, 0, 0)
    com_img[com_res>0.01*com_res.max()] = (255, 0, 0)

    return org_img, com_img

def Comparison(org_img, com_img):
    org_height, org_width, channels = org_img.shape
    com_height, com_width, chaneels = com_img.shape
    coordinate = []
    x = 0
    sep_num = 4
    slice_num = 4
    c_h,c_w = org_height // sep_num, org_width // sep_num
    dh,dw = c_h // slice_num, c_w // slice_num
    start_h, start_w = 0,0

    new_org_img = []
    new_com_img = []

    #各画像を分割した各ブロック内の特徴点の数用配列
    org_point_num = []
    com_point_num = []

    org_point, com_point = 0,0
    shape = []
    n = 1

    for i in range(sep_num * slice_num):
        for j in range(sep_num * slice_num):
            cutted_org_img = org_img[start_h:start_h + c_h, start_w:start_w + c_w]
            cutted_com_img = com_img[start_h:start_h + c_h, start_w:start_w + c_w]
            new_org_img.append(cutted_org_img)
            new_com_img.append(cutted_com_img)
            #print(dh,dw)
            #print('j = {}'.format(j))
            for p in range(cutted_org_img.shape[0]):
                #print('p = {}'.format(p))
                for q in range(cutted_org_img.shape[1]):
                    #print('q = {}'.format(q))
                    #画素の色が赤か判定
                    if np.all(cutted_org_img[p][q] == [255,0,0]):
                        org_point += 1
                    if np.all(cutted_com_img[p][q] == [255,0,0]):
                        com_point += 1
            org_point_num.append(org_point)
            com_point_num.append(com_point)
            if abs(org_point - com_point) > 40:
                shape.append([start_h, start_w, start_h + c_h, start_w + c_w])
                #print('i = {0}, j = {1}'.format(i,j))
            org_point, com_point = 0,0
            start_w += dw

        start_h += dh
        start_w = 0

    if not shape:
        after_img = []
    else:
        for i in range(len(shape)):
            after_img = cv2.rectangle(com_img, (shape[i][1], shape[i][0]), (shape[i][3], shape[i][2]), (255,0,0))

    return after_img

    def image_show(org_img, com_img):
        plt.figure(figsize = (16,8))
        plt.subplot(121)
        plt.title('Original_image')
        plt.imshow(org_img)

        plt.subplot(122)
        plt.title('Comparison_image')
        plt.imshow(com_img)


if __name__ == "__main__":
    org_img, com_img = Read_img()

    org_show_img = cv2.imread('img/test3.png')
    org_show_img= cv2.cvtColor(org_show_img, cv2.COLOR_RGB2BGR)

    #image_show(org_show_img, com_img)

    point_org_img, point_com_img = FeaturePointdetection(org_img, com_img)

    after_img = Comparison(point_org_img, point_com_img)
    plt.figure(figsize = (8,4))
    plt.title('Comparison result (harris)')
    plt.imshow(after_img)

    plt.show()
