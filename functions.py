import numpy as np
import cv2
from matplotlib import pyplot as plt

from class import *

#比較画像の読み込み
def Read_img(org_img_path, com_img_path):
    org_img = cv2.imread(org_img_path)
    com_img = cv2.imread(com_img_path)
    #色変換
    org_img= cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)
    com_img= cv2.cvtColor(com_img, cv2.COLOR_RGB2BGR)
    return org_img, com_img

#画像表示
def image_show(org_img, com_img):
    plt.figure(figsize = (10,8))
    plt.subplot(121)
    plt.title('Original_image')
    plt.imshow(org_img)

    plt.subplot(122)
    plt.title('Comparison_image')
    plt.imshow(com_img)

#矩形描画
def draw_rect(img,shape):
    for i in range(len(shape)):
        draw_img = cv2.rectangle(img,(shape[i][1], shape[i][0]), (shape[i][3], shape[i][2]), (255,0,0))

    return draw_img

#矩形を描画した画像の生成
def img_conversion(img,shape):
    if not shape:
        print('Failure')
    else:
        after_img = draw_rect(img, shape)
        plt.figure(figsize = (8,4))
        plt.title('Comparison result')
        plt.imshow(after_img)
        plt.show()

#特徴点のグラフ作成
def draw_graph(org_point_num, com_point_num):
    diff = []

    plt.figure(figsize=(10,4))
    plt.subplot(131)
    plt.title('Original image point')
    x = [i for i in range(len(org_point_num))]
    plt.bar(x, org_point_num)

    plt.subplot(132)
    plt.title('Comparison image')
    x = [i for i in range(len(com_point_num))]
    plt.bar(x, com_point_num)

    for i in range(len(org_point_num)):
        diff.append(abs(org_point_num[i] - com_point_num[i]))

    plt.subplot(133)
    plt.title('diff')
    x = [i for i in range(len(diff))]
    plt.bar(x,diff)

    plt.show()

#画像中の特徴点比較
def Comparison(org_img, com_img, color, diff, point=False):
    '''
    org_height, org_width = org_img.shape[0], org_img.shape[1]
    com_height, com_width = com_img.shape[0], com_img.shape[1]
    x = 0
    sep_num = 3
    slice_num = 3
    c_h,c_w = org_height // sep_num, org_width // sep_num
    dh,dw = c_h // slice_num, c_w // slice_num
    start_h, start_w = 0,0
    '''

    param = Param()

    new_org_img = []
    new_com_img = []

    #各画像を分割した各ブロック内の特徴点の数用配列
    org_point_num = []
    com_point_num = []

    org_point, com_point = 0,0
    shape = []
    n = 1

    for i in range(param.sep_num * param.slice_num):
        for j in range(param.sep_num * param.slice_num):
            cutted_org_img = org_img[param.start_h:param.start_h + param.c_h, param.start_w:param.start_w + param.c_w]
            cutted_com_img = com_img[param.start_h:param.start_h + param.c_h, param.start_w:param.start_w + param.c_w]
            new_org_img.append(cutted_org_img)
            new_com_img.append(cutted_com_img)
            for p in range(cutted_org_img.shape[0]):
                for q in range(cutted_org_img.shape[1]):
                    if np.all(cutted_org_img[p][q] == color):
                        org_point += 1
                    if np.all(cutted_com_img[p][q] == color):
                        com_point += 1
            org_point_num.append(org_point)
            com_point_num.append(com_point)
            if abs(org_point - com_point) > diff:
                shape.append([param.start_h, param.start_w, param.start_h + param.c_h, param.start_w + param.c_w])
            org_point, com_point = 0,0
            param.start_w += param.dw

        param.start_h += param.dh
        param.start_w = 0

    if point == True:
        return shape, org_point_num, com_point_num
    else:
        return shape

def HarrisFeaturePointdetection(org_img, com_img):
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
