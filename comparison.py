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
    #画像をn×nに分割
    #今回は n = 4
    slice_num = 4
    dh = org_height // slice_num
    dw = org_width // slice_num
    start_h, start_w = 0,0

    new_org_img = []
    new_com_img = []

    #各画像を分割した各ブロック内の特徴点の数用配列
    org_point_num = []
    com_point_num = []

    org_point, com_point = 0,0
    shape = []
    n = 1

    for i in range(slice_num):
        #print('unko')
        for j in range(slice_num):
            cutted_org_img = org_img[start_h:start_h + dh, start_w:start_w + dw]
            cutted_com_img = com_img[start_h:start_h + dh, start_w:start_w + dw]
            new_org_img.append(cutted_org_img)
            new_com_img.append(cutted_com_img)
            #print(dh,dw)
            #print('j = {}'.format(j))
            for p in range(dh):
                #print('p = {}'.format(p))
                for q in range(dw):
                    #print('q = {}'.format(q))
                    #画素の色が赤か判定
                    if np.all(cutted_org_img[p][q] == [255,0,0]):
                        org_point += 1
                    if np.all(cutted_com_img[p][q] == [255,0,0]):
                        com_point += 1
            org_point_num.append(org_point)
            com_point_num.append(com_point)
            if abs(org_point - com_point) > 40:
                shape.append([start_h, start_w, start_h + dh, start_w + dw])
                print('i = {0}, j = {1}'.format(i,j))
            org_point, com_point = 0,0
            start_w += dw

        start_h += dh
        start_w = 0

    for i in range(len(shape)):
        after_img = cv2.rectangle(com_img, (shape[i][1], shape[i][0]), (shape[i][3], shape[i][2]), (255,0,0))

    return after_img

if __name__ == "__main__":
    org_img, com_img = Read_img()

    print('unko')

    org_show_img = cv2.imread('img/test3.png')
    org_show_img= cv2.cvtColor(org_show_img, cv2.COLOR_RGB2BGR)

    plt.figure(figsize = (16,8))
    plt.subplot(221)
    plt.title('Original_image')
    plt.imshow(org_show_img)

    plt.subplot(222)
    plt.title('Comparison_image')
    plt.imshow(com_img)

    point_org_img, point_com_img = FeaturePointdetection(org_img, com_img)

    after_img = Comparison(point_org_img, point_com_img)

    plt.subplot(223)
    plt.imshow(point_org_img)

    plt.subplot(224)
    plt.imshow(after_img)

    '''
    org_img= cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    com_img= cv2.cvtColor(com_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('img/result_1.png', org_img)
    cv2.imwrite('img/result_2.png', com_img)
    '''

    plt.show()
