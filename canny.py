import numpy as np
import cv2
from matplotlib import pyplot as plt

from comparison import Read_img

if __name__ == "__main__":
    org_img, com_img = Read_img()

    print(type(org_img))
    print(type(com_img))

    fig = plt.figure(figsize = (16,9))

    plt.subplot(221)
    plt.imshow(org_img)

    plt.subplot(222)
    plt.imshow(com_img)

    gray_org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    gray_com_img = cv2.cvtColor(com_img, cv2.COLOR_BGR2GRAY)

    min_val = 100
    max_val = 200
    org_edge_img = cv2.Canny(gray_org_img, min_val, max_val)
    com_edge_img = cv2.Canny(gray_com_img, min_val, max_val)

    org_height, org_width, channels = org_img.shape
    com_height, com_width, chaneels = com_img.shape

    coordinate = []
    x = 0
    #画像をn×nに分割
    #今回は n = 4
    slice_num = 5
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
        for j in range(slice_num):
            cutted_org_img = org_edge_img[start_h:start_h + dh, start_w:start_w + dw]
            cutted_com_img = com_edge_img[start_h:start_h + dh, start_w:start_w + dw]
            new_org_img.append(cutted_org_img)
            new_com_img.append(cutted_com_img)
            #print(dh,dw)
            #print('j = {}'.format(j))
            for p in range(dh):
                #print('p = {}'.format(p))
                for q in range(dw):
                    #print('q = {}'.format(q))
                    if np.all(cutted_org_img[p][q] == [255,255,255]):
                        org_point += 1
                    if np.all(cutted_com_img[p][q] == [255,255,255]):
                        com_point += 1
            org_point_num.append(org_point)
            com_point_num.append(com_point)
            if abs(org_point - com_point) > 300:
                shape.append([start_h, start_w, start_h + dh, start_w + dw])
                #print('i = {0}, j = {1}'.format(i,j))
            org_point, com_point = 0,0
            start_w += dw

        start_h += dh
        start_w = 0

    #print(org_point_num)
    #print(com_point_num)
    '''
    for i in range(len(org_point_num)):
        print('{}, diff = '.format(i + 1), end = " ")
        print(abs(org_point_num[i] - com_point_num[i]))
        '''
    for i in range(len(shape)):
        after_img = cv2.rectangle(com_img, (shape[i][1], shape[i][0]), (shape[i][3], shape[i][2]), (255,0,0))

    plt.subplot(223)
    plt.imshow(org_edge_img, cmap="gray")
    plt.subplot(224)
    plt.imshow(com_edge_img, cmap="gray")

    print(shape)
    print(after_img)

    plt.figure(figsize = (8,4))
    plt.imshow(after_img)



    plt.show()
