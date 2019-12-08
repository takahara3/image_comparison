import numpy as np
import cv2
from matplotlib import pyplot as plt

from function import Read_img

if __name__ == "__main__":
    org_img, com_img = Read_img()
    plt.figure(figsize = (16,8.5))

    plt.subplot(221)
    plt.imshow(org_img)

    plt.subplot(223)
    plt.imshow(com_img)

    gray_org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    gray_com_img = cv2.cvtColor(com_img, cv2.COLOR_BGR2GRAY)

    min_val = 100
    max_val = 200
    org_edge_img = cv2.Canny(gray_org_img, min_val, max_val)
    com_edge_img = cv2.Canny(gray_com_img, min_val, max_val)

    org_height, org_width, channels = org_img.shape
    com_height, com_width, chaneels = com_img.shape

    sep_num = 4
    slice_num = 10
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
        #print('i = {}'.format(i))
        for j in range(sep_num * slice_num):
            cutted_org_img = org_edge_img[start_h:start_h + c_h, start_w:start_w + c_w]
            cutted_com_img = com_edge_img[start_h:start_h + c_h, start_w:start_w + c_w]
            new_org_img.append(cutted_org_img)
            new_com_img.append(cutted_com_img)
            #print('cutted_img shape = {}'.format(cutted_org_img.shape[0]))
            #print('j = {}'.format(j))
            for p in range(cutted_org_img.shape[0]):
                #print('p = {}'.format(p))
                for q in range(cutted_org_img.shape[1]):
                    #print('q = {}'.format(q))
                    if np.all(cutted_org_img[p][q] == [255,255,255]):
                        org_point += 1
                    if np.all(cutted_com_img[p][q] == [255,255,255]):
                        com_point += 1
            org_point_num.append(org_point)
            com_point_num.append(com_point)
            if abs(org_point - com_point) > 300:
                shape.append([start_h, start_w, start_h + c_h, start_w + c_w])
                #print('i = {0}, j = {1}'.format(i,j))
            org_point, com_point = 0,0
            start_w += dw

        start_h += dh
        start_w = 0

    diff = []


    plt.subplot(222)
    plt.title('Original image')
    x = [i for i in range(len(org_point_num))]
    plt.bar(x, org_point_num)


    plt.subplot(224)
    plt.title('Comparison image')
    x = [i for i in range(len(com_point_num))]
    plt.bar(x, com_point_num)

    for i in range(len(org_point_num)):
        #print('{}, diff = '.format(i + 1), end = " ")
        #print(abs(org_point_num[i] - com_point_num[i]))
        diff.append(abs(org_point_num[i] - com_point_num[i]))

    '''
    plt.subplot(223)
    plt.imshow(org_edge_img, cmap="gray")
    plt.subplot(224)
    plt.imshow(com_edge_img, cmap="gray")
    '''
    plt.figure(figsize = (8,4))
    plt.title('diff')
    x = [i for i in range(len(diff))]
    plt.bar(x,diff)



    if not shape:
        pass
    else:
        for i in range(len(shape)):
            after_img = cv2.rectangle(com_img, (shape[i][1], shape[i][0]), (shape[i][3], shape[i][2]), (255,0,0))

        plt.figure(figsize = (8,4))
        plt.title('Comparison result (canny)')
        plt.imshow(after_img)



    plt.show()
