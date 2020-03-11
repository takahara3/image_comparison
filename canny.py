import numpy as np
import cv2
import sys
import time
from matplotlib import pyplot as plt
from tqdm import tqdm

from functions import *

if __name__ == "__main__":
    start_time = time.time()
    args = get_option()

    ###比較画像の読み込み
    org_img_path = 'img/' + args.image_1
    com_img_path = 'img/' + args.image_2
    org_img, com_img = read_img(org_img_path, com_img_path)
    two_images_show(org_img, com_img)

    ###比較するために比較元画像を射影変換
    org_img_before = org_img
    org_img = image_conversion(org_img, com_img)
    #two_images_show(org_img_before, org_img)

    org_sample_path = 'img/org_grid.png'
    com_sample_path = 'img/com_grid.png'
    org_sample, com_sample =  read_img(org_sample_path, com_sample_path)
    #two_images_show(org_sample, com_sample)

    ###特徴点検出のためにGRAYスケール化
    gray_org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    gray_com_img = cv2.cvtColor(com_img, cv2.COLOR_BGR2GRAY)

    ###Canny法によるエッジ検出
    min_val = 100
    max_val = 200
    org_edge_img = cv2.Canny(gray_org_img, min_val, max_val)
    com_edge_img = cv2.Canny(gray_com_img, min_val, max_val)
    org_edge_img_show = cv2.cvtColor(org_edge_img, cv2.COLOR_RGB2BGR)
    com_edge_img_show = cv2.cvtColor(com_edge_img, cv2.COLOR_RGB2BGR)
    #two_images_show(org_edge_img_show, com_edge_img_show)

    result_img_show = cv2.imread('./out/Canny_result.png')
    result_img_show = cv2.cvtColor(result_img_show, cv2.COLOR_RGB2BGR)
    #two_images_show(org_img_before, result_img_show)

    ###特徴点の画素のRGB値
    color = [255,255,255]

    ###差分判定に使う特徴点の差
    ###特徴点の差が250個以上の矩形を差分とする
    diff = 300

    ###矩形ごとの特徴点を比較，差分の大きい矩形の座標＆各矩形の特徴点の数を取得
    shape, org_point_num, com_point_num = Comparison(org_edge_img, com_edge_img, color, diff)
    ###全ての矩形の共通部分の矩形を求める
    #common_rect = seek_intersect(shape)

    ###比較画像をそれぞれコピー
    img_1 = com_img.copy()
    img_2 = com_img.copy()

    #処理時間を表示
    end_time = time.time() - start_time
    print('elapsed time = {}'.format(end_time))
    two_images_show(img_1, img_2)

    ###矩形を描画
    if not shape:
        print('Not found')
    else:
        #result_img = draw_common_rect(img_1, common_rect)
        result_img = draw_rect(img_1, shape)
        #image_show(result_img)
        ###検出結果の保存
        if args.save == True:
            if args.name != None:
                write_img(result_img, img_name = args.name)
            else:
                write_img(result_img)

    ###グラフの作成
    if args.figure == True:
        draw_graph(org_point_num, com_point_num)
