import numpy as np
import cv2
from matplotlib import pyplot as plt

from functions import *
from classes import *

def checkKeypointCoords(kp, top, left, bottom, right, margin=0):
    u = kp.pt[0]
    v = kp.pt[1]
    return (top + margin < v) and (left + margin < u) and (v < bottom - margin) and (u < right - margin)

def image_conversion(org_img,com_img):
    org_h, org_w, org_ch = org_img.shape
    com_h, com_w, com_ch = com_img.shape

    if org_h!=com_h or org_w!=com_w or org_ch!=com_ch:
        print("Not equal size")
    else:
        h = org_h
        w = org_w
        ch = org_ch
        multiple = 1.0
        left   = 0
        right  = w
        top    = 0
        bottom = h

        gray_org_img = cv2.cvtColor(org_img, cv2.COLOR_RGB2GRAY)
        gray_com_img = cv2.cvtColor(com_img, cv2.COLOR_RGB2GRAY)

        detector = cv2.AKAZE_create()
        org_kp, org_des = detector.detectAndCompute(gray_org_img, None)
        com_kp, com_des = detector.detectAndCompute(gray_com_img, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        matches = bf.match(org_des, com_des)
        matches = sorted(matches, key = lambda x:x.distance)

        margin = 50
        matches = [match for match in matches \
                   if checkKeypointCoords(org_kp[match.queryIdx], top, left, bottom, right, margin) \
                   and checkKeypointCoords(com_kp[match.trainIdx], top, left, bottom, right, margin)]
        num_correpondences = len(matches)
        #print("Num of correspondences (filtered by image coordinates): {}".format(num_correpondences))

        filtered_matches = matches[:100]

        height = h
        width  = w* 2

        out = np.zeros((height, width, ch), np.uint8)
        cv2.drawMatches(gray_org_img, org_kp, gray_com_img, com_kp, filtered_matches, out, flags=0)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        good_match_rate = 0.7
        good = matches[:int(len(matches) * good_match_rate)]
        src_org_pts = np.float32([ org_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_com_pts = np.float32([ com_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        H, mask = cv2.findHomography(src_org_pts, dst_com_pts, cv2.RANSAC, 5.0)

        warped_org_img = cv2.warpPerspective(org_img, H, (w, h))

        return warped_org_img


if __name__ == "__main__":
    org_img_path = "img/origin_image.png"
    com_img_path = "img/comparison_image.png"
    org_img,com_img = Read_img(org_img_path, com_img_path)
    warped_org_img = image_conversion(org_img, com_img)
    image_show(warped_org_img)
