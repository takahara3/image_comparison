import numpy as np
import cv2
from matplotlib import pyplot as plt

from functions import *
from classes import *

if __name__ == "__main__":
    org_img_path = "img/origin_image.png"
    com_img_path = "img/comparison_image.png"
    org_img,com_img = Read_img(org_img_path, com_img_path)
    warped_org_img = image_conversion(org_img, com_img)
    image_show(warped_org_img)
