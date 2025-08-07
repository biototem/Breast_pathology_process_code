# @Time    : 2019.03.01
# @Author  : Bohrium Kwong
# @Licence : bio-totem


import cv2
import numpy as np
import sys
sys.path.append('../')
from utils.opencv_utils import OpenCV


def get_tissue(im, contour_area_threshold):
    """
    Get the tissue contours from image(im)
    :param im: numpy 3d-array object, image with RGB mode
    :param contour_area_threshold: python integer, contour area threshold, tissue contour is less than it will omit
    :return: tissue_cnts: python list, tissue contours that each element is numpy array with shape (n, 2)
    """

    im[np.std(im,axis=2)<10] = 255
    #在对原图进行灰度化之前,先将颜色接近全黑的部分改为白色(有些slide可能有污点),因为后续的基于阈值进行二值化的方法只能过来颜色偏白的部分,\
    # 如果不加这个条件,原图接近全黑的部分也会被认为有效的mask区域。
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    opencv = OpenCV(im)
#    binary = opencv.gray_binary(thresh=230, show=False)
#    morphology = opencv.erode_dilate(binary, erode_iter=0, dilate_iter=3, show=False)
    cnts = opencv.find_contours(is_erode_dilate = True)
#    tissue_cnts = []
    mask_keep_list = []

#    for each, cnt in enumerate(cnts):
    for cnt in cnts:
        contour_area = cv2.contourArea(cnt)
        if contour_area >= contour_area_threshold:
            # omit the small area contour
#            del cnts[each]
#            continue
#            tissue_cnts.append(np.squeeze(np.asarray(cnt)))
            mask_keep_list.append(cnt)

    # initialize mask to zero
    mask = np.zeros((im.shape[0], im.shape[1])).astype(im.dtype)
    color = [1]
    mask = cv2.fillPoly(mask, mask_keep_list, color)

    return mask, mask_keep_list


if __name__ == "__main__":
    from skimage import io
    import matplotlib.pyplot as plt
    im = io.imread("/Users/kawa/Desktop/52800/52800_.png")
    im = im.astype(np.uint8)
    mask, _ = get_tissue(im, contour_area_threshold=10000)
    plt.imshow(mask)
    plt.show()
