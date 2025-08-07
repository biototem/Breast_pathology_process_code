from scipy import signal
import cv2
import numpy as np
from skimage import measure
from skimage.measure import regionprops
from copy import deepcopy
from scipy.signal import argrelextrema
from skimage.filters.rank import entropy
from skimage.morphology import disk


def image2mask(image: np.ndarray):
    # 阈值蒙版
    # # tim1 -> 滤除纯白区域，要求至少在某一RGB上具有小于210的色值
    # tim1 = np.any(image <= 210, 2).astype(np.uint8)
    # # tim2 -> 滤除灰度区域，要求RGB的相对色差大于18
    # tim2 = (np.max(image, 2) - np.min(image, 2) > 18).astype(np.uint8)

    # temp = cv2.resize(image, (w // 4, h // 4))
    tim1 = np.any(image <= 216, 2).astype(np.uint8)
    # tim2 = (np.max(temp, 2) - np.min(temp, 2) > 2).astype(np.uint8)
    m1 = np.max(image, 2)
    m2 = np.min(image, 2)
    m3 = np.mean(image, 2)
    m4 = m2 / 255 * (255-m1)
    m5 = np.clip(m4, 1, 8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m6 = signal.convolve2d(m5, k, mode='same') / k.sum()
    # print(m1.mean(),m2.mean(),m4.mean(),m5.mean())
    tim2 = (m1 - m2 > m6).astype(np.uint8)
    tim = tim1 * tim2
    # 捕获边缘
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    tim = cv2.dilate(tim, k, dst=tim, iterations=1)
    tim = cv2.erode(tim, k, dst=tim, iterations=2)
    tim = cv2.dilate(tim, k, dst=tim, iterations=1)

    # from utils import PPlot
    # # PPlot().add(temp, tim1, m1-m2, m5, tim2).show()
    # PPlot().add(temp, tim1, tim2, tim).show()
    return tim


def image2mask_probability(image: np.ndarray, smooth_level: int):
    """
    try to convert a image with (height, width, channel)->uint8
    into a mask-hit-map with (height, width)->float32
    while the pixel in mask standards the probability of this pixel
    if it belongs to background or not
    """
    # 精细的背景分割应当通过 AI 来完成
    tim1 = np.any(image <= 238, 2).astype(np.uint8)
    # tim2 = (np.max(temp, 2) - np.min(temp, 2) > 2).astype(np.uint8)
    m1 = np.max(image, 2)
    m2 = np.min(image, 2)
    m3 = np.mean(image, 2)
    m4 = m2 / 255 * (255-m1)
    m5 = np.clip(m4, 1, 8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m6 = signal.convolve2d(m5, k, mode='same') / k.sum()
    # print(m1.mean(),m2.mean(),m4.mean(),m5.mean())
    tim2 = (m1 - m2 > m6).astype(np.uint8)
    tim = tim1 * tim2
    # 捕获边缘
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    tim = cv2.dilate(tim, k, dst=tim, iterations=1)
    tim = cv2.erode(tim, k, dst=tim, iterations=2)
    tim = cv2.dilate(tim, k, dst=tim, iterations=1)

    # from utils import PPlot
    # # PPlot().add(temp, tim1, m1-m2, m5, tim2).show()
    # PPlot().add(temp, tim1, tim2, tim).show()
    return tim


# function to apply the entropy mask/filter
def __filter_entropy_image(image, filter, disk_radius: int = 3):
    eimage = entropy(image, disk(disk_radius))
    new_picture = np.ndarray(shape=eimage.shape)  # [[False] * image.shape[1]] * image.shape[0]
    for rn, row in enumerate(eimage):
        for pn, pixel in enumerate(row):
            if pixel < filter:
                new_picture[rn, pn] = True
            else:
                new_picture[rn, pn] = False
    return new_picture.astype('b')


# main program to apply to a given image
def image2mask_cross_entropy(inputs: np.ndarray) -> np.ndarray:
    source = deepcopy(inputs)
    ent = entropy(source, disk(5))
    hist = list(np.histogram(ent, 30))
    minindex = list(argrelextrema(hist[0], np.less))

    thresh_localminimal = 0
    for i in range(len(minindex[0])):
        temp_thresh = hist[1][minindex[0][i]]
        if 1 < temp_thresh < 4:
            thresh_localminimal = temp_thresh

    thresh1 = (255 * __filter_entropy_image(inputs, thresh_localminimal)).astype('uint8')
    mask_255 = cv2.bitwise_not(deepcopy(thresh1))

    redolbl = measure.label(np.array(mask_255), connectivity=2)
    redprops = regionprops(redolbl)

    redAreaData = []
    for r in range(len(redprops)):
        redAreaData.append(redprops[r].area)

    # 新方法
    redAreaData = sorted(enumerate(redAreaData), key=lambda p: p[1], reverse=True)
    # choses = [i for i, v in redAreaData]
    choses = []
    for i, v in redAreaData:
        if v > 0 and v / sum(p[1] for p in redAreaData[:i+1]) > 0.005:
            choses.append(i)
        else:
            break
    max_mask = np.zeros_like(inputs)
    for chose in choses:
        max_mask[redolbl == chose + 1] = 255

    # 旧方法
    # max_value = max(redAreaData)
    # max_index = redAreaData.index(max_value)
    #
    # max_mask = (redolbl == (max_index + 1)) * 255

    return max_mask
