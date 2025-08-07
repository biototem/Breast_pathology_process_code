#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 11:06:32 2025

@author: Bohrium Kwong
"""
import ujson
import os
import cv2
import numpy as np

def geojson_2_mask(geojson,label_mask,downsample,label_class_name = "Tumor"):
    # final_contour_list = []
    label_mask_2 = label_mask.copy()
    for tmp_dict in geojson:
        if True:#tmp_dict["properties"]["classification"]["name"] in [label_class_name]:
            # 这里按需改成其他条件
            contour_list = tmp_dict["geometry"]["coordinates"]
            contour_list = [(np.array(x)/downsample).astype(int) for x in contour_list]
            # final_contour_list.append(np.array(contour_list))
            label_mask = cv2.drawContours(label_mask_2, [np.array(contour_list)], -1, 1, -1)
    return label_mask



if __name__ == "__main__":
    import openslide
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = 13, 13
    
    slide = openslide.OpenSlide("/media/totem_data_backup2/totem/2024-9-23-NPC标注/2-胶原-ZY.svs")
    json_file = "/media/totem_data_backup2/totem/2024-9-23-NPC标注/2-胶原-ZY.geojson"
    level = 3
    downsample = slide.level_downsamples[level]
    label_mask = np.zeros((slide.level_dimensions[level][1],slide.level_dimensions[level][0]),dtype=np.uint8)
    with open(json_file, "r",encoding="utf-8") as FP:
        mask_geojson = ujson.load(FP)
        if type(mask_geojson) == dict: mask_geojson = mask_geojson["features"]
        
    label_mask = geojson_2_mask(mask_geojson,label_mask,downsample)
    plt.imshow(label_mask)
