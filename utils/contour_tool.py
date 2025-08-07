#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:43:24 2024

@author: Bohrium Kwong
"""

import cv2
import shapely

assert cv2.__version__ >= '4.0'
import numpy as np
from typing import Iterable, List
from shapely.geometry import Polygon, MultiPolygon, MultiPoint, Point, LineString, LinearRing
from shapely.geometry.base import BaseMultipartGeometry
from shapely.ops import unary_union

def shapely_merge_multi_contours(polygons: List[Polygon]):
    '''
    直接合并多个轮廓
    :param polygons: 多个轮廓，要求为shapely的格式
    :return:
    '''
    ps = unary_union(polygons)
    ps = shapely_ensure_polygon_list(ps)
    return ps


def shapely_ensure_polygon_list(ps):
    '''
    确保返回值是 List[Polygon]
    本函数目的是减少shapely返回值可能是任何类的问题
    :param ps:
    :return:
    '''
    if isinstance(ps, Polygon):
        if ps.is_empty:
            ps = []
        else:
            ps = [ps]
    elif isinstance(ps, MultiPolygon):
        ps = list(ps.geoms)
    elif isinstance(ps, BaseMultipartGeometry):
        ps = [p for p in ps.geoms if isinstance(p, Polygon)]
    elif isinstance(ps, Iterable):
        ps = [p for p in ps if isinstance(p, Polygon)]
    elif isinstance(ps, (Point, LineString, LinearRing)):
        ps = []
    else:
        raise RuntimeError('Error! Bad input in shapely_ensure_polygon_list.', str(type(ps)), ps)
    return ps

def shapely_morphology_contour(contour: Polygon, distance, resolution=16):
    '''
    对轮廓进行形态学操作，例如膨胀和腐蚀
    对轮廓腐蚀操作后，可能会返回多个轮廓，也可能返回一个空轮廓，此时做好检查，并排除
    因为可能返回多个轮廓，所以统一用list来包装
    :param contour:                 输入轮廓
    :param distance:                形态学操作距离
    :param resolution:              分辨率
    :return:
    '''
    out_c = contour.buffer(distance=distance, quad_segs=resolution)
    out_c = shapely_ensure_polygon_list(out_c)
    out_c = [c for c in out_c if not c.is_empty and c.is_valid and c.area > 0]
    return out_c

def shapely_diff_contours_1toN(c1: Polygon, batch_c: List[Polygon]):
    '''
    计算一个轮廓与轮廓组的不相交轮廓
    :param c1:
    :param batch_c:
    :return:
    '''
    cs = [c1]
    for c in batch_c:
        new_cs = []
        for c2 in cs:
            diff = c2.difference(c)
            diff = shapely_ensure_polygon_list(diff)
            new_cs.extend(diff)
        cs = new_cs
    return cs

def tr_my_to_cv_contours(my_contours):
    '''
    轮廓格式转换，转换我的格式到opencv的格式
    :param my_contours:
    :return:
    '''
    out_contours = [c[:, None, ::-1] for c in my_contours]
    return out_contours

def calc_contours_area(contours: np.ndarray):
    '''
    求一组轮廓的面积
    :param contours: 输入一组轮廓
    :return:
    '''
    cs = tr_my_to_cv_contours(contours)
    areas = [cv2.contourArea(c) for c in cs]
    return areas

def calc_contour_area(contour: np.ndarray):
    '''
    求轮廓的面积
    :param contour: 输入一个轮廓
    :return:
    '''
    area = cv2.contourArea(tr_my_to_cv_contours([contour])[0])
    return area

def tr_my_to_polygon(my_contours):
    '''
    轮廓格式转换，转换我的格式到polygon
    :param my_contours:
    :return:
    '''
    polygons = []
    for c in my_contours:
        # 如果c是float64位的，要转化为float32位
        if c.dtype == np.float64:
            c = c.astype(np.float32)
        # 如果点数少于3个，就先转化为多个点，然后buffer(1)转化为轮廓，可能得到MultiPolygon，使用convex_hull得到凸壳
        if len(c) < 3 or calc_contour_area(c) == 0:
            p = MultiPoint(c[:, ::-1]).buffer(1).convex_hull
        else:
            p = Polygon(c[:, ::-1])
        if not p.is_valid:
            # 如果轮廓在buffer(0)后变成了MultiPolygon，则尝试buffer(1)，如果仍然不能转换为Polygon，则将轮廓转换为凸壳，强制转换为Polygon
            p1 = p.buffer(0)
            if not isinstance(p1, Polygon):
                p1 = p.buffer(1)
            if not isinstance(p1, Polygon):
                # warnings.warn('Warning! Found an abnormal contour that cannot be converted directly to Polygon, currently will be forced to convex hull to allow it to be converted to Polygon')
                p1 = p.convex_hull
            p = p1
        polygons.append(p)
    return polygons

def tr_polygons_to_my(polygons: List[Polygon], dtype: np.dtype=np.float32):
    '''
    转换shapely的多边形到我的格式
    :param polygons:
    :param dtype: 输出数据类型
    :return:
    '''
    my_contours = []
    tmp = shapely_ensure_polygon_list(polygons)
    assert len(tmp) == len(polygons), 'Error! The input polygons has some not Polygon item.'
    polygons = tmp
    for poly in polygons:
        x, y = poly.exterior.xy
        c = np.array(list(zip(y, x)), dtype)
        my_contours.append(c)
    return my_contours

def morphology_contours(contours, distance, resolution=16, merge=True):
    '''
    对轮廓进行形态学操作，例如膨胀和腐蚀
    对轮廓腐蚀操作后，可能会返回多个轮廓
    因为可能返回多个轮廓，所以统一用list来包装
    :param contours:                输入轮廓组 list[np.ndarray]
    :param distance:                形态学操作距离
    :param resolution:              分辨率
    :return:
    '''
    if len(contours) > 0:
        dtype = contours[0].dtype
    else:
        dtype = np.float32

    cs = tr_my_to_polygon(contours)

    out_cs = []
    for c in cs:
        out_cs.extend(shapely_morphology_contour(c, distance, resolution=resolution))

    if len(out_cs) == 0:
        return []
    
    if merge:
        out_cs = shapely_merge_multi_contours(out_cs)
    
    out_cs = tr_polygons_to_my(out_cs, dtype)
    return out_cs

def merge_multi_contours(contours):
    '''
    直接合并多个轮廓
    :param contours: 多个轮廓，要求为我的格式
    :return: 返回合并后的剩余轮廓
    '''
    polygons = tr_my_to_polygon(contours)
    mps = shapely_merge_multi_contours(polygons)
    cs = tr_polygons_to_my(mps)
    return cs

def diff_contours_1toN(c1, batch_c):
    '''
    计算一个轮廓与轮廓组的不相交轮廓
    :param c1:
    :param batch_c:
    :return:
    '''
    c1 = tr_my_to_polygon([c1])[0]
    batch_c = tr_my_to_polygon(batch_c)
    cs = shapely_diff_contours_1toN(c1, batch_c)
    cs = tr_polygons_to_my(cs)
    return cs

def shapely_calc_occupancy_ratio(contour1, contour2):
    '''
    计算轮廓1和轮廓2的相交区域与轮廓2的占比，原型
    :param contour1: polygon多边形1
    :param contour2: polygon多边形1
    :return:    IOU分数
    '''
    c1 = contour1
    c2 = contour2
    if not c1.intersects(c2):
        return 0.
    area2 = c2.area
    inter_area = c1.intersection(c2).area
    ratio = inter_area / max(area2, 1e-8)
    return ratio

def calc_occupancy_ratio_Nto1(contours, contour1):
    '''
    计算多个轮廓与轮廓1的相交区域与轮廓1的占比
    :param contours:
    :param contour1:
    :return:
    '''
    contour1 = tr_my_to_polygon([contour1])[0]
    contours = tr_my_to_polygon(contours)
    ratio = np.zeros([len(contours)], np.float32)
    for i, c in enumerate(contours):
        ratio[i] = shapely_calc_occupancy_ratio(c, contour1)
    return ratio


def _resize_contour(contour, scale_factor_hw, offset_yx, dtype):
    return ((contour - offset_yx) * scale_factor_hw + offset_yx).astype(dtype)

def resize_contours(contours, scale_factor_hw=1.0, offset_yx=(0, 0)):
    '''
    缩放轮廓
    :param contours: 输入一组轮廓
    :param scale_factor_hw: 缩放倍数
    :param offset_yx: 偏移位置，默认为左上角(0, 0)
    :return:
    '''
    scale_factor_hw = np.asarray(scale_factor_hw).reshape([1, -1])
    offset_yx = np.asarray(offset_yx, np.float32).reshape([1, 2])
    out = [_resize_contour(c, scale_factor_hw, offset_yx, contours[0].dtype) for c in contours]
    return out
