from typing import Iterable

import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection

from .shape import Shape, ComplexMultiPolygon


class PPlot(object):
    def __init__(self, name: str = 'pplt'):
        self.name = name
        self.imgs = []
        self.ttls = []
        self.fig = plt.figure(name)

    def title(self, *titles):
        self.ttls.extend(titles)
        return self

    def add(self, *img):
        self.imgs.extend(img)
        return self

    def save(self, fname: str, dpi: int = 1000):
        self.__plot__()
        plt.savefig(fname=fname, dpi=dpi)
        self.fig.clear()
        # plt.close(self.name)
        return self

    def show(self):
        self.__plot__()
        plt.show()
        return self

    def __plot__(self):
        n = len(self.imgs)
        i = int(n ** 0.5)
        j = (n + i - 1) // i
        for p in range(i):
            for q in range(j):
                k = p * j + q
                if k >= n:
                    break
                ax = plt.subplot(i, j, k + 1)
                if k < len(self.ttls) and self.ttls:
                    plt.title(self.ttls[k])
                # ax.axis('equal')
                # plt.imshow(self.imgs[k])
                self.__draw__(ax, self.imgs[k])

                plt.axis('equal')

    def __draw__(self, ax, img):
        if isinstance(img, np.ndarray):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.imshow(img)
        elif isinstance(img, torch.Tensor):
            self.__draw__(ax, img.detach().cpu().numpy())
        elif isinstance(img, Image.Image):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.imshow(img)
        elif isinstance(img, Shape):
            l, u, r, d = list(map(int, img.bounds))
            # print(l, u, r, d)
            # plt.xticks([l, r])
            # plt.yticks([u, d])
            ax.set_xticks([l, r])
            ax.set_yticks([u, d])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # 变换矩阵: matrix = [xAx, xAy, yAx, yAy, xb, yb]
            # img = affine_transform(img, [1, 0, 0, -1, 0, d])
            if img:
                # https://stackoverflow.com/questions/55522395/how-do-i-plot-shapely-polygons-and-objects-using-matplotlib
                collections = []
                for im in img.sep_out():
                    outer = im.geo.exterior.coords
                    inners = [inner.coords for inner in im.geo.interiors]
                    path = Path.make_compound_path(
                        Path(np.asarray(outer)[:, :2]),
                        *[Path(np.asarray(inner)[:, :2]) for inner in inners]
                    )
                    patch = PathPatch(path)
                    collections.append(patch)
                collections = PatchCollection(collections, facecolor='lightblue', edgecolor='blue')   # facecolor='lightblue', edgecolor='red'
                ax.add_collection(collections, autolim=True)
                ax.autoscale_view()
                # 该方法适用于 shapely 1.7，不适用于 2.0.1
                # ax.add_patch(
                #     PolygonPatch(im.geo, fc=[0, 0, 1, 0.5], ec=[0, 0, 1, 0.5], alpha=0.5, zorder=2)
                # )

            ax.plot()
        elif isinstance(img, BaseGeometry):
            self.__draw__(ax, ComplexMultiPolygon(geo=img))
            # l, u, r, d = list(map(int, img.bounds))
            # # print(l, u, r, d)
            # # plt.xticks([l, r])
            # # plt.yticks([u, d])
            # ax.set_xticks([l, r])
            # ax.set_yticks([u, d])
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)
            # # 变换矩阵: matrix = [xAx, xAy, yAx, yAy, xb, yb]
            # # img = affine_transform(img, [1, 0, 0, -1, 0, d])
            # ax.plot()
        elif isinstance(img, Iterable):
            self.__draw__(ax, Polygon(img))
        else:
            raise NotImplementedError(f'{type(img)} is not supported!')

    def clear(self):
        # self.fig.clear()
        # plt.close(self.name)
        self.imgs.clear()
        self.ttls.clear()
        return self


def draw_contours_tuple(*contours):
    """
    cv2.findContours 的 contours 格式: tuple[array] -> array(points, 1, (y, x))
    cv2.fillPoly 的 contours 格式: (batch, points, (y, x))
    """
    pplt = PPlot()
    for i, contour in enumerate(contours):
        contour = contour.copy()
        contour[:, 0] -= contour[:, 0].min()
        contour[:, 1] -= contour[:, 1].min()
        img = np.zeros(shape=(contour[:, 0].max(), contour[:, 1].max()), dtype=np.uint8)
        contour = np.array(contour)
        contour = np.expand_dims(contour, axis=0)
        # cv2.polylines(img, contour, True, i+1, 5)
        cv2.fillPoly(img, contour, 1, lineType=1)
        pplt.add(img)
    pplt.show()


def draw_contours_shapely(*multi_polygons):
    pplt = PPlot()
    for i, multi_polygon in enumerate(multi_polygons):
        pplt.add(multi_polygon)
    pplt.show()


def ___draw_contours_shapely(*multi_polygons):
    for i, multi_polygon in enumerate(multi_polygons):
        ax = plt.subplot(1, len(multi_polygons), i+1)
        # https://stackoverflow.com/questions/55522395/how-do-i-plot-shapely-polygons-and-objects-using-matplotlib
        collections = []
        for im in multi_polygon.sep_out():
            outer = im.geo.exterior.coords
            inners = [inner.coords for inner in im.geo.interiors]
            path = Path.make_compound_path(
                Path(np.asarray(outer)[:, :2]),
                *[Path(np.asarray(inner)[:, :2]) for inner in inners]
            )
            patch = PathPatch(path)
            collections.append(patch)
        collections = PatchCollection(collections, facecolor='lightblue', edgecolor='blue')   # facecolor='lightblue', edgecolor='red'
        ax.add_collection(collections, autolim=True)
        ax.autoscale_view()
        # ax.plot()
        l, u, r, d = multi_polygon.bounds
        ax.set_xticks((l, r))
        ax.set_yticks((u, d))
    plt.axis('equal')
    plt.show()
