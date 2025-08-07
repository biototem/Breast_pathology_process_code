from typing import List, Dict, Any, Union, Callable

from .shapely_impl import Shape, ComplexPolygon, ComplexMultiPolygon

"""
ShapeSet 是对 Shape 的类型化打包
在 Shape 有关章节中我们知道：
    一个平面图形
    可能是 Single 的，也可能是 Multi 的
    可能是 Simple 的，也可能是 Complex 的
在本公司涉及的业务中，若两个平面图形轮廓指代同一类型，则它们几乎不可能相交
若指代的是不同轮廓，相交通常呈现为 A 含于 B 的情况
这使得我们可以不必关心或计较轮廓间的相交类型和先后次序，而将注意力放在不同类型的管理上

考虑一个符合规定的 label，它包含 N 种类型，每种类型用一个 ComplexMultiPolygon 对象表示
我们将这种贴上标签的、表征一组图像标注的 Shape 的集合称为 ShapeSet
ShapeSet = {
    '{label_name}': ComplexMultiPolygon,
}
为了避免不必要的复杂度，我们统一用最高等级的轮廓表示方法 ComplexMultiPolygon 来承载一切
相应的，当需要序列化时，我们只需统一按照它的 sep_p 格式存储即可
{
    '{label_name}': {
        'outers': List[points],
        'inners': List[points],
        'adjacencies': List[index],
    },
}

然而通常情况下，我们的 ShapeSet 并非凭空产生，而源自某种标定格式，例如 geojson
其标定格式为：
[
    {
        "type": "Feature",
        "geometry": {
            "type": "Polygon"
            "coordinates": [outer_points, *inner_points]
        },
        "properties": {
            "object_type": "annotation",
            "classification": {
                "name": "label_name",
                "colorRGB":  int32
            },
            "isLocked": bool
        }
    },
    {
        "type": "Feature",
        "geometry": {
            "type": "MultiPolygon"
            "coordinates": [polygon]
        },
        "properties": {
            "object_type": "annotation",
            "classification": {
                "name": "label_name",
                "colorRGB":  int32
            },
            "isLocked": bool
        }
    },
]
可以看到，该格式完整记载了标定过程中的各种信息，且易于阅读和理解，是一种非常优秀的标定格式
但其中夹杂了太多我们用不到的信息，对我们来说有用的只有三个东西：
    geometry.type
    geometry.coordinates
    properties.classification.name
原格式中同一类型可能存在若干个相互独立的轮廓，在新格式中，我们会将它们合并在一起

应当考虑到命名相关的问题：
    同一事物可以有不同的名称
    同一名称可以有不同的书写风格
为规避源文件类型名称与项目中类型名称相冲突的风险，我们提供一个 map_to 方法，并要求类型名全部为大写字母
它的作用是重新设定类型名称，这既可以通过映射表实现，也可以通过自定义函数实现

此外，实际业务中存在一种“基于ROI的聚合分解”运算
具体的讲，每种轮廓都应当唯一对应某个 ROI，所谓 “对应” 的判定依据则是 “相交”
而后，对应于每个 ROI 的轮廓族单独成立一个 label，该操作既是 divide_by
它将返回 List[ShapeSet]
"""


class ShapeSet(object):
    def __init__(self, data: Dict[str, ComplexMultiPolygon] = None):
        self.data: Dict[str, ComplexMultiPolygon] = {} if data is None else data

    def from_geojson(self, geojson_content: List[Dict[str, Any]]):
        # geojson -> shape_set
        for item in geojson_content:
            key = item['properties']['classification']['name'].upper()
            if key not in self.data:
                self.data[key] = Shape.EMPTY
            contour_type = item['geometry']['type'].upper()
            contours = item['geometry']['coordinates']
            # 单形 -> 多形
            if contour_type == 'POLYGON':
                contours = [contours]
            # 多形 -> 采用单形遍历合成的方式，原因是原数据结构就这样
            polygons = [ComplexPolygon(contour[0], *contour[1:]) for contour in contours]
            # 虽然来来回回算了好几次，效率低下，但考虑到这本来也不是效率节点，问题不大
            self.data[key] |= ComplexMultiPolygon(singles=polygons)
        return self

    def from_shape_set(self, shape_set_content: Dict[str, Dict[str, Union[str, list]]]):
        # shape_set_serializable -> shape_set
        for key, shape_json in shape_set_content.items():
            self.data[key] = ComplexMultiPolygon(
                outers=shape_json['outers'],
                inners=shape_json['inners'],
                adjacencies=shape_json['adjacencies']
            )
        return self

    def as_struct(self) -> Dict[str, Dict[str, Union[str, list]]]:
        # shape_set -> shape_set_serializable
        return {
            key: dict(zip(
                ['outers', 'inners', 'adjacencies'], list(value.sep_p())
            )) for key, value in self.data.items()
        }

    def divide_by(self, key: str, jointless_ignore: bool = False, margin_limit: float = -1) -> list:
        # shape_set -> List[shape_set]
        areas: List[Dict[str, ComplexMultiPolygon]] = [
            {
                key: ComplexMultiPolygon(singles=[polygon])
            } for polygon in self.data[key].sep_out()
        ]
        for k, contours in self.data.items():
            if k == key: continue
            for contour in contours:
                joint_count = 0
                for area in areas:
                    if not area[key].is_joint(contour): continue
                    if k not in area:
                        area[k] = Shape.EMPTY
                    if margin_limit > 0:
                        contour &= area[key].buffer(margin_limit)
                    area[k] |= contour
                    joint_count += 1

                if joint_count < 1:
                    print(f'shape not joint {key} class {k} area {contour.area}')
                    if not jointless_ignore:
                        raise ValueError(f'Shape Not joint with any key!')
        return [ShapeSet(data=area) for area in areas]

    def map_to(self, keymap: Union[Callable, Dict[str, str]]):
        data = {}
        # callable -> dict
        if isinstance(keymap, Callable):
            keymap = {k: keymap(k) for k in self.data.keys()}
        for key, value in self.data.items():
            nkey = keymap[key] if key in keymap else key
            if nkey not in data:
                data[nkey] = Shape.EMPTY
            # 值迁移
            data[nkey] |= value
        # 值变换
        self.data = data
        return self
# 
# 
# def trans(labels: dict) -> List[dict]:
#     """
#     将 json 标签翻译为区域族集，翻译方法如下：
#     1. 扫描所有的 ROI 类型轮廓
#     2. 断言，任何轮廓至少与其中一个 ROI 相交
#     3. 以 ROI 为基准，生成若干区域，每个区域是一个字典
#     """
#     # step 1: scaning rois
#     rois = []
#     for label in labels:
#         if label['properties']['classification']['name'] != 'ROI': continue
#         # coords of roi
#         coords = label['geometry']['coordinates'][0]
#         # coords to bounds
#         left = min(x for x, y in coords)
#         up = min(y for x, y in coords)
#         right = max(x for x, y in coords)
#         down = max(y for x, y in coords)
#         # bounds to region
#         rois.append(Region(left, up, right, down))
#     # step 2: assert
#     counters = {}
#     for label in labels:
#         name = label['properties']['classification']['name']
#         if name == 'ROI': continue
#         if name not in counters:
#             counters[name] = []
#         if label['geometry']['type'].upper() == 'POLYGON':
#             coords = label['geometry']['coordinates']
#             shape = ComplexPolygon(coords[0], *coords[1:])
#         elif label['geometry']['type'].upper() == 'MULTIPOLYGON':
#             coords = label['geometry']['coordinates']
#             shape = ComplexMultiPolygon(singles=[
#                 ComplexPolygon(cds[0], *cds[1:]) for cds in coords
#             ])
#         else:
#             raise ValueError(f'New geo-type: {label["geometry"]["type"]}')
#         # shape = shape.simplify()
#         for roi in rois:
#             if roi.is_joint(shape): break
#         else:
#             raise ValueError(f'Disjoint shape at {name}')
#         counters[name].append(shape)
# 
#     # step 3: contours grouped by rois
#     areas = []
#     for roi in rois:
#         area = {'ROI': roi}
#         for name, shapes in counters.items():
#             for shape in shapes:
#                 if not roi.is_joint(shape): continue
#                 if name not in area:
#                     area[name] = ComplexPolygon.EMPTY
#                 area[name] |= shape
#         areas.append(area)
#     return areas
