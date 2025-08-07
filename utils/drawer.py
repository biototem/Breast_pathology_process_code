import time
from skimage.segmentation import watershed
import traceback
import numpy as np
import cv2
import threading

from utils import Canvas, Timer

"""
说明：
该代码原本是 renal 项目 confidence 方法下的一个子系统，实现手动绘图标记
现对该系统做一定调整，以图实现该项目的手动标图任务
该任务包含以下操作：
1. 选择 label 值
2. 拖拽划线 -> 改变一条线上的 label 值，用作区域封闭
3. 区域调整 -> 改变一片区域的 label 值，配合上面那个使用
4. 封闭线 -> 当拖拽划线构成封闭区域时，直接对该区域改变
5. 撤销 -> 手动修图的基本功能
6. 保存和退出 -> 略
7. 图片切换 -> 略
在实现的层面来说，这种任务原本应该由专门的标绘软件负责，但存在学习成本
而且可控性差，容易引起不必要的麻烦
因此不如直接在代码里实现掉
具体的讲，这套代码包含这样几个层次的任务：
1. 线程控制 -> 因为操作需要时间，而进程不能卡死，画完图还有别的代码任务要做呢
2. 键鼠指令识别 -> 基本的 event 编程
3. 图层处理 -> 绘图时希望看到 image 与 label 的叠加效果，而叠加比例应当可变
4. 图缩放 -> 直接在原图上绘图太奢侈了，会很卡，所以应该配一套缩放
5. 状态池 -> 为撤销准备的
"""


class Painter(threading.Thread):
    def __init__(self, save: callable):
        super().__init__()
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1000, 1000)
        cv2.moveWindow('image', -1, -1)
        cv2.setMouseCallback('image', self)
        # 基本控制参数
        self.mix_rate = 0.5
        self.value = 0
        self.area_mode = False
        # 管理全部图片
        self.source = []
        self.processing = False
        self.index = 0
        # 管理图片状态
        self.name = 'empty'
        self.message = 'message'
        self.box = None
        self.image = None
        self.history = None
        # 管理绘画状态
        self.label = None
        self.draw_time = False
        self.lines = []
        # 保存回调
        self.save = save

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.run()
        if exc_type is not None or exc_val is not None or exc_tb is not None:
            traceback.print_exc()
        return True

    def add(self, name, message, image, label):
        self.source.append({
            'name': name,
            'message': message,
            'box': (0, 0, *image.shape),
            'image': image,
            'label': label.copy(),
            'history': [label.copy()],
        })
        # 'image': Canvas().with_image(image).image[:, :, (2, 1, 0)].copy()
        # 'image': Canvas().with_image(image)
        #      .draw_color(label[:, :, 1], hyp['visual.color.outline'][1])
        #      .draw_border(label[:, :, 2], hyp['visual.color.outline'][2])
        #      .draw_border(label[:, :, 3], hyp['visual.color.outline'][3])
        #      .image[:, :, (2, 1, 0)].copy()
        return self

    def run(self):
        while len(self.source) < 1:
            time.sleep(0.25)
        self.switch('current')
        while True:
            self.processing = False
            # 每次绘完图都需要重新展示e
            cv2.imshow('image', self.draw())
            key = cv2.waitKey(20) & 0xFF
            # s 表示保存
            if key == 115:
                print('saved')
                self.processing = True
                data = self.source.pop(self.index)
                self.save(data['name'], data['history'][-1])
                del data
                # 所有图片都保存了就退出
                if len(self.source) == 0:
                    break
                self.switch('current')
            # q 表示上一张
            if key == 113:
                print('switch to last')
                self.processing = True
                self.switch('last')
            # e 表示下一张
            if key == 101:
                print('switch to next')
                self.processing = True
                self.switch('next')
            # + 表示增加混淆比例
            if key == 43:
                print(f'add mix_rate into {self.mix_rate}')
                self.mix_rate = min(1, self.mix_rate + 0.1)
            # - 表示减少混淆比例
            if key == 45:
                print(f'reduce mix_rate into {self.mix_rate}')
                self.mix_rate = max(0, self.mix_rate - 0.1)
            # 0-7 数字表示切换类型
            if key in range(48, 56):
                print(f'set value into {self.value}')
                self.value = key - 48
            # a 表示区域修图
            if key == 97:
                print(f'area changed')
                self.area_mode = not self.area_mode
            # 27 是 ESC 的退出码，表示主动退出
            if key == 27:
                print('exit')
                break
        cv2.destroyWindow('image')
        # return [business['history'][-1] for business in self.source]

    def switch(self, flag: str):
        if flag == 'next':
            p = 1
        elif flag == 'last':
            p = -1
        elif flag == 'current':
            p = 1
            self.index -= 1
        else:
            raise RuntimeError('flag not exists')
        i = self.index + p
        while i < 0 or i >= len(self.source):
            i += p
            if i >= len(self.source):
                i = 0
            elif i < 0:
                i = len(self.source) - 1
            elif i == self.index:
                raise RuntimeError('no more images')
        self.index = i
        # print(f'painting image {self.source[i]["name"]}')
        self.name = self.source[i]['name']
        self.message = self.source[i]['message']
        self.box = self.source[i]['box']
        self.image = self.source[i]['image']
        self.label = self.source[i]['history'][-1].copy()
        self.history = self.source[i]['history']
        self.draw_time = False
        self.lines.clear()
        cv2.setWindowTitle('image', f'{self.name} -> {self.message}')

    def __call__(self, event, x, y, flags, param):
        # 鼠标左键按下状态移动划线
        if not self.area_mode and event == cv2.EVENT_MOUSEMOVE:
            if not self.draw_time: return
            # print('line', x, y)
            x0, y0 = self.lines[-1]
            if (x - x0) ** 2 + (y - y0) ** 2 < 25: return
            self.lines.append((x, y))
            self.trace()
        # 鼠标按下记录状态
        elif event == cv2.EVENT_LBUTTONDOWN:
            # print('draw')
            self.lines.append((x, y))
            self.draw_time = time.time()
        # 标定状态下执行这套代码
        elif not self.area_mode and event == cv2.EVENT_LBUTTONUP:
            # print('drawn')
            self.lines.append((x, y))
            self.paint()
            self.draw_time = False
        # 区域状态下执行这套代码
        elif self.area_mode and event == cv2.EVENT_LBUTTONUP:
            x0, y0 = self.lines[-1]
            # 判定鼠标单击： 鼠标按下、弹起的距离要近，时间要短
            if (x - x0) ** 2 + (y - y0) ** 2 < 10 * 10 and time.time() - self.draw_time < 0.75:
                self.area_change(x, y)
            self.lines.clear()
            self.draw_time = False
        # 标定状态下双击执行这套代码
        elif not self.area_mode and event == cv2.EVENT_LBUTTONDBLCLK:
            # print('clear')
            if len(self.history) > 1:
                self.history.pop()
            self.label = self.history[-1].copy()
            self.paint()
            self.lines.clear()
            self.draw_time = False

    def draw(self):
        return (self.image * (1 - self.mix_rate) + COLORS[self.label] * self.mix_rate).astype(np.uint8)

    def trace(self):
        x1, y1 = self.lines[-2]
        x2, y2 = self.lines[-1]
        cv2.line(self.label, (x1, y1), (x2, y2), self.value, 11)

    def paint(self):
        # 时间太短、长度太短的直接摒弃
        if not self.draw_time or time.time() - self.draw_time < 0.7 or len(self.lines) <= 4:
            self.draw_time = False
            self.lines.clear()
            self.label = self.history[-1].copy()
            return
        d1, x1, y1 = self.moveto(self.lines[0])
        d2, x2, y2 = self.moveto(self.lines[-1])
        d = {**d1, **d2}
        # 有划线的，总共分定四种具体情况，两种归并情况
        # 自围 -> 围区
        if (x1 - x2) ** 2 + (y1 - y2) ** 2 < 40 * 40:
            flag = 0
        # 围角 -> 围区
        elif 'x' in d and 'y' in d:
            flag = 0
            self.lines.append((x2, y2))
            self.lines.append((d['x'], d['y']))
            self.lines.append((x1, y1))
        # 围边 -> 围区
        elif ('x' in d1 and 'x' in d2 and d1['x'] == d2['x']) or \
                ('y' in d1 and 'y' in d2 and d1['y'] == d2['y']):
            flag = 0
            self.lines.append((x2, y2))
            self.lines.append((x1, y1))
        # 未围 -> 划线
        else:
            flag = 1
        # 修改 label 的取值，flag == 1 表示划线，否则表示围区
        if flag == 1:
            # cv2.polylines(self.label, [np.array(self.lines)], True, self.value, 11)
            # 由于 label 已经画好了，不需要做任何其它事情
            pass
        else:
            cv2.fillPoly(self.label, [np.array(self.lines)], self.value)
        self.lines.clear()
        self.history.append(self.label.copy())

    def moveto(self, p):
        b = self.box
        l = abs(p[0] - b[0])
        u = abs(p[1] - b[1])
        r = abs(p[0] - b[2])
        d = abs(p[1] - b[3])
        if min(l, u, r, d) > 100:
            return {}, p[0], p[1]
        if min(l, u, r, d) == l:
            return {'x': b[0]}, b[0], p[1]
        if min(l, u, r, d) == u:
            return {'y': b[1]}, p[0], b[1]
        if min(l, u, r, d) == r:
            return {'x': b[2]}, b[2], p[1]
        if min(l, u, r, d) == d:
            return {'y': b[3]}, p[0], b[3]

    def area_change(self, x, y):

        # 获得区块分割结果图
        mask = self.label == self.label[y, x]
        result = watershed(~mask, mask=mask)

        # 将目标区域改变取值
        self.label[result == result[y, x]] = self.value

        # # 然后对每个区块判定：
        # # marker = label == self.label[y, x]
        # self.label[result == result.max()] = self.valuex
        # # for i in range(1, result.max() + 1):
        # #     area = result == i
        # #     if (area & marker).any():
        # #         self.label[area] = self.value

        self.history.append(self.label.copy())
