from typing import Dict, Tuple, Iterable
import torch

from .interface import MergerInterface


class ColMerger(MergerInterface):
    def __init__(self, channel: int, line_width: int, height: int, dtype: torch.dtype = torch.float32, device: str = 'cpu'):
        """
        列融合，x 非减，每当 set 过程中 grid['x'] 的取值发生变化，执行行卷动
        :param channel:
        :param line_width:
        :param height:
        """
        self.x = torch.nan
        self.c = channel
        self.w = line_width
        self.h = height
        # m 表示行裕度，需结合 kernel 方能确定，因此初始化时设为 0
        self.m = torch.nan
        # 预测融合行
        self.target = torch.zeros(channel, height, line_width, dtype=dtype, device=device, requires_grad=False)
        self.helper = torch.zeros(1, height, line_width, dtype=dtype, device=device, requires_grad=False) + 1e-19
        # 融合完成队列
        self.que = []

    def set(self, grid: Dict[str, int], target: torch.Tensor, helper: torch.Tensor):
        """
        添加融合图块，需自行传入 patch、kernel
        当融合图块所指示的 grid 超出本行范围时，自动卷动并将冗余部分弹出至 readies 队列中
        dtype 和 device 向创建时的指定对象对齐
        :param grid:     {x: int, y: int}
        :param target:   [c, h, w] -> dtype :: device
        :param helper:   [1, h, w] -> dtype :: device
        """
        x = grid['x']
        y = grid['y']
        kh, kw = target.shape[1:]
        if self.x is torch.nan:
            # 第一个图块将会设定行初始坐标
            self.x = x
            self.m = self.w - kw
        if x < self.x:
            # 行融合要求 y 非减，如果出现 y 减少，说明调用有误
            raise Exception(f'数据异常:: 上一个 x 轴坐标是 {self.x} 而这一个是 {x}，x 居然减少了！按照约定，列融合，x 非减！')
        if x == self.x:
            # if not(self.h > grid['y']):
            #     sys.stderr.write(f'ERROR :: merge grid {grid} out of bounds with max-height {self.h} \n')
            #     return
            # x 相等说明在同一列上，进行简单融合即可
            # 边界规范化操作
            temp_left = self.m
            temp_up = max(0, y)
            temp_right = min(self.w, self.m + kw)
            temp_down = min(self.h, max(0, y + kh))
            patch_left = max(0, -(x - self.x))
            patch_up = max(0, -y)
            patch_right = min(self.m + kw, self.w)
            patch_down = min(kh, max(0, self.h - y))
            self.target[:, temp_up: temp_down, temp_left: temp_right] += target[:, patch_up: patch_down, patch_left: patch_right]
            if helper is not None:
                self.helper[:, temp_up: temp_down, temp_left: temp_right] += helper[:, patch_up: patch_down, patch_left: patch_right]
            else:
                self.helper[:, temp_up: temp_down, temp_left: temp_right] = 1
            return
        if x > self.x:
            # y 不相等说明行向下卷动了，执行卷动
            pass

        # step 是行向下卷动的步长，注意，这个是实际卷动步长，不包括由 m 引起的逻辑卷动步长
        step = min(x - self.x, self.w - self.m)

        patch = (self.target[:, :, self.m: self.m + step] / self.helper[:, :, self.m: self.m + step]).clone()
        self.que.append(({'x': self.x, 'y': 0}, patch))

        # 列交叠
        self.target[:, :, :-step] = self.target[:, :, step:].clone()
        self.target[:, :, -step:] = 0
        self.helper[:, :, :-step] = self.helper[:, :, step:].clone()
        self.helper[:, :, -step:] = 1e-19

        # 重设行坐标
        self.x += step

        # 当 x 不对齐时，需要先进行上面一大堆卷动处理，别忘了，此时新进来的这个图块还没处理呢
        # 所以别忘了递归一下处理自己当前的这个数据
        self.set(grid, target, helper)

    def readies(self) -> Iterable[Tuple[Dict[str, int], torch.Tensor]]:
        # set、finish 的写出都仅仅是将坐标和图块丢到 que 里，实际上要从这里取出并清空的
        # 当然啦，虽然 que 实现的是队列的功能，但实际上它并不是队列，而是用 list 代替的
        for grid, tile in self.que:
            yield grid, tile.type(torch.float32)
        self.que.clear()

    def finish(self) -> None:
        # 当 finish 时，需将剩余的全部坐标写出
        grid = {'x': self.x, 'y': 0}
        patch = (self.target[:, :, self.m:] / self.helper[:, :, self.m:]).clone()
        self.que.append((grid, patch))
