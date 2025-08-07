from typing import Dict, Tuple, Iterable
import torch

from .interface import MergerInterface


class RowMerger(MergerInterface):
    def __init__(self, channel: int, line_height: int, width: int, dtype: torch.dtype = torch.float32, device: str = 'cpu'):
        """
        行融合，y 非减，每当 set 过程中 grid['y'] 的取值发生变化，执行行卷动
        :param channel:
        :param line_height:
        :param width:
        """
        self.y = torch.nan
        self.c = channel
        self.h = line_height
        self.w = width
        # m 表示行裕度，需结合 kernel 方能确定，因此初始化时设为 0
        self.m = torch.nan
        # 预测融合行
        self.target = torch.zeros(channel, line_height, width, dtype=dtype, device=device, requires_grad=False)
        self.helper = torch.zeros(1, line_height, width, dtype=dtype, device=device, requires_grad=False) + 1e-19
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
        if self.y is torch.nan:
            # 第一个图块将会设定行初始坐标
            self.y = y
            self.m = self.h - kh
        if y < self.y:
            # 行融合要求 y 非减，如果出现 y 减少，说明调用有误
            raise Exception(f'数据异常:: 上一个 y 轴坐标是 {self.y} 而这一个是 {y}，y 居然减少了！按照约定，行融合，y 非减！')
        if y == self.y:
            # if not(self.w > grid['x']):
            #     sys.stderr.write(f'ERROR :: merge grid {grid} out of bounds with max-width {self.w} \n')
            #     return
            # y 相等说明在同一行上，进行简单融合即可
            # 边界规范化操作
            temp_left = max(0, x)
            temp_up = self.m
            temp_right = min(self.w, max(0, x + kw))
            temp_down = min(self.h, self.m + kh)
            patch_left = max(0, -x)
            patch_up = max(0, -(y - self.y))
            patch_right = min(kw, max(0, self.w - x))
            patch_down = min(self.m + kh, self.h)
            self.target[:, temp_up: temp_down, temp_left: temp_right] += target[:, patch_up: patch_down, patch_left: patch_right]
            if helper is not None:
                self.helper[:, temp_up: temp_down, temp_left: temp_right] += helper[:, patch_up: patch_down, patch_left: patch_right]
            else:
                self.helper[:, temp_up: temp_down, temp_left: temp_right] = 1
            return
        if y > self.y:
            # y 不相等说明行向下卷动了，执行卷动
            pass

        # step 是行向下卷动的步长，注意，这个是实际卷动步长，不包括由 m 引起的逻辑卷动步长
        step = min(y - self.y, self.h - self.m)

        patch = (self.target[:, self.m: self.m + step, :] / self.helper[:, self.m: self.m + step, :]).clone()
        self.que.append(({'x': 0, 'y': self.y}, patch))

        # 行交叠
        self.target[:, :-step, :] = self.target[:, step:, :].clone()
        self.target[:, -step:, :] = 0
        self.helper[:, :-step, :] = self.helper[:, step:, :].clone()
        self.helper[:, -step:, :] = 1e-19

        # 重设行坐标
        self.y += step

        # 当 y 不对齐时，需要先进行上面一大堆卷动处理，别忘了，此时新进来的这个图块还没处理呢
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
        grid = {'x': 0, 'y': self.y}
        patch = (self.target[:, self.m:, :] / self.helper[:, self.m:, :]).clone()
        self.que.append((grid, patch))
