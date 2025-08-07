from typing import Dict, Iterable, Tuple
import torch

from .interface import MergerInterface


class DictMerger(MergerInterface):
    def __init__(self):
        """
        字典融合方法整理，该方法由 @何炳豆 创作
        时间效率、空间效率都很高
        要求：融合的 patch 核必须是偶数宽高
        """
        self.target_map = MyDict(default=lambda: MyDict(default=lambda: 0))
        self.helper_map = MyDict(default=lambda: MyDict(default=lambda: 0))
        # pop 队列
        self.que = []

    def set(self, grid: Dict[str, int], target: torch.Tensor, helper: torch.Tensor):
        """
        添加融合图块，需自行传入 patch、kernel
        需注意的是，此处 grid 需求的不是 x、y 坐标，而是 i、j 序号
        :param grid:     {i: int, j: int}
        :param target:   [c, h, w] -> dtype :: device
        :param helper:   [1, h, w] -> dtype :: device
        """
        i = grid['i']
        j = grid['j']

        helper = helper if helper is not None else torch.ones_like(target)

        _, h, w = target.shape
        assert h == w, 'emm……应该没有这么奇葩的输出需求吧'
        assert h & 1 == 0, '必须得偶数才得了的'
        t = h // 2

        # 一张预测图分成四个局部
        # 其它三部分融合进去
        self.target_map[i + 0][j + 1] += target[:, :t, t:]
        self.target_map[i + 1][j + 0] += target[:, t:, :t]
        self.target_map[i + 1][j + 1] += target[:, t:, t:]

        self.helper_map[i + 0][j + 1] += helper[:, :t, t:]
        self.helper_map[i + 1][j + 0] += helper[:, t:, :t]
        self.helper_map[i + 1][j + 1] += helper[:, t:, t:]

        # 第一个图块直接融合然后写出
        pop_target = self.target_map[i + 0].pop(j + 0) + target[:, :t, :t]
        pop_helper = self.helper_map[i + 0].pop(j + 0) + helper[:, :t, :t]
        result = pop_target / pop_helper

        # 相隔两行的肯定已经没用了，直接清除
        self.target_map.pop(i - 1)

        # 送到输出序列里去
        self.que.append((grid, result))

    def readies(self) -> Iterable[Tuple[Dict[str, int], torch.Tensor]]:
        # set、finish 的写出都仅仅是将坐标和图块丢到 que 里，实际上要从这里取出并清空的
        # 当然啦，虽然 que 实现的是队列的功能，但实际上它并不是队列，而是用 list 代替的
        for grid, tile in self.que:
            yield grid, tile.type(torch.float32)
        self.que.clear()

    def finish(self) -> None:
        # 在这个代码逻辑下，finish 时什么也不用做
        pass


class MyDict(dict):
    def __init__(self, default: callable = None):
        super(MyDict, self).__init__()
        self.default = default

    def load(self, data: dict):
        self.update(data)
        return self

    def __setitem__(self, key, value):
        return super(MyDict, self).__setitem__(key, value)

    def __getitem__(self, item):
        if item not in self:
            self[item] = self.default and self.default()
        return super().__getitem__(item)

    def pop(self, key):
        if key in self:
            return super().pop(key)
        else:
            return self.default and self.default()
