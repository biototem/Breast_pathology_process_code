from typing import Dict, Tuple, Iterable
import torch

from .interface import MergerInterface


class MemoryMerger(MergerInterface):
    def __init__(self, channel: int, height: int, width: int, dtype: torch.dtype = torch.float32, device: str = 'cpu'):
        """
        基础融合库（基于 pytorch，可以在 cpu 或 gpu 上运行）
        :param channel: 通道形状描述符
        """
        self.c = channel
        self.h = height
        self.w = width
        self.target = torch.zeros(self.c, self.h, self.w, dtype=dtype, device=device, requires_grad=False)
        self.helper = torch.zeros(1, self.h, self.w, dtype=dtype, device=device, requires_grad=False) + 1e-19
        self.end = False

    def set(self, grid: Dict[str, int], target: torch.Tensor, helper: torch.Tensor) -> None:
        """
        添加融合图块，需自行传入 patch、kernel
        dtype 和 device 向创建时的指定对象对齐
        :param grid:     {x: int, y: int}
        :param target:   [c, h, w] -> dtype :: device
        :param helper:   [1, h, w] -> dtype :: device
        """
        # 内存融合计算起来最简单，以下全部数值都是在融合分辨率下进行的
        x, y = grid['x'], grid['y']
        kh, kw = target.shape[1:]
        # self.w, self.h -> merger 的宽高
        # w, h -> target 的宽高
        # 边界规范化操作
        temp_left = max(0, x)
        temp_up = max(0, y)
        temp_right = min(self.w, max(0, x + kw))
        temp_down = min(self.h, max(0, y + kh))
        patch_left = max(0, -x)
        patch_up = max(0, -y)
        patch_right = min(kw, max(0, self.w - x))
        patch_down = min(kh, max(0, self.h - y))
        # 上面这通计算可以确保下面这行代码左和右的形状永远是相同的
        self.target[:, temp_up: temp_down, temp_left: temp_right] += target[:, patch_up: patch_down, patch_left: patch_right]
        if helper is not None:
            self.helper[:, temp_up: temp_down, temp_left: temp_right] += helper[:, patch_up: patch_down, patch_left: patch_right]
        else:
            self.helper[:, temp_up: temp_down, temp_left: temp_right] = 1

    def readies(self) -> Iterable[Tuple[Dict[str, int], torch.Tensor]]:
        # 对内存融合来说，只有整个融合程序结束时，才应当考虑一次性将全部图块写出
        if not self.end:
            return
        yield {'x': 0, 'y': 0}, self.target / self.helper

    def finish(self) -> None:
        self.end = True
