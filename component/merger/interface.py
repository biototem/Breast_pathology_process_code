import abc
from typing import Dict, Tuple, Iterable
import torch


class MergerInterface(abc.ABC):
    def set(self, grid: Dict[str, int], target: torch.Tensor, helper: torch.Tensor) -> None:
        """
        添加融合图块，根据具体设计，可以是在内存中添加，也可以是在显存中添加
        grid 和 target 是必须提供的，而 helper 则不必要
        当不提供 helper 时，约定相关区域的权重强制设为 1
        :param grid:    {'x': int, 'y': int}                    # 用来指示位置的坐标（请自行变换至融合分辨率下，但不要计算卷动）
        :param target:  [c, h, w] -> torch.float32              # 融合好的图块（没错，本工具不负责高斯核乘积）
        :param helper:  [1, h, w] -> torch.float32              # 融合附加的权重核（没错，本工具不负责高斯核乘积）
        """
        raise NotImplemented

    def readies(self) -> Iterable[Tuple[Dict[str, int], torch.Tensor]]:
        """
        获得阶段性的融合结果 [grid, target]:
            grid: {'x': int, 'y': int}
            target: torch.Tensor
            以上：grid 表示 target 的左上角锚点，target 则为每一项被取出的张量元素，至于这个张量元素具体表示什么，取决于具体的实现细节约定
            目前的想法是：将一切融合好的原始结果取出，至于之后做什么，在外面做
        约定采用生成器模式 (generator)
        代码中将采用 yield 输出全部可返回结果
        对内存融合来说，只有当调用 finish 后才一次性返回全部结果
        对行、列融合来说，只有当一行、一列融合完成后才返回一行、一列的结果
        对字典融合来说，每次调用都可能返回左上角的结果
        """
        raise NotImplemented

    def finish(self) -> None:
        """
        声明融合结束
        当声明融合结束时，融合器将全部可返回而尚未返回的结果加入 readies 的结果中
        """
        raise NotImplemented
