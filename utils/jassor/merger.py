from typing import Dict
import traceback
import torch


class Merger(object):
    def __init__(self, channel: int, device: str = 'cpu'):
        """
        基础融合库（基于 pytorch，可以在 cpu 或 gpu 上运行）
        :param channel: 通道形状描述符
        :param device: 设备描述符，建议 cpu 即可
        """
        self.c = channel
        self.w = self.h = 0
        self.target = self.helper = None
        self.device = device

    def set(self, target: torch.Tensor, helper: torch.Tensor, grid: Dict[str, int]) -> None:
        """
        添加融合图块，需自行传入 patch、kernel
        :param target:   [c, h, w] -> torch.float32
        :param helper:  [h, w] -> torch.float32         若 kernel == None，则 helper 对应区域设为 1
        :param grid:    {x: int, y: int}
        """
        x, y = grid['x'], grid['y']
        h, w = target.shape[1:]
        # self.w, self.h -> merger 的宽高
        # w, h -> target 的宽高
        temp_left = max(0, x)
        temp_up = max(0, y)
        temp_right = min(self.h, x + w)
        temp_down = min(self.h, y + h)
        patch_left = max(0, -x)
        patch_up = max(0, -y)
        patch_right = min(w, self.w - x)
        patch_down = min(h, self.h - y)
        self.target[:, temp_up: temp_down, temp_left: temp_right] += target[:, patch_up: patch_down, patch_left: patch_right]
        if helper is None:
            self.helper[:, temp_up: temp_down, temp_left: temp_right] = 1
        else:
            self.helper[0, temp_up: temp_down, temp_left: temp_right] += helper[patch_up: patch_down, patch_left: patch_right]

    def tail(self) -> torch.Tensor:
        return self.target / self.helper

    def with_shape(self, w: int, h: int):
        del self.helper
        del self.target
        self.w = w
        self.h = h
        self.helper = torch.zeros(1, self.h, self.w, dtype=torch.float32, device=self.device) + 1e-17
        self.target = torch.zeros(self.c, self.h, self.w, dtype=torch.float32, device=self.device)
        return self


# 关于下列注释函数的说明：
# 理论上讲，该 merger 可以实现对 batch 预测结果做融合
# 但考虑到实践中几乎用不到该方法，因此舍弃这部分冗余的复杂逻辑（有需求时请参照下列代码进行修改）
# class TorchMerger(object):
#     def __init__(self, class_num: int = 8, kernel_size: int = 256, kernel_steep: float = 2, zoom: float = 1, device: int = 0):
#         # self.kns = {self.__kernel__(zoom, steep) for zoom, steep in kernel_params.items()}
#         self.C = class_num
#         self.ksize = kernel_size
#         self.W = self.H = 0
#         self.target = self.helper = None
#         self.zoom = zoom
#         # kernel 用于 同预测结果相乘
#         self.kernel = self.__kernel__(steep=kernel_steep, device=device)
#         # kns 用于 对齐切图方法
#         self.__kns__ = {}
#
#     def __kernel__(self, steep: float, device: int):
#         kernel = gaussian_kernel(size=self.ksize, steep=steep)
#         kernel = np.expand_dims(kernel, axis=(0, 1))
#         return torch.tensor(kernel).to(device)
#
#     def with_shape(self, W: int, H: int):
#         self.W = W
#         self.H = H
#         return self
#
#     def set(self, patches, grids):
#         # 计算图
#         targets = patches * self.kernel
#         targets = targets.cpu().numpy().transpose((0, 2, 3, 1))
#         # x,y 标定原图起止点坐标,size标定原图切图尺寸(用于缩放),当grids中不存在size属性时,默认全部按照patch_size切图(对应缩放为1)
#         # print(grids)
#         for x, y, target in zip(grids['x'], grids['y'], targets):
#             # print(x, y)
#             x, y = int(x / self.zoom), int(y / self.zoom)
#             size = self.ksize // self.zoom
#             # (x, y) 为图片中心，而此处需要将其处理为左上角坐标
#             x -= size // 2
#             y -= size // 2
#             if size != self.ksize:
#                 target = cv2.resize(target, (size, size))
#             kernel = self.kns(size)
#             temp_left = max(0, x)
#             temp_up = max(0, y)
#             temp_right = min(self.W, x + size)
#             temp_down = min(self.H, y + size)
#             patch_left = max(0, -x)
#             patch_up = max(0, -y)
#             patch_right = min(size, self.W - x)
#             patch_down = min(size, self.H - y)
#             # print(x, y, target[patch_up: patch_down, patch_left: patch_right, :].shape, self.target[temp_up: temp_down, temp_left: temp_right, :].shape)
#             self.target[temp_up: temp_down, temp_left: temp_right, :] += target[patch_up: patch_down, patch_left: patch_right, :]
#             self.helper[temp_up: temp_down, temp_left: temp_right, :] += kernel[patch_up: patch_down, patch_left: patch_right, :]
#
#     def tail(self):
#         return self.target / self.helper
#
#     def kns(self, size: int):
#         if size not in self.__kns__:
#             kn = self.kernel[0, 0, :, :].cpu().numpy()
#             self.__kns__[size] = np.expand_dims(cv2.resize(kn, (size, size)), axis=2)
#         return self.__kns__[size]
#
#     def __enter__(self):
#         self.helper = np.zeros(shape=(self.H, self.W, 1), dtype=np.float32) + 1e-17
#         self.target = np.zeros(shape=(self.H, self.W, self.C), dtype=np.float32)
#         return self
#
#     def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
#         self.W = 0
#         self.H = 0
#         self.helper = None
#         self.target = None
#         return False


# 关于下列注释函数的说明：
# 理论上讲，该 merger 可以实现多返回值热图的融合
# 但考虑到实践中几乎用不到该方法，因此舍弃这部分冗余的复杂逻辑（有需求时请参照下列代码进行修改）
# class Merger(object):
#     def __init__(
#             self,
#             model_return_channels,
#             width: int,
#             height: int,
#             kernel_size: int,
#             kernel_steep: float,
#             device: str,
#     ):
#         # self.kns = {self.__kernel__(zoom, steep) for zoom, steep in kernel_params.items()}
#         self.k = kernel_size
#         self.w = width
#         self.h = height
#         # target 用来存放多个返回结果，helper 只存一个高斯核
#         self.targets: List[torch.Tensor] = [
#             torch.zeros(channel, self.h, self.w, dtype=torch.float64, device=device)
#             for channel in model_return_channels
#         ]
#         self.helper: torch.Tensor = torch.zeros(1, self.h, self.w, dtype=torch.float64, device=device) + 1e-17
#         # kernel 用于 同预测结果相乘 (1, )
#         self.kernel = gaussian_kernel(size=kernel_size, steep=kernel_steep, device=device)[None, None, :, :]
#
#     def set(self, patches_group: Iterable[torch.Tensor], grids: List[Tuple[int, int]]) -> None:
#         # 拆 returns
#         for target, patches in zip(self.targets, patches_group):
#             helper = self.helper
#             # 高斯融合
#             patches = patches * self.kernel
#             # 贴片
#             for (x, y), patch in zip(grids, patches):
#                 target[:, y: y+self.k, x: x+self.k] += patch
#                 helper[:, y: y+self.k, x: x+self.k] += self.kernel[0, :, :, :] / len(self.targets)
#
#     def tail(self) -> List[torch.Tensor]:
#         return [target / self.helper for target in self.targets]


# 关于下列注释函数的说明：
# 理论上讲，该 merger 可以实现任意尺寸任意角度的图融合
# 但考虑到实践中几乎用不到该方法，因此舍弃这部分冗余的复杂逻辑（有需求时请参照下列代码进行修改）
# class Merger(object):
#     def __init__(self, shape: Tuple[int, int, int], xyc_dim: Tuple[int, int, int] = (0, 1, 2), kernel_steep: float = 2):
#         self.x_dim, self.y_dim, self.c_dim = xyc_dim
#         self.W, self.H, self.C = shape[self.x_dim], shape[self.y_dim], shape[self.c_dim]
#         self.helper = np.zeros(shape=(self.H, self.W, 1), dtype=np.float32) + 1e-17
#         self.target = np.zeros(shape=(self.H, self.W, self.C), dtype=np.float32)
#         self.kernel_steep = kernel_steep
#         self.kns = {}
#
#     def __kernel__(self, size):
#         if size not in self.kns:
#             self.kns[size] = np.expand_dims(gaussian_kernel(size=size, steep=self.kernel_steep), 2)
#         return self.kns[size]
#
#     def set(self, data, site: Tuple[int, int], size: int, degree: float = 0, scale: float = 1):
#         data = data.transpose((self.y_dim, self.x_dim, self.c_dim))
#         # 计算图
#         kernel = self.__kernel__(size)
#         target = data * kernel
#         kernel = rotate(kernel, degree, scale)
#         target = rotate(target, degree, scale)
#         # 计算原图坐标
#         h, w = kernel.shape[:2]
#         x0, y0 = site
#         x1, y1 = round(x0 - w / 2), round(y0 - h / 2)
#         # 拼图
#         ml = max(0, x1)
#         mu = max(0, y1)
#         mr = min(self.W, x1+w)
#         md = min(self.H, y1+h)
#         pl = max(0, -x1)
#         pu = max(0, -y1)
#         pr = min(w, self.W-x1)
#         pd = min(h, self.H-y1)
#         if self.target[mu:md, ml:mr, :].shape != target[pu:pd, pl:pr, :].shape:
#             print(ml, mu, mr, md, pl, pu, pr, pd)
#         self.target[mu:md, ml:mr, :] += target[pu:pd, pl:pr, :]
#         self.helper[mu:md, ml:mr, :] += kernel[pu:pd, pl:pr, :]
#
#     def tail(self):
#         return self.target / self.helper
#
#
# def test():
#     tif_path = '/media/totem_disk/totem/guozunhu/Project/kidney_biopsy/tiff_data/tif/H1804782 1 HE/H1804782 1 HE_Wholeslide_默认_Extended.tif'
#     cp = SlideRotateCropper(filepath=tif_path)
#     P = 8000
#     S = 1000
#
#     merger = Merger(shape=(S, 3, S), xyc_dim=(0, 2, 1), kernel_steep=8)
#     for i in range(55):
#         x = np.random.randint(0, S)
#         y = np.random.randint(0, S)
#         degree = np.random.randint(-44, 45)
#         scale = np.random.random() * 3 + 0.25
#         img = cp.get(site=(P + x, P + y), size=128, degree=degree, scale=scale)[:, :, :3].transpose((1, 2, 0))
#         merger.set(img, site=(x, y), size=128, degree=-degree, scale=1/scale)
#         tl = merger.tail().astype(np.uint8)
#         plt.imshow(tl)
#         plt.show()
#
#
# if __name__ == '__main__':
#     test()
