from typing import Union, Tuple
import numpy as np
import cv2


def gaussian_kernel(size: Union[int, Tuple[int, int]], steep: float = 4) -> np.ndarray:
    """
    jassor 工具库基本代码之高斯核生成器
    :param size:  (w, h)，表示高斯核的形状
    :param steep: 陡度，数值越高，核越陡峭
    :return: [h, w]->np.float32 满足均值归一化
    例：
    gaussian_kernel(size = (3, 5), steep = 2)
    得到：
    [[0.78 0.97 0.78]
     [0.99 1.24 0.99]
     [1.07 1.34 1.07]
     [0.99 1.24 0.99]
     [0.78 0.97 0.78]]
    可以看到，1.35 / 0.78 = 1.717 是在 2 附近的一个数值
    因此说， steep 大致描述了高斯核的陡峭程度
    """
    w, h = size if isinstance(size, tuple) else (size, size)
    # cv2 获得的 GaussianKernel 是一个列向量，shape == (k, 1)
    x = cv2.getGaussianKernel(ksize=w, sigma=w / steep).T
    y = cv2.getGaussianKernel(ksize=h, sigma=h / steep)
    # 在本人面临的任务中，通常并不需要高斯核满足概率归一化条件
    # 相反，通常需要保证运算过程中的数值精度
    # 因此，本人生成的高斯核对均值做归一化
    x /= np.average(x)
    y /= np.average(y)
    return np.matmul(y, x)


# np.set_printoptions(2)
# print(gaussian_kernel((3, 5), 2))

# 关于下列注释函数的说明：
# 这里备份了 gaussian_kernel 基于 torch 的生成方法
# import torch
# def gaussian_kernel(size: int = 3, steep: float = 2, device: str = 'cpu') -> torch.Tensor:
#     """
#     provide an square matrix that matches the gaussian function
#     this may used like an kernel of weight
#     :param size:    就是高斯核的尺寸
#     :param steep:   描述高斯核的陡峭程度，由于 sigma 必须结合 size 才有意义，因此剥离出 steep 来描述它
#     :param device:
#     """
#     sigma = size / steep
#     kernel_seed = torch.tensor([[
#         -(x - size // 2) ** 2 / float(2 * sigma ** 2)
#         for x in range(size)
#     ]], dtype=torch.float64, device=device)
#     kernel_1d = torch.exp(kernel_seed)
#     # the numbers are too small ~ and there is no influence on multiple
#     kernel = torch.matmul(kernel_1d.T, kernel_1d)
#     kernel /= kernel.mean()
#     return kernel
