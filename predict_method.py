import config
import os
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_USE
import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from component import Timer
from utils import TiffReader #,openslide_utils, image2mask, Writer

def get_npy_file_paths(directory):
    directory_path = Path(directory)
    npy_file_paths = []
    for file_suffix in config.file_suffix_list:
        npy_file_paths += sorted([str(file_path) for file_path in directory_path.rglob('*.' + file_suffix)])
    return npy_file_paths

def preditct_dict_algorithm(dataloader, model, kernel, writer, kp, mask, T: Timer):
    t = kp // 2
    # result_map = defaultdict(lambda: defaultdict(dict))
    result_map = MyDict(default=lambda: MyDict(default=lambda: 0))

    for patches, orders, mask_sites in dataloader:
        with T['algorithm_predict']:
            predicts = model(patches.cuda())
            predicts *= kernel
            predicts = predicts.permute(0, 2, 3, 1)       #.cpu().detach()
            predicts[0, 0, 0, 0].item()

        for b in range(predicts.shape[0]):
            with T['algorithm_merge']:
                predict = predicts[b, :, :, :]
                i, j = int(orders[0][b]), int(orders[1][b])

                # 一张预测图分成四个局部
                part_1 = predict[:t, :t, :]
                part_2 = predict[:t, t:, :]
                part_3 = predict[t:, :t, :]
                part_4 = predict[t:, t:, :]

                # 其它三部分融合进去
                result_map[i + 0][j + 1] += part_2
                result_map[i + 1][j + 0] += part_3
                result_map[i + 1][j + 1] += part_4

                # 第一个图块直接融合然后写出
                pop_result = result_map[i + 0].pop(j + 0) + part_1

                # 相隔两行的肯定已经没用了，直接清除
                result_map.pop(i - 1)

            with T['algorithm_post_process']:
                # * mask
                xm, ym, km = mask_sites[0][b], mask_sites[1][b], mask_sites[2][b]
                m = mask[ym: ym + km, xm: xm + km].astype(bool).astype(np.float32)
                m = torch.tensor(m).cuda()[None, None, :, :]
                m = torch.nn.functional.interpolate(m, (t * 2, t * 2), mode='nearest')
                pop_result = pop_result * m[0, 0, :t, :t, None]

                # argmax
                pop_result = pop_result.argmax(axis=2).cpu().numpy().astype(dtype=np.uint8)

            with T['algorithm_write']:
                writer.write(pop_result, x=j * t, y=i * t)
    result_map.clear()

class TempDataset(Dataset):
    STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))

    def __init__(self, source, ksize):
        super().__init__()
        self.source = source
        self.ksize = ksize
        self.cache = {}

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        info = self.source[index]

        svs_path = info['svs_path']

        if svs_path not in self.cache:
            self.cache[svs_path] = TiffReader(svs_path)
        image = self.cache[svs_path]

        patch = image.read_region(location=info['location'], level=info['level'], size=info['size'], as_array=True)

        patch = cv2.resize(patch, (self.ksize, self.ksize))
        patch = ((patch / 255 - self.MEAN) / self.STD).transpose((2, 0, 1)).astype(np.float32)
        patch = torch.tensor(patch)

        return patch, info['order'], info['mask_site']


def get_kernel(width: int, height: int, steep: float):
    # create gaussian kernel
    kernel_x = cv2.getGaussianKernel(ksize=width, sigma=width / steep)
    kernel_x /= np.average(kernel_x)
    kernel_y = cv2.getGaussianKernel(ksize=height, sigma=height / steep)
    kernel_y /= np.average(kernel_y)
    kernel = np.matmul(kernel_y, kernel_x.T)
    return torch.tensor(kernel, requires_grad=False)


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


COLORS = config.label_colors
def predict2visual(image: TiffReader, predict: TiffReader, output_path: str, downsample: int):

    wp, hp = predict.get_level_dimension(level=0)
    hv = round(hp / downsample)
    wv = round(wp / downsample)

    level_image = 0
    value_image = None
    for test_level in range(image.get_level_count()):
        hl, wl = image.get_level_dimension(level=test_level)
        test_value = (hl / hv - 1) ** 2 + (wl / wv - 1) ** 2
        if value_image is None or test_value < value_image:
            value_image = test_value
            level_image = test_level

    level_predict = 0
    value_predict = None
    for test_level in range(predict.get_level_count()):
        hl, wl = predict.get_level_dimension(level=test_level)
        test_value = (hl / hv - 1) ** 2 + (wl / wv - 1) ** 2
        if value_predict is None or test_value < value_predict:
            value_predict = test_value
            level_predict = test_level

    image_thumb = image.get_thumb(level=level_image)
    image_thumb = np.asarray(image_thumb)
    image_thumb = cv2.resize(image_thumb, dsize=(wv, hv))

    predict_thumb = predict.get_thumb(level=level_predict)
    predict_thumb = np.asarray(predict_thumb)
    predict_thumb = cv2.resize(predict_thumb, dsize=(wv, hv), interpolation=cv2.INTER_NEAREST)
    predict_thumb = COLORS[predict_thumb]

    visual_thumb = image_thumb * 0.75 + predict_thumb * 0.25

    visual_thumb = cv2.cvtColor(visual_thumb.astype(np.uint8), cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, visual_thumb.astype(np.uint8))

