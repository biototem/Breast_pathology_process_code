import config
import os
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_USE
import re
from tqdm import tqdm
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from component import Timer
from utils import TiffReader,openslide_utils, image2mask, Writer
from utils.xml_utils import region_all_binary_image
from predict_method import get_npy_file_paths,preditct_dict_algorithm,TempDataset, get_kernel


# 何炳豆方法
def main():
    T = Timer()
    log_file = open('./log_jit.txt', 'a')
    # log_file = sys.stdout
    IMAGE_ROOT = config.wsi_dir
    # dataset_name = os.path.basename(IMAGE_ROOT)
    LABEL_ROOT = config.xml_label_dir
    # OUT_ROOT = config.predict_result_out_root_dir
    MASK_ROOT = config.predict_mask_dir
    PREDICT_JIT_ROOT = config.output_20X_model_seg_dir
    # VISUAL_JIT_ROOT = f'{OUT_ROOT}/seg_10x_v/{dataset_name}'
    MODEL_JIT_PATH = config.seg_20X_model_weight_path

    os.makedirs(PREDICT_JIT_ROOT, exist_ok=True)
    # os.makedirs(VISUAL_JIT_ROOT, exist_ok=True)
    os.makedirs(MASK_ROOT, exist_ok=True)
    # 嗅探数据源
    with T['jit']:
        model = torch.jit.load(MODEL_JIT_PATH)
        model.eval()
        model.cuda()
        all_list = get_npy_file_paths(IMAGE_ROOT)
        for image_path in tqdm(all_list[::-1]):
            fname = os.path.basename(image_path)
            # 文件名合规性检查
            if '.' not in fname: continue
            name, ext = re.search(r'^(.*)\.([^\.]*)$', fname).groups()
            # if ext not in ['svs', 'tif']: continue
            if os.path.exists(os.path.join(PREDICT_JIT_ROOT, f'{name}.tif')) \
                and os.path.getsize(os.path.join(PREDICT_JIT_ROOT, f'{name}.tif')) > 20:continue
            print(image_path + " is processing...")
            with T[name]:
                try:
                # 执行预测任务
                    start(
                        image_path=os.path.join(IMAGE_ROOT, fname),
                        label_path=os.path.join(LABEL_ROOT, f'{name}.xml'),
                        mask_path=os.path.join(MASK_ROOT, f'{name}.png'),
                        predict_path=os.path.join(PREDICT_JIT_ROOT, f'{name}.tif'),
                        model=model,
                        target_mpp=0.5,
                        T=T,
                         )
                except Exception as e:
                    print(e)
                    print(f"found wrong in processing of {image_path}")

    log_file.write('----------------------------------------------------------------------------------------------------\n')
    log_file.write(f'# -> jit cost time in all {T["jit"].stamp()}\n')
    log_file.write('----------------------------------------------------------------------------------------------------\n')
    names = [re.search(r'^(.*)\.([^\.]*)$', fname).groups()[0] for fname in os.listdir(IMAGE_ROOT) if re.search(r'^(.*)\.([^\.]*)$', fname).groups()[1] in config.file_suffix_list]
    for name in T.sub_timers:
        if name in names + ['jit', 'trt']: continue
        log_file.write(f'# -> %-24s cost time {T[name].stamp()}\n' % name)
    log_file.write('----------------------------------------------------------------------------------------------------\n')
    for name in names:
        log_file.write(f'# -> %-68s cost time {T[name].stamp()}\n' % name)
    log_file.flush()
    log_file.close()


def start(image_path, label_path, mask_path, predict_path,model: torch.nn.Module, target_mpp, T: Timer):
    # 准备宏信息
    with T['prepare']:
        try:
            image = TiffReader(image_path)
            origin_mpp = image.get_mpp() * 1000
        except:
            image = openslide_utils.Slide(image_path)
            origin_mpp = image.get_mpp() * 1000
        print(origin_mpp)
        best_divide = 99999
        level = 0
        level_mpp = origin_mpp
        for test_level in range(image.get_level_count()):
            mpp_in_level = origin_mpp * image.get_level_downsample(test_level)
            divide = abs(target_mpp - mpp_in_level)
            if divide < best_divide:
                best_divide = divide
                level = test_level
                level_mpp = mpp_in_level

    # 准备 mask 信息
    if not os.path.exists(mask_path):
        thumb = image.get_thumb(level=2)
        # thumb = np.asarray(thumb)
        mask = image2mask(np.array(thumb)[:,:,:3])
        if os.path.exists(label_path):
           label_mask = region_all_binary_image(thumb, image.get_level_downsample(2),label_path)
           if np.sum(label_mask>0) > 1e-6 * mask.shape[0] * mask.shape[1]:
               mask = mask * label_mask
           del label_mask
        mask = mask * 255
        # PPlot().add(thumb, mask).show()

        cv2.imwrite(mask_path, mask.astype(np.uint8))
    else:
        mask = cv2.imread(mask_path, 0)

    with T['prepare']:
        # 分辨率大换算 - 整图
        w0, h0 = image.get_level_dimension(level=0)
        wp = round(w0 * origin_mpp / target_mpp)
        hp = round(h0 * origin_mpp / target_mpp)
        hm, wm = mask.shape
        # 分辨率大换算 - 块图
        kp = 512
        # k0 = round(kp * target_mpp / origin_mpp)
        km = round(kp * wm / wp)
        kl = round(kp * target_mpp / level_mpp)

        # 坐标采样
        source = []
        for i, yp in enumerate(range(0, hp, kp // 2)):
            for j, xp in enumerate(range(0, wp, kp // 2)):
                xm = round(xp * wm / wp)
                ym = round(yp * hm / hp)
                if not mask[ym: ym + km, xm: xm + km].any():
                    continue
                x0 = round(xp * target_mpp / origin_mpp)
                y0 = round(yp * target_mpp / origin_mpp)

                source.append({
                    'svs_path': image_path,
                    'location': (x0, y0),
                    'level': level,
                    'size': (kl, kl),
                    'order': (i, j),
                    'mask_site': (xm, ym, km),
                })

        # 数据集打包
        dataset = TempDataset(source=source, ksize=kp)
        dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False, pin_memory=True)

        # 高斯核准备
        kernel = get_kernel(width=512, height=512, steep=4).cuda().unsqueeze(0).unsqueeze(0)

    # 进入算法
    with T['preditct_dict_algorithm']:
        with torch.no_grad():
            with Writer(
                output_path=predict_path,
                tile_size=kp // 2,
                dimensions=(wp, hp),
                spacing=target_mpp,
                color_type='MONOCHROME',
            ) as writer:
                preditct_dict_algorithm(
                    dataloader=dataloader,
                    model=model,
                    kernel=kernel,
                    writer=writer,
                    kp=kp,
                    mask=mask,
                    T=T,
                )

    # # 可视化
    # with T['visual']:
    #     predict = TiffReader(svs_file=predict_path)
    #     predict2visual(image=image, predict=predict, output_path=visual_path, downsample=4)



if __name__ == "__main__":
    main()
