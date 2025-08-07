import os
import numpy as np
GPU_USE = '0'

target_mpp = 0.5  ###目标分辨率
pred_size = 512
tile_size = int(pred_size / 2)
batch_size = 16
num_workers = int(batch_size*0.5)
class_num = 8

wsi_dir = '/media/totem_data_backup2/totem/breast_test/test'
## path of WSI to be predicted
file_suffix_list = ['svs','ndpi','tif']
# WSI filename Extension
xml_label_dir = '/media/totem_data_backup2/totem/breast_test/test'
## path of wsis label in xml format to be predicted, which is not required. If there is no label file, fill in an empty string or the same path as wsi_dir
predict_result_out_root_dir = "/mnt/totem_new4/totem/breast_test/whole_process_output"
## path of root directory for saving predict output files
predict_mask_dir = os.path.join(predict_result_out_root_dir,'mask')
## path of saving WSI's region mask which would be generated in predict processing
output_10X_model_seg_dir = os.path.join(predict_result_out_root_dir,'seg_10x')
output_20X_model_seg_dir = os.path.join(predict_result_out_root_dir,'seg_20x')
# output_step_xy_dir = '/media/totem_disk/totem/jizheng/slide_predict_new/data_strange/tmp/0.5um_xy_index/'#1_---->生成的xy坐标
# output_step_Visualization= '/media/totem_disk/totem/jizheng/slide_predict_new/data_strange/tmp/0.5um_xy_index/可视化/'#1_---->生成的xy坐标---可视化

seg_10X_model_weight_path = '/media/totem_data_backup2/totem/hebingdou_scripts/乳腺癌流程整理/model_weight/jit28_10x.pth'
## 10X_model_weight_path
seg_20X_model_weight_path = '/media/totem_data_backup2/totem/hebingdou_scripts/乳腺癌流程整理/model_weight/jit8_20x.pth'
## 20X_model_weight_path

label_colors = np.asarray([
    [100, 100, 100],  # 灰色 -> background
    [255, 0, 0],  # 亮红 -> Invasive Tumor -- 浸润性肿瘤
    [180, 120, 0],  # 橙黄 -> Tumor-associated Stroma -- 肿瘤相关基质
    [120, 0, 0],  # 暗红 -> In-situ Tumor -- 原位肿瘤
    [0, 180, 0],  # 绿色 -> Healthy Glands -- 健康腺体（上皮细胞）
    [0, 0, 0],  # 黑色 -> Necrosis not in-situ -- 非原位坏死
    [255, 255, 0],  # 亮黄 -> Inflamed Stroma -- 炎症基质
    [255, 255, 255],  # 白色 -> Rest -- others
], dtype=np.uint8)


# os.makedirs(os.path.dirname(output_step_xy_dir), exist_ok=True)
# os.makedirs(os.path.dirname(output_step_Visualization), exist_ok=True)

# output_dir = '/media/totem_disk/totem/jizheng/slide_predict_new/data_strange/tmp/out_seg/'    #########分割结果路径
# out_seg_png_dir = os.path.join(output_dir+'/png/')
# out_seg_tif_dir =   os.path.join(output_dir+'/asap_tif/')
# out_seg_v_dir =   os.path.join(output_dir+'/v/')

# os.makedirs(os.path.dirname(out_seg_png_dir), exist_ok=True)
# os.makedirs(os.path.dirname(out_seg_tif_dir), exist_ok=True)
# os.makedirs(os.path.dirname(out_seg_v_dir), exist_ok=True)