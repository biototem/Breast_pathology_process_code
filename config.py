import os
import numpy as np
GPU_USE = '0'

target_mpp = 0.5  ### target mpp for 20x model
seg_pred_size = 512
tile_size = int(seg_pred_size / 2)
batch_size = 16
num_workers = int(batch_size*0.5)
class_num = 8

wsi_dir = '/breast_test/test'
## path of WSI to be predicted
file_suffix_list = ['svs','ndpi','tif']
# WSI filename Extension
xml_label_dir = '/breast_test/test'
## path of wsis label in xml format to be predicted, which is not required. If there is no label file, fill in an empty string or the same path as wsi_dir
predict_result_out_root_dir = "/breast_test/whole_process_output"
## path of root directory for saving predict output files
predict_mask_dir = os.path.join(predict_result_out_root_dir,'mask')
## path of saving WSI's region mask which would be generated in predict processing
output_10X_model_seg_dir = os.path.join(predict_result_out_root_dir,'seg_10x')
output_20X_model_seg_dir = os.path.join(predict_result_out_root_dir,'seg_20x')

blurred_mask_dir = os.path.join(predict_result_out_root_dir,'blurred_mask')

seg_10X_model_weight_path = './model_weight/jit28_10x.pth'
## 10X_model_weight_path
seg_20X_model_weight_path = './model_weight/jit8_20x.pth'
## 20X_model_weight_path

qc_10x_model_weight_path = './model_weight/HE_10X.pth'
qc_20x_model_weight_path = './model_weight/HE_20X.pth'

invasive_blur_box_xml_output_dir = os.path.join(predict_result_out_root_dir,'invasive_blur_box_xml_output_dir')

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
