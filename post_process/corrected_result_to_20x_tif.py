import os,sys
sys.path.append('../')
import imageio.v3
import numpy as np
from utils.asap_slide import Writer
import config
import math
import cv2
from tiffslide import TiffSlide
import  multiprocessing  as mul
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import glob

def get_npy_file_paths(directory,out_dir3):
    npy_file_paths = []
    for file_path in glob.glob(os.path.join(directory,"*.tif")):
        if os.path.getsize(file_path) < 500:continue
        file_name = os.path.basename(file_path).split(".tif")[0]
        if file_name.startswith("temp"):continue
        if os.path.exists(os.path.join(out_dir3,file_name + ".tif")):continue
        npy_file_paths.append(file_path)
    # directory_path = Path(directory)
    # npy_file_paths = sorted([str(file_path) for file_path in directory_path.rglob('*.tif') if os.path.getsize(file_path) > 500])
    return sorted(npy_file_paths)

def fun(path_list):
    ##################################
    seg_path = path_list[0]

    ys_2um_seg_path = path_list[1]#str(path1).replace(dir1,out_dir).replace('.tif','_20x_ys.png')#原始分割结果，2um下的png图像路径
    seg_2um_correct_path = path_list[2]#str(path1).replace(dir1,out_dir).replace('.tif','.png')#原始分割结果--修正，2um下的png图像路径
    output_path  = path_list[3]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

     ###########################

    seg_slide = TiffSlide(seg_path)
    seg_ys = imageio.v3.imread(ys_2um_seg_path)
    seg_correct = imageio.v3.imread(seg_2um_correct_path)
    mask_change = np.uint8((seg_ys - seg_correct) != 0)
    mpp_um = seg_slide.properties['tiffslide.mpp-x']
    wsi_w_l0,wsi_h_l0 = seg_slide.level_dimensions[0]
    mask_level_ds = (wsi_w_l0/mask_change.shape[1])
    assert seg_slide.properties['tiffslide.level[0].tile-width'] == seg_slide.properties['tiffslide.level[0].tile-height'] , 'ASAP写的tif中，title应该是H和W是一致的'
    tilesize = seg_slide.properties['tiffslide.level[0].tile-width']
    mask_level_tilesize = round(tilesize/mask_level_ds)
    w_count = math.ceil(wsi_w_l0 / tilesize)
    h_count = math.ceil(wsi_h_l0 / tilesize)
    with Writer(
            output_path=output_path,
            tile_size=tilesize,
            dimensions=(wsi_w_l0,wsi_h_l0),
            spacing=mpp_um,
            color_type = 'MONOCHROME'
    ) as writer1:
        for w in range(w_count):
            for h in range(h_count):
                title = seg_slide.read_region((w*tilesize,h*tilesize),0,(tilesize,tilesize),as_array=True)[:,:,0]
                if np.sum(title)==0:continue
                title_diff = mask_change[h * mask_level_tilesize:h * mask_level_tilesize + mask_level_tilesize, w * mask_level_tilesize:w * mask_level_tilesize + mask_level_tilesize]
                if np.sum(title_diff)==0:
                    writer1.write(tile=title, x=w * tilesize, y=h * tilesize)
                else:
                    title_correct = seg_correct[h * mask_level_tilesize:h * mask_level_tilesize + mask_level_tilesize, w * mask_level_tilesize:w * mask_level_tilesize + mask_level_tilesize]
                    title_correct = cv2.resize(title_correct, (tilesize, tilesize), interpolation=cv2.INTER_NEAREST)
                    title_diff = cv2.resize(title_diff, (tilesize, tilesize), interpolation=cv2.INTER_NEAREST)
                    list_tmp = np.unique(title*title_diff).tolist()
                    for type in list_tmp:
                        if type==0:continue
                        tmp_mask1 = np.uint8(title==type)
                        contours, _ = cv2.findContours(tmp_mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
                        for cnt in contours:
                            mask_cnt = np.zeros((tilesize,tilesize),dtype=np.uint8)
                            mask_cnt = cv2.drawContours(mask_cnt, [cnt], 0, 1, -1)
                            title_修正_in_mask_cnt = mask_cnt*title_correct
                            # 使用unique函数获取唯一元素和它们的索引
                            unique_elements, counts = np.unique(title_修正_in_mask_cnt, return_counts=True)
                            # 找到非零元素的索引
                            nonzero_indices_new = unique_elements[unique_elements != 0]
                            counts_new = counts[unique_elements != 0]
                            if counts_new.shape[0]>0:
                                most_common_element = nonzero_indices_new[np.argmax(counts_new)]
                                title[mask_cnt==1] = int(most_common_element)
                    writer1.write(tile=title, x=w * tilesize, y=h * tilesize)



if __name__ == '__main__':
    dir_20x_seg_tif = config.output_20X_model_seg_dir
    dir_ys = os.path.join(config.predict_result_out_root_dir,'result_merge','ys')
    dir_final_result = os.path.join(config.predict_result_out_root_dir,'result_merge','10_20x_convert')
    tif_out_dir = os.path.join(config.predict_result_out_root_dir,'result_merge','10_20x_convert_final')
    list_tmp = get_npy_file_paths(dir_20x_seg_tif,tif_out_dir)

    list_all = []
    for file_path in list_tmp:
        file_name = os.path.basename(file_path).split(".tif")[0]
        path_20x_segtif = file_path
        path_20x_2um_png = os.path.join(dir_ys, file_name + '_20x_ys.png')
        path_seg_2um_correct = os.path.join(dir_final_result, file_name + '.png')
        out_path = os.path.join(tif_out_dir,file_name + '.tif')
        list_tmp1 = [path_20x_segtif,path_20x_2um_png,path_seg_2um_correct,out_path]
        list_all.append(list_tmp1)
    pool = mul.Pool(3)
    rel = pool.map(fun,list_all)




