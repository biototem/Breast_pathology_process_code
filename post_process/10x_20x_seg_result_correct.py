import os,sys
sys.path.append('../')
import cv2
import time
import imageio.v3
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import tiffslide
# from scipy.spatial import distance
# from pathlib import Path
import glob
import config

COLORS = config.label_colors


def get_npy_file_paths(directory_20x,directory_10x,out_dir3):
    npy_file_paths = []
    for file_path in glob.glob(os.path.join(directory_20x,"*.tif")):
        if os.path.getsize(file_path) < 500:continue
        file_name = os.path.basename(file_path).split(".tif")[0]
        if file_name.startswith("temp"):continue
        if not os.path.exists(os.path.join(directory_10x,os.path.basename(file_path))):continue
        if os.path.exists(os.path.join(out_dir3,file_name + ".png")):continue
        npy_file_paths.append(file_path)
    # directory_path = Path(directory)
    # npy_file_paths = sorted([str(file_path) for file_path in directory_path.rglob('*.tif') if os.path.getsize(file_path) > 500])
    return sorted(npy_file_paths)

def xiuz_524(seg_array_2um):
    mask_xiuz = ((seg_array_2um == 1) | (seg_array_2um == 3) | (seg_array_2um == 4) | (seg_array_2um == 5)).astype(np.uint8)
    mask_cnts_xiuz, _ = cv2.findContours(mask_xiuz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in mask_cnts_xiuz:
        x, y, w, h = cv2.boundingRect(cnt)
        cnt_T = cnt[:,0,:][:,::-1].T
        tmp_array = seg_array_2um[cnt_T[0], cnt_T[1]]   #目的是获取坐标的分割结果情况
        cnt_new = cnt.copy()
        cnt_new[:,:,0] = cnt_new[:,:,0]-x
        cnt_new[:,:,1] = cnt_new[:,:,1]-y
        tmp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(tmp_mask, [cnt_new], 0, 1, cv2.FILLED)
        tmp_seg_ee_ys = seg_array_2um[y:y + h, x:x + w]
        tmp_seg_ee = tmp_seg_ee_ys * tmp_mask
        tmp_mask = np.uint8((tmp_seg_ee==1)|(tmp_seg_ee==3)|(tmp_seg_ee==4))
        if np.sum(tmp_mask)<2:continue
        list00 = list(np.unique(tmp_array))
        if len(list00)>1:
            is_Truexiuz = True
        elif np.unique(tmp_seg_ee).shape[0]>2:
            is_Truexiuz = True
        else:
            is_Truexiuz = False
        if is_Truexiuz:
            s = np.sum(tmp_mask)
            cls_a1 = np.sum(tmp_seg_ee == 1)
            cls_a3 = np.sum(tmp_seg_ee == 3)
            cls_a4 = np.sum(tmp_seg_ee == 4)
            my_list = [cls_a1,cls_a3,cls_a4]
            list_cls = [1,3,4]
            my_list1 = [cls_a3,cls_a4]
            list_cls1 = [3,4]
            max_cls_c = max(my_list)
            mask_tmp345 = np.uint8((tmp_seg_ee == 3) | (tmp_seg_ee == 4) | (tmp_seg_ee == 5))
            mask_cnts_tmp, _ = cv2.findContours(mask_tmp345, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(tmp_seg_ee, mask_cnts_tmp, -1, 0, cv2.FILLED)
            if np.sum(tmp_seg_ee == 1) > 2:
                max_cls = list_cls[my_list.index(max_cls_c)]
            else:
                max_cls = list_cls1[my_list1.index(max(my_list1))]
                tmp_seg_ee_ys[tmp_mask == 1] = max_cls
                continue
            if max_cls_c >= s * 0.8:
                tmp_seg_ee_ys[tmp_mask == 1] = max_cls
            elif cls_a1 > s * 0.25:
                tmp_seg_ee_ys[tmp_mask == 1] = 1
            else:
                if cls_a3 >= cls_a4:
                    tmp_seg_ee_ys[tmp_mask == 1] = 3
                else:
                    tmp_seg_ee_ys[tmp_mask == 1] = 4
            del tmp_mask
            seg_array_2um[y:y + h, x:x + w] = tmp_seg_ee_ys
    return seg_array_2um


def fun(path_20x):

    path_10x = str(path_20x).replace(dir_20x,dir_10x)
    """
        既然已经知道乳腺癌的模型是0.5um下训练的，也就意味着分割结果0级一定是0.5um, 误差应该要小于0.01
        同理按照1 / 2 / 4 的下采样，也就意味着第2层级就是对应2um的分割结果，这样简单取分割结果
    """
    time1 = time.time()
    level_tmp = 2
    print(path_20x)
    seg_slide_20x = tiffslide.TiffSlide(path_20x)
    wsi_mpp = round(float(seg_slide_20x.properties['tiffslide.mpp-x']),2)
    assert abs(wsi_mpp-0.5)<0.01,      '看上面分析'
    wsi_mpp_level2 = round(float(seg_slide_20x.properties['tiffslide.mpp-x']), 2)*seg_slide_20x.level_downsamples[2]
    assert abs(wsi_mpp_level2-2)<0.01, '看上面分析'
    seg_array_2um_20x = seg_slide_20x.read_region((0,0),level_tmp,seg_slide_20x.level_dimensions[level_tmp],as_array=True)[:,:,0]

    level_tmp = 1
    seg_slide_10x = tiffslide.TiffSlide(path_10x)
    wsi_mpp = round(float(seg_slide_10x.properties['tiffslide.mpp-x']),2)
    assert abs(wsi_mpp-1)<0.01,      '看上面分析'
    wsi_mpp_level2 = round(float(seg_slide_10x.properties['tiffslide.mpp-x']), 1)*seg_slide_10x.level_downsamples[1]
    assert abs(wsi_mpp_level2-2)<0.01, '看上面分析'
    seg_array_2um_10x = seg_slide_10x.read_region((0,0),level_tmp,seg_slide_10x.level_dimensions[level_tmp],as_array=True)[:,:,0]
    if  seg_array_2um_20x.shape!=seg_array_2um_10x.shape:
        assert abs(seg_array_2um_20x.shape[0] - seg_array_2um_10x.shape[0] ) < 2
        assert abs(seg_array_2um_20x.shape[1] - seg_array_2um_10x.shape[1] ) < 2
        seg_array_2um_10x = cv2.resize(seg_array_2um_10x,(seg_array_2um_20x.shape[1],seg_array_2um_20x.shape[0]),interpolation=cv2.INTER_NEAREST)

    path22 = str(path_20x).replace(dir_20x,out_dir_ys).replace('.tif', '_10x_ys.png')
    os.makedirs(os.path.dirname(path22),exist_ok=True)
    imageio.imwrite(path22, seg_array_2um_10x)
    seg_array_2um_10x = xiuz_524(seg_array_2um_10x)
    path22 = str(path_20x).replace(dir_20x,out_dir1).replace('.tif', '_10x_convert.png')
    os.makedirs(os.path.dirname(path22),exist_ok=True)
    imageio.imwrite(path22, seg_array_2um_10x)



    path22 = str(path_20x).replace(dir_20x,out_dir_ys).replace('.tif', '_20x_ys.png')
    os.makedirs(os.path.dirname(path22),exist_ok=True)
    imageio.imwrite(path22, seg_array_2um_20x)
    seg_array_2um_20x = xiuz_524(seg_array_2um_20x)
    path22 = str(path_20x).replace(dir_20x,out_dir2).replace('.tif', '_20x_convert.png')
    os.makedirs(os.path.dirname(path22),exist_ok=True)
    imageio.imwrite(path22, seg_array_2um_20x)



    mask_xiuz = np.uint8(seg_array_2um_20x==1)
    mask_cnts_xiuz, _ = cv2.findContours(mask_xiuz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # fhfh = 0
    for cnt in mask_cnts_xiuz:
        x, y, w, h = cv2.boundingRect(cnt)
        cnt_new = cnt.copy()
        cnt_new[:,:,0] = cnt_new[:,:,0]-x
        cnt_new[:,:,1] = cnt_new[:,:,1]-y
        tmp_mask_20x = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(tmp_mask_20x, [cnt_new], 0, 1, cv2.FILLED)
        tmp_seg_ee_ys_20x = seg_array_2um_20x[y:y + h, x:x + w]
        tmp_seg_ee_20x = tmp_seg_ee_ys_20x * tmp_mask_20x

        tmp_seg_ee_ys_10x = seg_array_2um_10x[y:y + h, x:x + w]
        tmp_seg_ee_10x = tmp_seg_ee_ys_10x * tmp_mask_20x
        tmp_mask_10x = np.uint8(tmp_seg_ee_10x == 1)

        tmp_mask_20x = np.uint8(tmp_seg_ee_20x==1)
        tmp_sum_20x_1 = np.sum(tmp_mask_20x)
        if tmp_sum_20x_1<75:
            tmp_seg_ee_ys_20x[tmp_mask_20x == 1] = 7
            seg_array_2um_20x[y:y + h, x:x + w] = tmp_seg_ee_ys_20x
            continue

        if (np.sum(tmp_mask_10x)/(tmp_sum_20x_1+1e-9))<0.5:
            cls_a3 = np.sum(tmp_seg_ee_10x == 3)
            cls_a4 = np.sum(tmp_seg_ee_10x == 4)
            my_list1 = [cls_a3,cls_a4]
            list_cls1 = [3,4]
            if max(my_list1)>1:
                max_cls = list_cls1[my_list1.index(max(my_list1))]
                tmp_seg_ee_ys_20x[tmp_mask_20x == 1] = max_cls
            elif 1 in tmp_mask_10x:
                pass
            else:
                tmp_seg_ee_10x_flatten = tmp_seg_ee_10x.flatten()
                counts = np.bincount(tmp_seg_ee_10x_flatten,weights=tmp_seg_ee_10x_flatten)
                max_cls = int(np.argmax(counts))
                tmp_seg_ee_ys_20x[tmp_mask_20x == 1] = max_cls
            seg_array_2um_20x[y:y + h, x:x + w] = tmp_seg_ee_ys_20x
    time2 = time.time()
    path22 = str(path_20x).replace(dir_20x,out_dir3).replace('.tif', '.png')
    os.makedirs(os.path.dirname(path22),exist_ok=True)
    imageio.imwrite(path22, seg_array_2um_20x)
    print("Finished inference %s, needed %.2f sec." % (os.path.basename(path_20x),time2 - time1))


if __name__ == '__main__':
    import multiprocessing as mul

    dir_20x = config.output_20X_model_seg_dir
    dir_10x = config.output_10X_model_seg_dir

    out_dir = os.path.join(config.predict_result_out_root_dir,'result_merge')
    out_dir_ys = os.path.join(out_dir,'ys')
    out_dir1 = os.path.join(out_dir, '10x_convert')
    out_dir2 = os.path.join(out_dir, '20x_convert')
    out_dir3 = os.path.join(out_dir, '10_20x_convert')
    list_correct = get_npy_file_paths(dir_20x,dir_10x,out_dir3)
    pool = mul.Pool(1)
    pool.map(fun,list_correct)
