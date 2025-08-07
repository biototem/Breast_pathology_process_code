import os
import pickle
import math
import shutil

import imageio.v3
import numpy as np
import cv2
import openslide
import tiffslide
from PIL import Image
from scipy.spatial import distance
Image.MAX_IMAGE_PIXELS = None
import  multiprocessing  as mul
from torchvision.transforms import autoaugment, transforms
from utils.imagescope_xml_utils import ImageScopeXmlReader,ImageScopeXmlWriter

train_transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


COLORS = np.asarray([
    [100, 100, 100],  # 灰色 -> 背景
    [255, 0, 0],  # 亮红 -> Invasive Tumor -- 浸润性肿瘤
    [180, 120, 0],  # 橙黄 -> Tumor-associated Stroma -- 肿瘤相关基质
    [120, 0, 0],  # 暗红 -> In-situ Tumor -- 原位肿瘤
    [0, 180, 0],  # 绿色 -> Healthy Glands -- 健康腺体（上皮细胞）
    [0, 0, 0],  # 黑色 -> Necrosis not in-situ -- 非原位坏死
    [255, 255, 0],  # 亮黄 -> Inflamed Stroma -- 炎症基质
    [255, 255, 255],  # 白色 -> Rest -- 余项
], dtype=np.uint8)


def get_最靠近指定分辨率下的层级(img_slide, target_mpp):
    最靠近目标分辨率下的层级 = 0
    jfjfp_pfpf = 99999
    min_img_lever = img_slide.level_count
    for tmp_i in range(min_img_lever):
        mpp_in_level = img_slide.properties['tiffslide.mpp-x'] * img_slide.level_downsamples[tmp_i]
        ooooo_tmp = abs(target_mpp - mpp_in_level)
        if ooooo_tmp < jfjfp_pfpf:
            jfjfp_pfpf = ooooo_tmp
            最靠近目标分辨率下的层级 = tmp_i
    return 最靠近目标分辨率下的层级
def fun(list111111):
    # try:
    wsi_path, seg_convert_path,mohu_path, xml_path = list111111
    in_xml = str(xml_path).replace('.xml','_ys.xml')

    if os.path.exists(in_xml):
        reader = ImageScopeXmlReader(in_xml, keep_arrow_tail=False,use_box_y1x1y2x2=True)
        arrows, arrow_colors = reader.get_arrows()
        boxes, box_colors = reader.get_boxes()
        contours, contour_colors = reader.get_contours()
        ellipses, ellipse_colors = reader.get_ellipses()
        判定轮廓闭合的距离 = 1000  # 1000个像素点，原因是，已经至少发现相隔500-700像素的闭合轮廓
        error_cnts_dict = {}
        all_cnt = []

        for idx, cnt in enumerate(contours):
            c0 = cnt[0, :]
            c1 = cnt[-1, :]

            c0 = c0.astype(np.int64)
            c1 = c1.astype(np.int64)
            d = np.sqrt(np.sum((c0 - c1) ** 2))

            if d > 判定轮廓闭合的距离:
                error_cnts_dict[idx] = cnt
            else:
                all_cnt.append(cnt)

        if len(error_cnts_dict.keys()) > 1:
            while True:
                tmp_key1111 = sorted(list(error_cnts_dict.keys()))[0]
                sorted_cnts_最终 = error_cnts_dict[tmp_key1111]
                error_cnts_dict.pop(tmp_key1111)
                for gkgk in range(len(error_cnts_dict.keys())):
                    c00_最终 = sorted_cnts_最终[0, :].astype(np.int64)
                    c11_最终 = sorted_cnts_最终[-1, :].astype(np.int64)
                    dict1 = {}
                    list_dict_v = []
                    for tmp_key in error_cnts_dict.keys():
                        tmp_cnt = error_cnts_dict[tmp_key]
                        c00 = tmp_cnt[0, :].astype(np.int64)
                        c11 = tmp_cnt[-1, :].astype(np.int64)

                        d1 = np.sqrt(np.sum((c00_最终 - c00) ** 2))  ###头头
                        d2 = np.sqrt(np.sum((c11_最终 - c11) ** 2))  ###尾尾

                        d3 = np.sqrt(np.sum((c11_最终 - c00) ** 2))  ###尾头
                        d4 = np.sqrt(np.sum((c00_最终 - c11) ** 2))  ###头尾

                        # d1 = math.sqrt((c00_最终[0] - c00[0]) ** 2 + (c00_最终[1] - c00[1]) ** 2)###头头
                        # d2 = math.sqrt((c11_最终[0] - c11[0]) ** 2 + (c11_最终[1] - c11[1]) ** 2)###尾尾
                        # d3 = math.sqrt((c11_最终[0] - c00[0]) ** 2 + (c11_最终[1] - c00[1]) ** 2)###尾头
                        # d4 = math.sqrt((c00_最终[0] - c11[0]) ** 2 + (c00_最终[1] - c11[1]) ** 2)###头尾

                        my_list1 = [d1, d2, d3, d4]
                        list_d_type = [1, 2, 3, 4]
                        min_d_tmp = min(my_list1)
                        min_cls = list_d_type[my_list1.index(min_d_tmp)]
                        dict1[min_d_tmp] = [tmp_key, min_cls]
                        list_dict_v.append(min_d_tmp)
                    key_1 = min(list_dict_v)
                    tmp_key, min_cls1 = dict1[key_1]
                    tmp_cnt1 = error_cnts_dict.pop(tmp_key)
                    if min_cls1 == 1:
                        sorted_cnts_最终 = np.concatenate([sorted_cnts_最终[::-1, :], tmp_cnt1], axis=0)
                    elif min_cls1 == 2:
                        sorted_cnts_最终 = np.concatenate([sorted_cnts_最终, tmp_cnt1[::-1, :]], axis=0)
                    elif min_cls1 == 3:
                        sorted_cnts_最终 = np.concatenate([sorted_cnts_最终, tmp_cnt1], axis=0)
                    else:
                        sorted_cnts_最终 = np.concatenate([sorted_cnts_最终[::-1, :], tmp_cnt1[::-1, :]], axis=0)
                    c0 = sorted_cnts_最终[0, :].astype(np.int64)
                    c1 = sorted_cnts_最终[-1, :].astype(np.int64)
                    d = np.sqrt(np.sum((c0 - c1) ** 2))
                    if (d < 判定轮廓闭合的距离) | (len(error_cnts_dict.keys()) == 0):
                        break
                all_cnt.append(sorted_cnts_最终)
                if (len(error_cnts_dict.keys()) == 0):
                    break
        elif len(error_cnts_dict.keys()) == 1:
            all_cnt.append(error_cnts_dict[list(error_cnts_dict.keys())[0]])
        writer = ImageScopeXmlWriter()
        writer.add_arrows(arrows, arrow_colors)
        writer.add_boxes(boxes, box_colors)
        writer.add_contours(all_cnt, [contour_colors[0] for i13 in all_cnt])
        writer.add_ellipses(ellipses, ellipse_colors)
        计数1 =  len(all_cnt)
    else:
        计数1 = 0
        print('请确认执行了一遍复制备份了原始xml文件，命名为_ys.xml')
        writer = ImageScopeXmlWriter()
    try:
        img_slide = tiffslide.TiffSlide(wsi_path)
        wsi_mpp_um = float(img_slide.properties['tiffslide.mpp-x'])
    except:
        img_slide = openslide.OpenSlide(wsi_path)
        wsi_mpp_um = float(img_slide.properties['openslide.mpp-x'])
    name = os.path.basename(wsi_path)
    blur_mask = imageio.v3.imread(mohu_path)

    ds = 2/wsi_mpp_um
    ds_40x = 0.25/wsi_mpp_um

    dist_threshold = 100

    seg_xiuz = imageio.v3.imread(seg_convert_path)
    blur_mask_xiuz = cv2.resize(blur_mask, (seg_xiuz.shape[1],seg_xiuz.shape[0]), interpolation=cv2.INTER_NEAREST)
    # blur_mask_xiuz = np.uint8((blur_mask_xiuz==3)|(blur_mask_xiuz==2))
    # seg_xiuz_0 = cv2.resize(seg_xiuz, (img_slide.level_dimensions[0]), interpolation=cv2.INTER_NEAREST)
    # blur_mask_xiuz_0 = cv2.resize(blur_mask_xiuz, (img_slide.level_dimensions[0]), interpolation=cv2.INTER_NEAREST)

    mask = np.uint8((seg_xiuz==1))
    mask_new = np.zeros((mask.shape), dtype=np.uint8)
    contours_list = []
    contours_list1 = []
    for ii in range(5):
        if ii == 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                x, y = x-1,y-1
                box = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
                cv2.drawContours(mask_new, [box], 0, 1, cv2.FILLED)
        else:
            if ii == 4:
                contours, _ = cv2.findContours(mask_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            else:
                contours, _ = cv2.findContours(mask_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                # 获取外接平行矩形
                x, y, w, h = cv2.boundingRect(cnt)
                w, h = w-1,h-1
                box = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
                if ii == 4:
                    cnt = cnt[:, 0, :]
                    if cnt.shape[0] > 5000:
                        cnt = cnt[::5]
                    contours_list.append(cnt.copy())
                    contours_list1.append(cnt.copy())
                cv2.drawContours(mask_new, [box], 0, 1, cv2.FILLED)
    list11 = []
    list_tmp = []
    for idx1, points1 in enumerate(contours_list):
        list22 = set()
        for idx2, points2 in enumerate(contours_list1):
            dist_min = np.min(distance.cdist(points1, points2))
            if dist_min < dist_threshold:
                list22.add(idx2)
        list11.append(list22)
        list_tmp = list_tmp + list(list22)
    for i1 in list11:
        list_tmp1 = []
        for i2 in i1:
            list_tmp1.append(contours_list[i2])
        box_new = np.concatenate(list_tmp1, axis=0)
        cv2.drawContours(mask_new, [box_new], 0, 1, cv2.FILLED)
    for ii in range(5):
        mask_new_new = np.zeros((mask.shape), dtype=np.uint8)
        contours, _ = cv2.findContours(mask_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # 获取外接平行矩形
            x, y, w, h = cv2.boundingRect(cnt)
            w, h = w - 1, h - 1
            box = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
            cv2.drawContours(mask_new_new, [box], 0, 1, cv2.FILLED)
        mask_new = mask_new_new
    contours, _ = cv2.findContours(mask_new_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes_new_list = []
    box_colors_new_list = []
    mfjfjfk = 0
    for cnt in contours:
        # 获取外接平行矩形
        x, y, w, h = cv2.boundingRect(cnt)
        x_new, y_new, w_new, h_new = round(x*ds), round(y*ds), round(w*ds), round(h*ds)
        w_40x, h_40x = round(w_new * ds_40x), round(h_new * ds_40x)
        # min_size = 448
        # if (w_40x<min_size):
        #     w_new_new = round(min_size * ds_40x)
        #     x_new = x_new - round((w_new_new-w_new)/2)
        #     w_new = w_new_new
        #     w_40x = 448
        # if (h_40x < min_size):
        #     h_new_new = round(min_size * ds_40x)
        #     y_new = y_new - round((h_new_new - h_new) / 2)
        #     h_new = h_new_new
        #     h_40x = 448

        w_new_ys,h_new_ys = w_new,h_new
        # tmp_img_ys = img_slide.read_region((x_new, y_new), 0, (w_new, h_new), as_array=True)[:, :, :3]
        # tmp_img_ys = cv2.resize(tmp_img_ys, (w_40x, h_40x), interpolation=cv2.INTER_NEAREST)

        x1_2um_new, y1_2um_new, x2_2um_new, y2_2um_new = round(x_new/ds),round(y_new/ds),round((x_new+w_new)/ds),round((y_new+h_new)/ds)

        tmp_seg_ee_ys = seg_xiuz[y1_2um_new:y2_2um_new, x1_2um_new:x2_2um_new]

        # tmp_seg_ee_ys_tmp111 =  tmp_seg_ee_ys.copy()

        tmp_blur_ys = blur_mask_xiuz[y1_2um_new:y2_2um_new, x1_2um_new:x2_2um_new]

        # is_error = np.sum(tmp_seg_ee_ys == 1) < 50

        tmp_seg_ee_ys = cv2.resize(tmp_seg_ee_ys, (w_40x, h_40x), interpolation=cv2.INTER_NEAREST)
        tmp_blur_ys = cv2.resize(tmp_blur_ys, (w_40x, h_40x), interpolation=cv2.INTER_NEAREST)

        # tumor_bulk_contours, _  = cv2.findContours(np.uint8(tmp_seg_ee_ys==1),mode=3,method=cv2.CHAIN_APPROX_SIMPLE)
        # tumor_bulk_visual = cv2.drawContours(tmp_img_ys,tumor_bulk_contours,-1,(255,0,0),5)
        # tmp_img_ys =cv2.resize(tumor_bulk_visual,(round(tmp_img_ys.shape[1]/2),round(tmp_img_ys.shape[0]/2)), interpolation=cv2.INTER_NEAREST )
        模糊占比 = np.sum((tmp_seg_ee_ys==1)&(tmp_blur_ys==3))/np.sum(tmp_seg_ee_ys==1)
        # print(模糊占比)
        if (模糊占比)<0.2:
            模糊占比1 = np.sum((tmp_seg_ee_ys == 1) & ((tmp_blur_ys == 3) | (tmp_blur_ys == 2))) / np.sum(tmp_seg_ee_ys == 1)
            if not ((模糊占比1>0.2)&((w_new_ys*h_new_ys)<(1000*1000))):
                box_ys_hw = np.array([[y_new, x_new], [y_new, x_new + w_new_ys], [y_new + h_new_ys, x_new + w_new_ys], [y_new + h_new_ys, x_new]])
                boxes_new_list.append(box_ys_hw)
                box_colors_new_list.append((255, 0, 0))
    计数2 = len(boxes_new_list)

    # os.makedirs(os.path.dirname(str(xml_path).replace('/media/USB_DISK/DCIS_batch2/','/media/USB_DISK/DCIS_batch2_pred/20x_jit8_10x_jit28/tmp_pkl/')), exist_ok=True)
    # file = open(str(xml_path).replace('/media/USB_DISK/DCIS_batch2/','/media/USB_DISK/DCIS_batch2_pred/20x_jit8_10x_jit28/tmp_pkl/').replace('.xml','.pkl'), 'wb')
    # str(xml_path)
    # pickle.dump([计数1,计数2], file)
    # file.close()
    writer.add_boxes(boxes_new_list, box_colors_new_list)
    # xml_path_name = os.path.basename(xml_path)
    # xml_path = '/media/USB_DISK/xml_对比/DCIS_batch2/原来的/'+xml_path_name
    print(xml_path)
    writer.write(xml_path)
    # except:
    #     print('error',list111111[0])

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from pathlib import Path
    import multiprocessing as mul
    def get_npy_file_paths(directory):
        directory_path = Path(directory)
        npy_file_paths = sorted([str(file_path) for file_path in directory_path.rglob('*.svs')])
        return npy_file_paths
    wsi_dir = '/mnt/totem_data2/totem/吴钰-TNBC+新辅助免疫治疗/sunpeng/微乳头穿刺/'
    seg_10_20x_convert_dir = '/mnt/totem_data2/totem/hebingdou_method_seg_2408/吴钰-TNBC+新辅助免疫治疗/微乳头穿刺/result_merge/10_20x_convert/'
    mohumask_dir = '/mnt/totem_data2/totem/hebingdou_method_seg_2408/吴钰-TNBC+新辅助免疫治疗/微乳头穿刺/mohu_mask/'
    xml_output_dir = '/mnt/totem_data2/totem/hebingdou_method_seg_2408/吴钰-TNBC+新辅助免疫治疗/微乳头穿刺/xml_Invasive/'
    os.makedirs(xml_output_dir,exist_ok=True)
    list_wsi_path = get_npy_file_paths(wsi_dir)  #获取输入的文件名称列表
    list_l111  =  []
    for wsi_path in list_wsi_path:
        seg_convert_path = str(wsi_path).replace(wsi_dir, seg_10_20x_convert_dir).replace( '.svs', '.png')
        if not os.path.exists(seg_convert_path):continue
        xml_path = str(wsi_path).replace(wsi_dir, xml_output_dir).replace('.svs', '.xml')
        if os.path.exists(xml_path):continue
        mohu_path = str(wsi_path).replace(wsi_dir, mohumask_dir).replace('.svs', '.png')
        list_l111.append([wsi_path, seg_convert_path, mohu_path, xml_path])

    pool = mul.Pool(2)
    pool.map(fun, list_l111)
    # fun(list_l111[0])

