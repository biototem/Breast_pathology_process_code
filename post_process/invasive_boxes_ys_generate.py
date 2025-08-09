import os,sys
sys.path.append("../")
# import pickle
# import math
# import shutil
import config
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_USE
import imageio.v3
import numpy as np
import cv2
import openslide
import tiffslide
from PIL import Image
from scipy.spatial import distance
Image.MAX_IMAGE_PIXELS = None
from torchvision.transforms import transforms #autoaugment
from utils.imagescope_xml_utils import ImageScopeXmlReader,ImageScopeXmlWriter

train_transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

COLORS = config.label_colors

def fun(list111111):
    # try:
    wsi_path, seg_convert_path,blur_mask_path, xml_path = list111111
    in_xml = str(xml_path).replace('.xml','_ys.xml')

    if os.path.exists(in_xml):
        reader = ImageScopeXmlReader(in_xml, keep_arrow_tail=False,use_box_y1x1y2x2=True)
        arrows, arrow_colors = reader.get_arrows()
        boxes, box_colors = reader.get_boxes()
        contours, contour_colors = reader.get_contours()
        ellipses, ellipse_colors = reader.get_ellipses()
        closed_contour_distance = 1000  # 1000个像素点，原因是，已经至少发现相隔500-700像素的闭合轮廓
        error_cnts_dict = {}
        all_cnt = []

        for idx, cnt in enumerate(contours):
            c0 = cnt[0, :]
            c1 = cnt[-1, :]

            c0 = c0.astype(np.int64)
            c1 = c1.astype(np.int64)
            d = np.sqrt(np.sum((c0 - c1) ** 2))

            if d > closed_contour_distance:
                error_cnts_dict[idx] = cnt
            else:
                all_cnt.append(cnt)

        if len(error_cnts_dict.keys()) > 1:
            while True:
                tmp_key1111 = sorted(list(error_cnts_dict.keys()))[0]
                sorted_cnts_final = error_cnts_dict[tmp_key1111]
                error_cnts_dict.pop(tmp_key1111)
                for gkgk in range(len(error_cnts_dict.keys())):
                    c00_final = sorted_cnts_final[0, :].astype(np.int64)
                    c11_final = sorted_cnts_final[-1, :].astype(np.int64)
                    dict1 = {}
                    list_dict_v = []
                    for tmp_key in error_cnts_dict.keys():
                        tmp_cnt = error_cnts_dict[tmp_key]
                        c00 = tmp_cnt[0, :].astype(np.int64)
                        c11 = tmp_cnt[-1, :].astype(np.int64)

                        d1 = np.sqrt(np.sum((c00_final - c00) ** 2))  ###头头
                        d2 = np.sqrt(np.sum((c11_final - c11) ** 2))  ###尾尾

                        d3 = np.sqrt(np.sum((c11_final - c00) ** 2))  ###尾头
                        d4 = np.sqrt(np.sum((c00_final - c11) ** 2))  ###头尾

                        # d1 = math.sqrt((c00_final[0] - c00[0]) ** 2 + (c00_final[1] - c00[1]) ** 2)###头头
                        # d2 = math.sqrt((c11_final[0] - c11[0]) ** 2 + (c11_final[1] - c11[1]) ** 2)###尾尾
                        # d3 = math.sqrt((c11_final[0] - c00[0]) ** 2 + (c11_final[1] - c00[1]) ** 2)###尾头
                        # d4 = math.sqrt((c00_final[0] - c11[0]) ** 2 + (c00_final[1] - c11[1]) ** 2)###头尾

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
                        sorted_cnts_final = np.concatenate([sorted_cnts_final[::-1, :], tmp_cnt1], axis=0)
                    elif min_cls1 == 2:
                        sorted_cnts_final = np.concatenate([sorted_cnts_final, tmp_cnt1[::-1, :]], axis=0)
                    elif min_cls1 == 3:
                        sorted_cnts_final = np.concatenate([sorted_cnts_final, tmp_cnt1], axis=0)
                    else:
                        sorted_cnts_final = np.concatenate([sorted_cnts_final[::-1, :], tmp_cnt1[::-1, :]], axis=0)
                    c0 = sorted_cnts_final[0, :].astype(np.int64)
                    c1 = sorted_cnts_final[-1, :].astype(np.int64)
                    d = np.sqrt(np.sum((c0 - c1) ** 2))
                    if (d < closed_contour_distance) | (len(error_cnts_dict.keys()) == 0):
                        break
                all_cnt.append(sorted_cnts_final)
                if (len(error_cnts_dict.keys()) == 0):
                    break
        elif len(error_cnts_dict.keys()) == 1:
            all_cnt.append(error_cnts_dict[list(error_cnts_dict.keys())[0]])
        writer = ImageScopeXmlWriter()
        writer.add_arrows(arrows, arrow_colors)
        writer.add_boxes(boxes, box_colors)
        writer.add_contours(all_cnt, [contour_colors[0] for i13 in all_cnt])
        writer.add_ellipses(ellipses, ellipse_colors)
        # 计数1 =  len(all_cnt)
    else:
        # 计数1 = 0
        print('请确认执行了一遍复制备份了原始xml文件，命名为_ys.xml')
        writer = ImageScopeXmlWriter()
    try:
        img_slide = tiffslide.TiffSlide(wsi_path)
        wsi_mpp_um = float(img_slide.properties[tiffslide.PROPERTY_NAME_MPP_X])
    except:
        img_slide = openslide.OpenSlide(wsi_path)
        wsi_mpp_um = float(img_slide.properties[openslide.PROPERTY_NAME_MPP_X])
    # name = os.path.basename(wsi_path)
    blur_mask = imageio.v3.imread(blur_mask_path)

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
        blur_proportion = np.sum((tmp_seg_ee_ys==1)&(tmp_blur_ys==3))/np.sum(tmp_seg_ee_ys==1)
        # print(blur_proportion)
        if (blur_proportion)<0.2:
            blur_proportion1 = np.sum((tmp_seg_ee_ys == 1) & ((tmp_blur_ys == 3) | (tmp_blur_ys == 2))) / np.sum(tmp_seg_ee_ys == 1)
            if not ((blur_proportion1>0.2)&((w_new_ys*h_new_ys)<(1000*1000))):
                box_ys_hw = np.array([[y_new, x_new], [y_new, x_new + w_new_ys], [y_new + h_new_ys, x_new + w_new_ys], [y_new + h_new_ys, x_new]])
                boxes_new_list.append(box_ys_hw)
                box_colors_new_list.append((255, 0, 0))
    # 计数2 = len(boxes_new_list)

    writer.add_boxes(boxes_new_list, box_colors_new_list)

    print(xml_path)
    writer.write(xml_path)
    # except:
    #     print('error',list111111[0])

if __name__ == '__main__':
    import re
    import multiprocessing as mul
    from predict_method import get_npy_file_paths

    wsi_dir = config.wsi_dir
    seg_10_20x_convert_dir = os.path.join(config.predict_result_out_root_dir,'result_merge','10_20x_convert')
    xml_output_dir = config.invasive_blur_box_xml_output_dir
    os.makedirs(xml_output_dir,exist_ok=True)
    list_wsi_path = get_npy_file_paths(wsi_dir)  #获取输入的文件名称列表
    list_l111  =  []
    for wsi_path in list_wsi_path:
        fname = os.path.basename(wsi_path)
        # 文件名合规性检查
        if '.' not in fname: continue
        name, ext = re.search(r'^(.*)\.([^\.]*)$', fname).groups()
        seg_convert_path = os.path.join(seg_10_20x_convert_dir,f"{name}.png")
        # str(wsi_path).replace(wsi_dir, seg_10_20x_convert_dir).replace( '.svs', '.png')
        if not os.path.exists(seg_convert_path):continue
        xml_save_path = os.path.join(xml_output_dir,f"{name}.xml")
        if os.path.exists(xml_save_path):continue
        blur_mask_dir = os.path.join(config.blurred_mask_dir, '20x')
        blur_mask_path = os.path.join(blur_mask_dir,f"{name}.png")
        if not os.path.exists(blur_mask_path):
            blur_mask_dir = os.path.join(config.blurred_mask_dir, '10x')
            blur_mask_path = os.path.join(blur_mask_dir, f"{name}.png")
        # str(wsi_path).replace(wsi_dir, blur_mask_dir).replace('.svs', '.png')
        if not os.path.exists(blur_mask_path): continue
        list_l111.append([wsi_path, seg_convert_path, blur_mask_path, xml_save_path])

    pool = mul.Pool(2)
    pool.map(fun, list_l111)
    # fun(list_l111[0])

