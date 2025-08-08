import os,sys
sys.path.append("../")
import config
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_USE
import imageio.v3
import re
import torch
from torchvision.models import DenseNet
import tiffslide
import numpy as np
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from utils import image2mask
from predict_method import get_npy_file_paths
from utils.xml_utils import region_all_binary_image
# from torchvision import transforms
import cv2

COLORS = config.label_colors

def load_model(model_weight_path):
    checkpoint = torch.load(model_weight_path, map_location=lambda storage, loc: storage)
    qc_model = DenseNet(growth_rate=checkpoint["growth_rate"], block_config=checkpoint["block_config"],
                     num_init_features=checkpoint["num_init_features"], bn_size=checkpoint["bn_size"],
                     drop_rate=checkpoint["drop_rate"], num_classes=checkpoint["nclasses"])

    qc_model.load_state_dict(checkpoint["model_dict"])
    qc_model.eval()
    qc_model.cuda()
    return qc_model

def get_pred_label(img_slide, target_mpp):
    """
    get the closest level to target resolution
    """
    pred_level = 0
    jfjfp_pfpf = 99999
    min_img_lever = img_slide.level_count
    for tmp_i in range(min_img_lever):
        mpp_in_level = float(img_slide.properties[tiffslide.PROPERTY_NAME_MPP_X]) * img_slide.level_downsamples[tmp_i]
        ooooo_tmp = abs(target_mpp - mpp_in_level)
        if ooooo_tmp < jfjfp_pfpf:
            jfjfp_pfpf = ooooo_tmp
            pred_level = tmp_i
    return pred_level

class roi_dataset(Dataset):
    def __init__(self,img_list,img_slide,model_pred_szie):
        super().__init__()
        self.images_lst = img_list
        self.img_slide = img_slide
        self.model_pred_szie = model_pred_szie
    def __len__(self):
        return len(self.images_lst)
    def __getitem__(self, idx):
        index_n  = self.images_lst[idx]
        # closest_level_to_target_resolution = index_n[4]
        index_20x_h1,index_20x_w1,_,_,index_20x_size = index_n[0],index_n[1],index_n[2],index_n[3],index_n[5]
        mask_index = torch.as_tensor(index_n[6])
        try:
            image_20x = np.array(self.img_slide.read_region((index_20x_w1,index_20x_h1), index_n[4], (index_20x_size,index_20x_size)))[:,:,:3]
            image_20x  = cv2.resize(image_20x,(self.model_pred_szie,self.model_pred_szie))
        except Exception as e:
            print(f"Found wrong in read tile at {(index_20x_w1,index_20x_h1)}.")
            print(e)
            image_20x = np.zeros((self.model_pred_szie,self.model_pred_szie,3),dtype=np.uint8)

        arr_out_gpu = torch.from_numpy(image_20x.transpose(2, 0, 1) / 255).type('torch.FloatTensor')

        return arr_out_gpu,mask_index


if __name__ == '__main__':
    model_weight_path = config.qc_20x_model_weight_path

    model_pred_szie = 256
    model = load_model(model_weight_path)

    if 'HE_10X.pth' in model_weight_path:
        model_target_mpp = 1
        out_blurred_mask_dir = os.path.join(config.blurred_mask_dir,'10x')
    elif 'HE_20X.pth' in model_weight_path:
        model_target_mpp = 0.5
        out_blurred_mask_dir = os.path.join(config.blurred_mask_dir, '20x')
    else:
        model_target_mpp = None
        print('Please check QC model weight file name !')


    wsi_dir = config.wsi_dir
    mask_dir = config.predict_mask_dir
    os.makedirs(out_blurred_mask_dir,exist_ok=True)
    list_wsi_path = get_npy_file_paths(wsi_dir)  #获取输入的文件名称列表

    for wsi_path in list_wsi_path:
        try:
            fname = os.path.basename(wsi_path)
            # 文件名合规性检查
            if '.' not in fname: continue
            name, ext = re.search(r'^(.*)\.([^\.]*)$', fname).groups()
            img_slide = tiffslide.TiffSlide(wsi_path)
            mask_path = os.path.join(mask_dir, f'{name}.png')
            label_path = os.path.join(config.xml_label_dir, f'{name}.xml')
            # image = TiffReader(svs_file=wsi_path)
            if not os.path.exists(mask_path):
                thumb = img_slide.get_thumbnail(img_slide.level_dimensions[2])
                thumb = np.asarray(thumb)[...,:3]
                mask = image2mask(thumb)
                if os.path.exists(label_path):
                    label_mask = region_all_binary_image(thumb, img_slide.level_downsamples[2], label_path)
                    if np.sum(label_mask > 0) > 1e-6 * mask.shape[0] * mask.shape[1]:
                        mask = mask * label_mask
                    del label_mask
                mask = mask * 255
                cv2.imwrite(mask_path, mask.astype(np.uint8))
            else:
                mask = cv2.imread(mask_path, 0)

            mask_01_array = np.uint8(mask>0)
            del mask
            svs_w,svs_h = img_slide.level_dimensions[0]
            wsi_mpp_um = float(img_slide.properties[tiffslide.PROPERTY_NAME_MPP_X])
            target_w = round(svs_w * wsi_mpp_um / model_target_mpp)
            target_h = round(svs_h * wsi_mpp_um / model_target_mpp)

            closest_level_to_target_resolution = get_pred_label(img_slide, model_target_mpp)
            mask_downsamples = (target_w / mask_01_array.shape[1])
            pred_size_closest_to_target_resolution = round(model_pred_szie * model_target_mpp / (wsi_mpp_um * img_slide.level_downsamples[closest_level_to_target_resolution]))

            # dict_11 = {0:1,1:2,2:3}
            list_all_index = []
            for h_tmp in range(0, target_h, model_pred_szie):
                for w_tmp in range(0, target_w, model_pred_szie):
                    mask_h1  = round(h_tmp / mask_downsamples)
                    mask_h2 = round((h_tmp + model_pred_szie) / mask_downsamples)
                    mask_w1 = round(w_tmp / mask_downsamples)
                    mask_w2 = round((w_tmp + model_pred_szie) / mask_downsamples)
                    mask_result_tmp = mask_01_array[mask_h1:mask_h2,mask_w1:mask_w2]
                    if 1 in mask_result_tmp :
                        h1_tmp_level0 = round(h_tmp * model_target_mpp / wsi_mpp_um)
                        w1_tmp_level0 = round(w_tmp * model_target_mpp / wsi_mpp_um)
                        h2_tmp_level0 = round((h_tmp + model_pred_szie) * model_target_mpp / wsi_mpp_um)
                        w2_tmp_level0 = round((w_tmp + model_pred_szie) * model_target_mpp / wsi_mpp_um)
                        list_all_index.append([h1_tmp_level0,
                                               w1_tmp_level0,
                                               h2_tmp_level0,
                                               w2_tmp_level0,
                                               closest_level_to_target_resolution,
                                               pred_size_closest_to_target_resolution,
                                               (mask_h1 , mask_h2, mask_w1, mask_w2)])
            # mask_01_array1 = np.zeros(mask_01_array.shape,dtype=np.uint8)

            dataset1 = roi_dataset(img_list=list_all_index,img_slide=img_slide,model_pred_szie=model_pred_szie)
            database_loader = DataLoader(dataset1, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False, pin_memory=True)
            with torch.no_grad():
                for batch_20X,mask_index in tqdm(database_loader):
                    output_batch = model(batch_20X.cuda())
                    # output_batch =  np.uint8(((torch.softmax(output_batch, dim=-1)[:,2]).cpu().numpy())> 0.8)
                    output_batch = output_batch.argmax(dim=1)
                    output_batch = output_batch.cpu().numpy()
                    for bi in range(output_batch.shape[0]):
                        tile_classes = int(output_batch[bi])+1
                        mask_h1, mask_h2, mask_w1, mask_w2 = mask_index[bi,]
                        mask_01_array[mask_h1:mask_h2, mask_w1:mask_w2] = tile_classes
            out_path = os.path.join(out_blurred_mask_dir,f'{name}.png')
            # str(wsi_path).replace(wsi_dir, out_blurred_mask_dir).replace('.svs', '.png')
            ###mask_01_array这里就不是01mask了
            imageio.v3.imwrite(out_path,mask_01_array)
        except Exception as e:
            print(f"Found wrong in {os.path.basename(wsi_path)}:")
            print(e)
