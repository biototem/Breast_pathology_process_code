# Breast_pathology_process_code

This repository provides inference and postprocessing scripts for the article *From Exhaustion to Efficiency: Human-AI Collaboration Enhances Detection of Invasive Carcinoma in Breast Cancer with Extensive Intraductal Component (EIC)*.
(currently under review).

## System requirements
Linux (Ubuntu2004 is recommended)

**Python3.8** or **Python3.9**

## Python packages Dependencies
- torch (≥1.12.0+cu)
- torchvision (≥0.13.0+cu) 
- [ASAP](https://github.com/computationalpathologygroup/ASAP) (==2.1)
- timm (==0.4.12)
- openslide
- tiffslide
- opencv
- shapely (2.0.6)

other library requirements, see  **./requirements.txt**

## Installation guide
Base [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or [Conda](https://anaconda.org/anaconda/conda),
and create a conda environment:
```bash
conda create -n your_env_name python=3.8
conda activate your_env_name
# Install torch
python -m pip install "torch==1.12.0" torchvision --index-url https://download.pytorch.org/whl/cu113
# git clone 
pip install -r /your_project_path/requirements.txt
```
### ASAP
Download ASAP deb form [ASAP 2.1](https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1/ASAP-2.1-py38-Ubuntu2004.deb) or other version match your system and python environment:
https://github.com/computationalpathologygroup/ASAP/releases
  (2.1 is recommended),and install it,for example:
```bash
dpkg -i /your_deb_path/ASAP-2.1-py38-Ubuntu2004.deb
# Sometimes need apt --fix-broken install
echo "/opt/ASAP/bin" >> /your_python_env_path/lib/python3.8/site-packages/asap.pth
```

## Demo
### in_put
Supported whole slide image formats：svs,ndpi,tif/tiff and other formats that can be read by openslide-python/tiffslide.

You can modify `file_suffix_list` to support in prediction and procession at **./config.py** line 14:
```python
file_suffix_list = ['svs','ndpi','tif']
```

The WSI files to be predicted need to be put in the same directory, and modify the real path at at **./config.py** line 12:
```python
wsi_dir = 'The path of WSI files to be predicted'
```
### out_put
You can modify `predict_result_out_root_dir` as the output root directory **./config.py** line 18:
```python
predict_result_out_root_dir = "The path of your output root dir"
```
Segmentation model output file format is **tif**(generic multi-resolution tiled), which can be read by tiffslide or ASAP in python. 
Or you can overlay the tif as visualization of segmentation result when you use ASAP to view WSI.

#### original 10X&20X Segmentation model output path

./predict_result_out_root_dir/seg_10x

./predict_result_out_root_dir/seg_20x

#### corrected Segmentation model output path
**./predict_result_out_root_dir/result_merge/10_20x_convert**
(result saved as normal image png format with size of mpp=1.0)

**./predict_result_out_root_dir/result_merge/10_20x_convert_final**
(result saved as generic multi-resolution tiled tif format with size of mpp=0.5)

#### QC classification model output path
**./predict_result_out_root_dir/blurred_mask/10x**

**./predict_result_out_root_dir/blurred_mask/20x**

#### Invasive blur box xml output path
**./predict_result_out_root_dir/invasive_blur_box_xml_output_dir**


## Model weights and sample data for testing
Segmentation model weights and sample data for testing are not available to the public yet. 
Please feel free to contact us should you require any additional information of model weights and sample data.

We use [HistoBlur](https://github.com/choosehappy/HistoBlur/tree/main) to detect blurry regions in Whole Slide Images,
and reorganized the code in our project.

## Instructions for use
1. Modify `wsi_dir`,`xml_label_dir`(which is not required. If there is no label file, fill in an empty string or the same path as wsi_dir),
`predict_result_out_root_dir`,`predict_mask_dir`,`output_10X_model_seg_dir`,`output_20X_model_seg_dir`,`seg_10X_model_weight_path`,`seg_20X_model_weight_path`,
`qc_10x_model_weight_path`,`qc_20x_model_weight_path`,`invasive_blur_box_xml_output_dir`
at **./config.py**


2. Run scripts for segmentation.
There are two model weight for segmentation：
*20x.pth(predict with mpp=1.0) and *10x.pth(predict with mpp=0.5).
Modify *20x.pth model weight file to `seg_20X_model_weight_path`,
*10x.pth model weight file to `seg_10X_model_weight_path` at **./config.py** line 29,27
```bash
python single_predict_compare_jit_10x.py
# 10x model prediction
python single_predict_compare_jit_20x.py
# 20x model prediction
```

3. Run scripts for post processing
```bash
cd post_process
python 10x_20x_seg_result_correct.py
python corrected_result_to_20x_tif.py
```
4. Run QC scripts for predicted WSI

There are two model weights for QC classification:
*20X.pth(predict with mpp=1.0) and *10X.pth(predict with mpp=0.5).
Modify *20X.pth model weight file to `qc_20x_model_weight_path`,
*10X.pth model weight file to `qc_10x_model_weight_path` at **./config.py** line 33,32
and select one of them for inferencing. *20X.pth model is recommended, requires more time can get better result.
Modify `model_weight_path` at **./post_process/wsi_blur_cls_predict.py** line 74:
```python
import config
model_weight_path = config.qc_20x_model_weight_path # or config.qc_10x_model_weight_path
```
and run script to inference.
```bash
# cd post_process
python wsi_blur_cls_predict.py
```

4. Run scripts for generating invasive boxes of predicted WSI
```bash
# cd post_process
python invasive_boxes_ys_generate.py
```

## Training scripts
Segmentation model training method and scripts can see at below repository:

[Segmentation and Detection training code](https://github.com/biototem/TIGER_challenge_2022/tree/master/train_script)



# LICENSE: CC Attribution-NonCommercial 4.0 International
Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg


