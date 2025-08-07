# Breast_pathology_process_code

This repository provides training,predicting and post processing scripts for the article *From Exhaustion to Efficiency: Human-AI Collaboration Enhances Detection of Invasive Carcinoma in Breast Cancer with Extensive Intraductal Component (EIC)*.
(currently under review).

## System requirements
Linux (Ubuntu is recommended)

**Python3.8** or **Python3.9**

## Python packages Dependencies
- torch (≥1.12.0+cu)
- torchvision (≥0.13.0+cu) 
- [ASAP](https://github.com/computationalpathologygroup/ASAP)  (==2.1)
- timm (==0.4.12)
- openslide
- tiffslide
- opencv
- shapely (2.0.6)

other library requirements, see  **./requirements.txt**

## Demo
### in_put
Supported whole slide image formats：svs,ndpi,tif/tiff and other formats that can be read by openslide-python/tiffslide.

You can modify `file_suffix_list` at **./config.py** line 14:
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
(result saved as png format with size of mpp=1.0)
**./predict_result_out_root_dir/result_merge/10_20x_convert_final**
(result saved as generic multi-resolution tiled tif format with size of mpp=0.5)

## Model weights
Model weights can be downloaded at this  [URL](https://drive.google.com/drive/folders/167IorZBsDn5Lcds_YAZUB9rAtwcGE9Qm?usp=drive_link)

## Instructions for use
1. modify `wsi_dir`,`xml_label_dir`(which is not required. If there is no label file, fill in an empty string or the same path as wsi_dir),
`predict_result_out_root_dir`,`predict_mask_dir`,`output_10X_model_seg_dir`,`output_20X_model_seg_dir`,`seg_10X_model_weight_path`,`seg_20X_model_weight_path`
at **./config.py**


2. run scripts for segmentation predicting
```bash
python single_predict_compare_jit_10x.py
python single_predict_compare_jit_20x.py
```


3. run scripts for post processing
```bash
cd post_process
python 10x_20x_seg_result_correct.py
python corrected_result_to_20x_tif.py
```

# LICENSE: CC Attribution-NonCommercial 4.0 International
Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg


