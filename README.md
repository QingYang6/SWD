### Updated: Adding new fix relative threshold based on [(mNDWI < NDVI or mNDWI < EVI) and (EVI < 0.1)], 08/03/23.
To use the new fix relative threshold, please refer to the following template:
```
python -u SWD.py /Users/qyang/Downloads/RR_05042023_psscene_analytic_sr_udm_all/composite.tif \
/Users/qyang/Downloads/RR_05042023_psscene_analytic_sr_udm_all/composite_udm2.tif ./OUTPUT/test_RWC_0504_all_RR.tif "{'index':'ALL','cloud_band':[0],'cloud_value':[0]}" RWC
```

RWC mode will export an additional mNDWI file with the extent of '_mndwi.tif' for reference. For Planet image, the mNDWI is calculated using the NIR and Green bands.

Without the RWC tag in the end will use the SWD by default.


Refernce:
Zou, Zhenhua, Xiangming Xiao, Jinwei Dong, Yuanwei Qin, Russell B. Doughty, Michael A. Menarguez, Geli Zhang, and Jie Wang. 2018. “Divergent Trends of Open-Surface Water Body Area in the Contiguous United States from 1984 to 2016.” Proceedings of the National Academy of Sciences of the United States of America 115 (15): 3810–15.


### Updated: Pulling land cover data from ESA WorldCover V100 2020, 05/04/23.
### Updated: Improved ancillary data fecthing, 03/11/23.

# SWD: Self-supervised Waterbody Detection
Generate water mask from sattliete images using self-supervised classification.

## 1 Description
1. The code automatically prepares ancillary data, including water occurrence and land cover.
2. The kernel classification model is LogisticRegression.
3. The training data is automatically sampled from the ancillary data.

## 2 Requirements
```
conda create --name swd
conda activate swd
conda install dask-ml gdal rasterio shapely matplotlib scipy pandas tqdm jupyter -c conda-forge
```

## 3 Usage
Clone or download this repository then:
### Option 1: run in terminal
```
python -u PATH_TO_SWD_FOLDER/SWD.py input_image_path cloud_mask_path output_file_path parameters
```
Args:
1. input_image_path: input image file path, could be any satellite image with CRS defined.
2. cloud_mask_path: cloud maks or unuse data mask, contain invlid pixel mask such as cloud, cloud shadow, snow, missing value, etc. Should be in the same geoextent as the input_image_path. Set to None if such mask is not applicable.
3. output_file_path: output flood depth file path.
4. parameters: a dictionary like string that contains key parameters including band number, band value and water index type. 
- An example for planet data using all bands: "{'index':'ALL','cloud_band':[0],'cloud_value':[0]}" (include quotes).
 - 'ALL' for index means using all input bands of image as features; 
 - [0] for 'cloud_band' means using the first band of cloud_mask_path as the indicator of cloud or other invalid pixels;
 - [0] for 'cloud_value' means the pixel value of 0 in the specify band is the cloud pixel (invalid pixel).
 - If there are multiple bands represent invalid area, could use something like 'cloud_band':[1,2,3...],'cloud_value':[[1,2,3,4],2,5...].

### Option 2: run in python
```
import sys
sys.path.insert(0, 'PATH_TO_SWD_FOLDER')
from SWD import *

SWD(input_image_path,cloud_mask_path,output_file_path,parameters)
```
Args:
1. input_image_path,cloud_mask_path,output_file_path are the same as Option 1.
2. parameters: similar as Option 1. Can directly input with a dictionary variable {'index':'ALL','cloud_band':[0],'cloud_value':[0]}, no need to be a string.


