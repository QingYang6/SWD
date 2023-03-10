"""
Generate water mask based on sel-supervised classification.
    1.The code automatically prepares ancillary data, including water occurrence and land cover.
    2.The kernel classification model is LogisticRegression.
    3.The sample for training the model is automatically pulling from ancillary data.
Created by Qing Yang, email: yang2473@uwm.edu
Usage:
python -u ./SWD.py input_image_path cloud_mask_path output_file_path parameters
Args:
    input_image_path: input image file path, could be any satellite image with CRS defined.
    cloud_mask_path: cloud maks or unuse data mask, contain invlid pixel mask such as cloud, cloud shadow, snow, missing value, etc. Should be in the same geoextent as the input_image_path. Set to None if such mask is not applicable.
    output_file_path: output flood depth file path.
    parameters: a dictionary variable that contains key parameters including band number, band value and water index type. 
    An example for planet data using all bands: "{'index':'ALL','cloud_band':[0],'cloud_value':[0]}". 
    'ALL' for index means using all input bands of image as features; 
    [0] for 'cloud_band' means using the first band of cloud_mask_path as the indicator of cloud or other invalid pixels;
    [0] for 'cloud_value' means the pixel value of 0 in the specify band is the cloud pixel (invalid pixel).
    If there are multiple band represent invalid area, could use something like 'cloud_band':[1,2,3...],'cloud_value':[[1,2,3,4],2,5...]
"""
import os
import glob
import sys
import ast
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds, calculate_default_transform
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
from utils import *
from ancillarydata import *
import dask
import dask.array as da
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegressionCV
from dask_ml.wrappers import ParallelPostFit
#from sklearn.svm import SVC

class input_data:
    def __init__(self, masktif:str, refbounds=None):
        if type(masktif)==list:
            print(f"Input list of files")
            for file in masktif:
                if not os.path.exists(file):
                    raise ValueError(f"{file} does not exist!")
                print(file)
            self.ref = mergelist(masktif)
        elif type(masktif)==str:
            if not os.path.exists(masktif):
                raise ValueError(f"{masktif} does not exist!")
            if os.path.isfile(masktif):
                print(f"single input file {masktif}")
                self.ref = masktif
            else:
                dirlist = glob.glob(masktif+'/*.tif')
                if len(dirlist)==0:
                    raise ValueError(f"{masktif} contains no file!")
                elif len(dirlist)==1:
                    print(f"single input file {dirlist[0]}")
                    self.ref = dirlist[0]
                else:
                    print(f"Multiple input files: ")
                    for file in dirlist:
                        print(file)
                    self.ref = mergelist(dirlist)
        else:
            raise ValueError(f"Input {masktif} not valid!")
        #if not is_wgs84(self.ref):
        #    self.ref = reproject_to_wgs84(self.ref)
        if refbounds is not None:
            inputbounds = [float(i) for i in refbounds.split(',')]
            self.ref = cropfile_oriproject(self.ref, inputbounds)

    def read(self):
        with rasterio.open(self.ref) as src:
            data = src.read(1)
        return data

    def read_multiband(self):
        with rasterio.open(self.ref) as src:
            data = src.read()
        return data

    def bounds(self):
        with rasterio.open(self.ref) as src:
            bounds = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return bounds

    def src(self):
        with rasterio.open(self.ref) as src:
            refsrc = src
        return refsrc

def get_ancillary(bounds_wgs84, ref_src ,func_name):
    # Call the function specified by func_name with bounds_wgs84 as input
    data_list = globals()[func_name](bounds_wgs84)
    # Merge the data in the list
    merged_data = mergelist(data_list)
    # Reproject and clip the merged data
    rc_data = reproject_clip(merged_data,ref_src)
    # Return the processed data
    return rc_data

def print_raster_info(file_path):
    """
    Prints all the information of a raster file
    """
    with rasterio.open(file_path) as src:
        print(src.profile)
        print(f"Band Count: {src.count}")
        print(f"Shape: {src.shape}")
        print(f"Data Type: {src.dtypes[0]}")
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")
        print(f"bounds: {src.bounds}")

def read_todask(filepath):
    with rasterio.open(filepath) as src:
        data = da.from_array(src.read(), chunks='auto')
    return data

def cloud_mask(in_ras,info_dict):
    cloud_band = info_dict['cloud_band']
    cloud_values = info_dict['cloud_value']
    bool_exprs = []
    for i, bn in enumerate(cloud_band):
        if isinstance(cloud_values[i], list):
            bool_exprs.append(np.in1d(in_ras[bn,:,:], cloud_values[i]).reshape(in_ras.shape[1:]))
        else:
            bool_exprs.append(in_ras[bn,:,:] == cloud_values[i])
    if len(bool_exprs) == 1:
        cloud_mask = bool_exprs[0]
    else:
        cloud_mask = da.logical_or(*bool_exprs)
    return cloud_mask

def MNDWI(in_ras,info_dict):
    info_dict = {'green': 1, 'swir': 3}
    '''for Planet'''
    MNDWI = (in_ras[info_dict['green'],:,:]-in_ras[info_dict['swir'],:,:]) / \
    (in_ras[info_dict['green'],:,:]+in_ras[info_dict['swir'],:,:])
    return MNDWI

def ALL(in_ras,info_dict):
    return in_ras

def get_waterindex(in_bands, info_dict):
    water_index = globals()[info_dict['index']](in_bands,info_dict)
    return water_index

def get_random_pixels(arr, labels, num_pixels=1000):
    indices = []
    for label in labels:
        # Find all indices where the array has the given label
        label_indices = np.where(arr == label)
        # If there are fewer than num_pixels pixels with this label, just use all of them
        if len(label_indices[0]) < num_pixels:
            indices.extend(list(zip(label_indices[0], label_indices[1])))
        else:
            # Otherwise, randomly select num_pixels indices from the list of all indices with this label
            random_indices = np.random.choice(len(label_indices[0]), size=num_pixels, replace=False)
            for idx in random_indices:
                indices.append((label_indices[0][idx], label_indices[1][idx]))
    return indices

@dask.delayed
def compute_f1_score(feature, y_true, threshold):
    y_pred_threshold = (feature > threshold).astype(int)
    return f1_score(y_true, y_pred_threshold)

def find_optimal_threshold(feature, y_true):
    thresholds = da.linspace(np.min(feature), np.max(feature), 1000)  # try 1000 threshold values between min and max feature values
    f1_scores = []
    for threshold in tqdm(thresholds,desc='Search optimal index'):
        f1_score_delayed = compute_f1_score(feature, y_true, threshold)
        f1_scores.append(f1_score_delayed)
    f1_scores_computed = dask.compute(*f1_scores)
    print(np.max(f1_scores_computed))
    optimal_threshold = thresholds[np.argmax(f1_scores_computed)]
    return optimal_threshold

def write_output(out_file, result, profile):
    with rasterio.open(out_file, 'w', **profile) as dst:
        dst.write(result)
    print(f"Out file: {out_file}")

    
def SWD(input_optical,input_cloud,outputfile,info_dict,geoextent=None): 
    if isinstance(info_dict, str):
        info_dict = ast.literal_eval(info_dict)
    print(info_dict)
    tqdm.monitor_interval = 0  # Default is 1000 ms, but we want to update more frequently
    #info_dict = {'index':'ALL','green': 1,'swir': 3,'cloud_band':[0],'cloud_value':[0]}
    os.makedirs(os.path.dirname(outputfile),exist_ok=True)
    if input_cloud=='None':
        data_image = input_data(input_optical, geoextent)
    else:
        data_image = dask.delayed(input_data)(input_optical, geoextent)
        data_cloud = dask.delayed(input_data)(input_cloud, geoextent)
        data_image, data_cloud = dask.compute(data_image, data_cloud)
    #get bounds of input image, in wgs84
    bounds_wgs84 = data_image.bounds()
    print(f'Working extent: {bounds_wgs84}')
    ref_src = data_image.src().meta.copy()
    #prepare ancillary data
    print(f'Preparing ancillary data')
    rc_wop = dask.delayed(get_ancillary)(bounds_wgs84,ref_src, 'getWO')
    rc_gplcc = dask.delayed(get_ancillary)(bounds_wgs84,ref_src, 'getGPLCC')
    rc_wop, rc_gplcc = dask.compute(rc_wop, rc_gplcc)
    #prepare sample
    with tqdm(total=8, desc='Sample generating...') as pbar:
        arr_image = read_todask(data_image.ref)
        if input_cloud != 'None':
            arr_cloud = read_todask(data_cloud.ref)
        pbar.update(1)

        WI = get_waterindex(arr_image, info_dict)
        pbar.update(1)

        if input_cloud != 'None':
            CM = cloud_mask(arr_cloud,info_dict).squeeze()
        else:
            CM = da.where(WI[0,:,:]==0,False,False)
        non_valid_mask = da.logical_or(CM,da.where(arr_image[0,:,:]==ref_src['nodata'],True,False))
        pbar.update(1)

        LCC_raw = read_todask(rc_gplcc).squeeze()
        wop_raw = read_todask(rc_wop).squeeze()
        LCC_raw = da.where(non_valid_mask,0,LCC_raw)
        wop_raw = da.where(non_valid_mask,0,wop_raw)
        pbar.update(1)

        PW = da.where(wop_raw>=50,1,0)#some how import, but the optimal is mysterious.
        num_PW = da.count_nonzero(PW==1).compute()
        if num_PW<=1:
            raise ValueError(f"Not enough persistent water in the image")
        pbar.update(1)
        PW_buffer = ndimage.binary_dilation(PW, iterations=int(np.sqrt(num_PW//1e3+1)//2), structure = ndimage.generate_binary_structure(2, 2))
        LCC = da.where(PW_buffer==1,0,LCC_raw)
        pbar.update(1)

        NW_indices = get_random_pixels(LCC.compute(), [10, 20, 30, 40, 80, 90], min(int(np.ceil(num_PW/25)),2e5)) #for GP LCC
        PW_indices = get_random_pixels(PW.compute(), [1], len(NW_indices))
        Total_Sample_indices = NW_indices + PW_indices
        pbar.update(1)
        targets = da.concatenate([da.zeros(len(NW_indices)), da.ones(len(PW_indices))]) 
        In_features = WI.compute()[:,np.array(Total_Sample_indices)[:, 0], np.array(Total_Sample_indices)[:, 1]].T
        In_features = da.from_array(In_features,chunks='auto')
        pbar.update(1)
    print(f'Total sample size: {len(targets)}')
    #ML method, RF, could also use something like grid search.
    clf = ParallelPostFit(LogisticRegressionCV(cv=5), scoring="accuracy")
    #clf = ParallelPostFit(SVC(kernel='rbf', C=1.0, random_state=0), scoring="accuracy")
    clf.fit(In_features, targets)
    print(f'Fitting score: {clf.score(In_features, targets)}')
    #import pdb; pdb.set_trace()
    values = da.reshape(WI,(WI.shape[0],-1)).T
    #prediction
    print(f'Water body detecting...')
    predictions = clf.predict(values)
    # Reshape the predicted target variable to match the shape of the `values_arr` mask
    predicted_mask = da.reshape(predictions, (1,WI.shape[1],WI.shape[2]))
    ref_src.update(dtype=rasterio.uint8, nodata=255, count=1)
    predicted_mask_write = da.where(non_valid_mask, ref_src['nodata'], predicted_mask).astype(ref_src['dtype'])
    write_output(outputfile, predicted_mask_write, ref_src)

if __name__ == "__main__":
    SWD(*sys.argv[1:])