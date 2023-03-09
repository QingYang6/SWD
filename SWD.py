import os
import glob
import sys
import argparse
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from dask_ml.wrappers import ParallelPostFit

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
        if refbounds is not None:
            inputbounds = [float(i) for i in refbounds.split(',')]
            self.ref = cropfile(self.ref, inputbounds)
        if not is_wgs84(self.ref):
            self.ref = reproject_to_wgs84(self.ref)

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
    bool_exprs = [in_ras[bn,:,:] == cloud_values[i] for i, bn in enumerate(cloud_band)]
    if len(cloud_band)==1:
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

    
def SWD(input_optical,input_cloud,outputfile,geoextent=None): 
    tqdm.monitor_interval = 0  # Default is 1000 ms, but we want to update more frequently
    info_dict = {'index':'ALL', 'green': 1, 'swir': 3,'cloud_band': [0], 'cloud_value': [0]}
    os.makedirs(os.path.dirname(outputfile),exist_ok=True)
    with tqdm(total=9, desc='Data input and ancillary preparing') as pbar:
        data_image = dask.delayed(input_data)(input_optical, geoextent)
        data_cloud = dask.delayed(input_data)(input_cloud, geoextent)
        data_image, data_cloud = dask.compute(data_image, data_cloud)
        pbar.update(1)

        bounds_wgs84 = data_image.bounds()
        print(bounds_wgs84)
        ref_src = data_image.src().meta.copy()
        pbar.update(1)

        rc_wop = dask.delayed(get_ancillary)(bounds_wgs84,ref_src, 'getWO')
        rc_gplcc = dask.delayed(get_ancillary)(bounds_wgs84,ref_src, 'getGPLCC')
        rc_wop, rc_gplcc = dask.compute(rc_wop, rc_gplcc)
        pbar.update(1)

        arr_image = read_todask(data_image.ref)
        arr_cloud = read_todask(data_cloud.ref)
        pbar.update(1)

        WI = get_waterindex(arr_image, info_dict)
        pbar.update(1)

        CM = cloud_mask(arr_cloud,info_dict).squeeze()
        non_valid_mask = da.logical_or(CM,da.where(np.isnan(WI[0,:,:]),True,False)) # ???
        pbar.update(1)

        LCC_raw = read_todask(rc_gplcc).squeeze()
        wop_raw = read_todask(rc_wop).squeeze()
        LCC_raw = da.where(non_valid_mask,0,LCC_raw)
        wop_raw = da.where(non_valid_mask,0,wop_raw)
        pbar.update(1)

        PW = da.where(wop_raw>=50,1,0)
        num_PW = da.count_nonzero(PW==1).compute()
        pbar.update(1)
        PW_buffer = ndimage.binary_dilation(PW, iterations=int(np.sqrt(num_PW//1e3+1)//2), structure = ndimage.generate_binary_structure(2, 2))
        pbar.update(1)

        LCC = da.where(PW_buffer==1,0,LCC_raw)
        pbar.update(1)

    print('get_sample')
    NW_indices = get_random_pixels(LCC.compute(), [10, 20, 30, 40, 80, 90], max(int(np.ceil(num_PW/1e2)),1000)) #for GP LCC
    PW_indices = get_random_pixels(PW.compute(), [1], len(NW_indices))
    Total_Sample_indices = NW_indices + PW_indices
    targets = da.concatenate([da.zeros(len(NW_indices)), da.ones(len(PW_indices))]) 
    In_features = WI.compute()[:,np.array(Total_Sample_indices)[:, 0], np.array(Total_Sample_indices)[:, 1]].T
    In_features = da.from_array(In_features,chunks='auto')
    #Total_Sample_indices[0], Total_Sample_indices[1]
    #ML method, RF, could also use something like grid search.
    clf = ParallelPostFit(LogisticRegressionCV(cv=5), scoring="accuracy")
    clf.fit(In_features, targets)
    clf.score(In_features, targets)
    #rf = RandomForestClassifier(n_estimators=50,max_depth=10,verbose=1)
    #rf.fit(In_features, targets)
    # Generate the pixel coordinates as an array of shape (2, num_pixels)
    #pixel_coords = da.indices(WI.shape[1:]).reshape(2, -1)
    # Get the values of the selected pixels for all bands
    #values = WI.compute()[:, pixel_coords[0], pixel_coords[1]].T
    # find the optimal threshold
    #optimal_threshold = find_optimal_threshold(In_features, targets)
    #print(optimal_threshold.compute())
    #predicted_mask = da.where(WI>optimal_threshold,1,0)
    #import pdb; pdb.set_trace()
    values = da.reshape(WI,(WI.shape[0],-1)).T
    #values = da.from_array(values,chunks='auto')
    # Define a function to apply clf.predict to each block of the dask.array
    #def predict_block(block):
    #    return rf.predict(block)
    predictions = clf.predict(X_large)
    import pdb; pdb.set_trace()
    # Make predictions for the values
    #predictions = values.map_blocks(predict_block, dtype=np.int8)
    #predictions = rf.predict(values)
    # Reshape the predicted target variable to match the shape of the `values_arr` mask
    predicted_mask = da.reshape(predictions, (1,WI.shape[1],WI.shape[2]))
    ref_src.update(dtype=rasterio.uint8, nodata=255, count=1)
    predicted_mask_write = da.where(non_valid_mask, predicted_mask, ref_src['nodata']).astype(ref_src['dtype'])
    write_output(outputfile, predicted_mask_write, ref_src)