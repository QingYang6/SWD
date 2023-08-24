"""
Generate water mask based on sel-supervised classification.
    1.The code automatically prepares ancillary data, including water occurrence and land cover.
    2.The kernel classification model is gaussian mixture model.
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
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from dask_ml.wrappers import ParallelPostFit
import argparse
import json
from scipy.stats import wasserstein_distance
from dask import delayed, compute
from itertools import product
import numpy as np
from AdaI import *
#from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

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
    data = [dask.delayed(reproject_clip_readsrc)(url,ref_src) for url in data_list]
    results = dask.compute(*data)
    rc_data = da.nanmax(da.stack(results, axis=0),axis=0).squeeze()
    # Merge the data in the list
    #merged_data = mergelist(data_list)
    # Reproject and clip the merged data
    #rc_data = reproject_clip(merged_data,ref_src)
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

def MNDWI(in_ras,info_dict=None):
    info_dict = {'green': 1, 'swir': 3}
    '''for Planet'''
    MNDWI = (in_ras[info_dict['green'],:,:]-in_ras[info_dict['swir'],:,:]) / \
    (in_ras[info_dict['green'],:,:]+in_ras[info_dict['swir'],:,:])
    return MNDWI

def NDVI(in_ras,info_dict=None):
    info_dict = {'NIR': 3, 'Red': 0}
    '''for Planet'''
    NDVI = (in_ras[info_dict['NIR'],:,:]-in_ras[info_dict['Red'],:,:]) / \
    (in_ras[info_dict['NIR'],:,:]+in_ras[info_dict['Red'],:,:])
    return NDVI

def EVI(in_ras,info_dict=None):
    info_dict = {'NIR': 3, 'Red': 0, 'Blue': 2}
    '''for Planet'''
    EVI = 2.5 * (in_ras[info_dict['NIR'],:,:]-in_ras[info_dict['Red'],:,:]) / \
    (in_ras[info_dict['NIR'],:,:] + 6 * in_ras[info_dict['Red'],:,:] - 7.5 * in_ras[info_dict['Blue'],:,:] + 1)
    return EVI

def ALL(in_ras,info_dict):
    in_ras = in_ras.astype('float32')
    '''for Planet, all bands and two indices'''
    info_dict = {'red': 0, 'green': 1, 'blue': 2, 'nir': 3}
    NDWI = - (in_ras[info_dict['green'],:,:]-in_ras[info_dict['nir'],:,:]) / \
    (in_ras[info_dict['green'],:,:]+in_ras[info_dict['nir'],:,:])
    NDVI = (in_ras[info_dict['nir'],:,:]-in_ras[info_dict['red'],:,:]) / \
    (in_ras[info_dict['nir'],:,:]+in_ras[info_dict['red'],:,:])
    in_ras = da.concatenate((in_ras, NDWI[np.newaxis, :, :], NDVI[np.newaxis, :, :]), axis=0)
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

def write_output(out_file, result, profile):
    with rasterio.open(out_file, 'w', **profile, compress='deflate') as dst:
        dst.write(result)
    print(f"Out file: {out_file}")

@tim
def get_PDF_Th(Th, gmm):
    n_components, n_features = gmm.means_.shape
    # Weighted average of the means
    marginal_means = np.average(gmm.means_, weights=gmm.weights_, axis=0)
    # Check if the covariances are in the 'full' or 'diagonal' format
    if len(gmm.covariances_.shape) == 3:  # Full covariance matrix
        marginal_vars = np.average(gmm.covariances_, weights=gmm.weights_, axis=0)
        marginal_scales = np.sqrt(np.diagonal(marginal_vars))
    else:  # Diagonal or spherical covariance matrix
        marginal_vars = np.average(gmm.covariances_, weights=gmm.weights_, axis=0)
        marginal_scales = np.sqrt(marginal_vars)
    PDF_Th_per_feature = norm.ppf(Th, loc=marginal_means, scale=marginal_scales)
    PDF_Th = np.mean(PDF_Th_per_feature)  # Taking average
    return PDF_Th

def fit_one_gmm(samples, n_components):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(samples)
    bic = gmm.bic(samples)
    return bic, gmm

@tim
def fit_gmm_serial(samples):
    # Initialize some parameters
    max_components = 15
    lowest_bic = np.inf
    # Initial guess
    best_gmm = None
    for n_components in range(1, max_components + 1):
        bic, gmm = fit_one_gmm(samples, n_components)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm
            initial_gmm = gmm  # Update the initial GMM for the next iteration
            best_n_components = n_components
    print(f"Best number of components: {best_n_components}")
    return best_gmm

@tim
def inference_gmm(gmm, samples, threshold):
    prediction = (gmm.score_samples(samples) > threshold).astype(int)
    return prediction

def single_GMM(WI, PW, num_PW):
    # Step 1: Initial GMM Fit
    PW_indices = get_random_pixels(PW.compute(), [1], num_PW)
    arr_PW_indices = np.array(PW_indices)
    In_features = WI.compute()[:,arr_PW_indices[:, 0], arr_PW_indices[:, 1]].T
    predict_features = WI.compute().reshape((WI.shape[0], -1)).T
    # Assuming WI has a shape (channels, rows, cols)
    _, num_rows, num_cols = WI.shape
    # Convert PW_indices to 1D indices
    In_indices_1D = arr_PW_indices[:, 0] * num_cols + arr_PW_indices[:, 1]
    # Generate a boolean index array for predict_features
    In_indices = np.zeros(predict_features.shape[0], dtype=bool)
    In_indices[In_indices_1D] = True
    # Fit the self-supervised GMM
    initial_gmm = fit_gmm_serial(In_features)  # Assume fitgmm function exists
    ini_PDF = get_PDF_Th(0.80, initial_gmm) # Actually, the CDF threshold 0.8 should be fine-tuned.
    ini_prediction = initial_gmm.score_samples(predict_features)
    predict_initial = (ini_prediction > ini_PDF).astype(int)
    return predict_initial

def write_predict_nparray(outputfile, predictions,non_valid_mask, ref_src, shapes):
    # Reshape the predicted target variable to match the shape of the `values_arr` mask
    predicted_mask = np.reshape(predictions, shapes)
    ref_src.update(dtype=rasterio.uint8, nodata=255, count=1)
    predicted_mask_write = da.where(non_valid_mask, ref_src['nodata'], predicted_mask).astype(ref_src['dtype'])
    write_output(outputfile, predicted_mask_write, ref_src)

def isolateF_positive(WI, PW, num_PW):
    PW_indices = get_random_pixels(PW.compute(), [1], num_PW)
    Total_Sample_indices =  PW_indices
    In_features = WI.compute()[:,np.array(Total_Sample_indices)[:, 0], np.array(Total_Sample_indices)[:, 1]].T
    In_features = da.from_array(In_features,chunks='auto')
    # Train the isolation forest
    clf = ParallelPostFit(IsolationForest(contamination=0.1), scoring="accuracy")
    clf.fit(In_features)
    targets = da.ones(len(PW_indices))
    print(f'Fitting score: {clf.score(In_features,targets)}')
    # Predict the scores (1 for inliers and -1 for outliers)
    predict_features = da.reshape(WI,(WI.shape[0],-1)).T
    preds = clf.predict(predict_features)
    preds.rechunk((preds.shape[0], 8192))
    print(preds.shape)
    predictions = (preds == 1).astype(np.int)
    return predictions

def quantile_based_th(in_ras, thPD=45):
    # thPD: quantile threshold for sample cleaning
    # return a quantile based threshold
    flat_arr = in_ras.flatten()
    quantile = np.percentile(flat_arr, thPD)
    print(quantile.compute())
    '''tempory solution for Planet'''
    return quantile

def SWD(input_optical,input_cloud,outputfile,info_dict,PW_threshold=25,geoextent=None): 
    geoextent=None
    if isinstance(PW_threshold, str):
        PW_threshold = int(PW_threshold)
    if isinstance(info_dict, str):
        info_dict = ast.literal_eval(info_dict)
    print(info_dict)
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
    wop_raw, LCC_raw = dask.compute(rc_wop, rc_gplcc)
    #prepare sample
    arr_image = read_todask(data_image.ref)
    if input_cloud != 'None':
        arr_cloud = read_todask(data_cloud.ref)
    #get the candidate bands
    WI = get_waterindex(arr_image, info_dict)

    if input_cloud != 'None':
        CM = cloud_mask(arr_cloud,info_dict).squeeze()
    else:
        CM = da.where(WI[0,:,:]==0,False,False)
    non_valid_mask = da.logical_or(CM,da.where(arr_image[0,:,:]==ref_src['nodata'],True,False))

    LCC_raw = da.where(non_valid_mask,0,LCC_raw)
    wop_raw = da.where(non_valid_mask,0,wop_raw)

    # Apply the mask to each band using masked arrays
    masked_image = np.ma.masked_array(WI, mask=np.broadcast_to(non_valid_mask, WI.shape))
    print("Input band size: ", masked_image.shape)
    print("Start implementing AdaI for initial samples")
    # Calculate the AdaI threshold
    #I_vis = masked_image[0, :, :] + masked_image[1, :, :] + masked_image[2, :, :]
    #Threshold_vis = AdaI(I_vis, [], thProbSeeds=90)
    #I_nir = masked_image[3, :, :]
    #Threshold_nir = AdaI(I_nir, [], thProbSeeds=90)
    I_negative_ndwi = masked_image[4, :, :]
    #Threshold_negative_ndwi = AdaI_PW_adj(I_negative_ndwi, wop_raw, thProbSeeds=90)
    Threshold_negative_ndwi = quantile_based_th(I_negative_ndwi)
    I_ndvi = masked_image[5, :, :]
    #Threshold_ndvi = AdaI_PW_adj(I_ndvi, wop_raw, thProbSeeds=90)
    Threshold_ndvi = quantile_based_th(I_ndvi)
    
    #print(f"Threshold_vis: {Threshold_vis}", f"Threshold_nir: {Threshold_nir}", f"Threshold_swir: {Threshold_swir}", f"Threshold_ndvi: {Threshold_ndvi}")
    print(f"Threshold_swir: {Threshold_negative_ndwi}", f"Threshold_ndvi: {Threshold_ndvi}")
    
    #NIR_thless = np.where( (I_vis < Threshold_vis) | (I_nir < Threshold_nir) | (I_swir < Threshold_swir) | (I_ndvi < Threshold_ndvi), 1, 0)
    NIR_thless = np.where( (I_negative_ndwi <= Threshold_negative_ndwi) | (I_ndvi <= Threshold_ndvi), 1, 0)
    PW = da.where(wop_raw>=PW_threshold,1,0)
    PW = da.where(NIR_thless==1, PW, 0)
    num_PW = da.count_nonzero(PW==1).compute()
    if num_PW<=1:
        raise ValueError(f"Not enough persistent water in the image")
    
    num_PW = max(3e5, int(num_PW*0.3))
    WI = da.concatenate((I_negative_ndwi[np.newaxis, :, :], I_ndvi[np.newaxis, :, :]), axis=0)
    WI = da.where(non_valid_mask, 0, WI)
    
    prediction = single_GMM(WI, PW, num_PW)
    
    shapes = (1,WI.shape[1],WI.shape[2])
    write_predict_nparray(outputfile, prediction, non_valid_mask, ref_src, shapes)
  
if __name__ == "__main__":
    # Get the number of input arguments (excluding the script name)
    SWD(*sys.argv[1:])