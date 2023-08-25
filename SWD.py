"""
SWD: Self-supervised Waterbody Detection, v1.1.0
Create water masks from satellite images using self-supervised classification.
The current version is specifically designed for Planet data.

    1. The code automatically pulls ancillary data, i.e., ESA water occurrence.
    2. The initial sample is automatically generated based on persistent water occurrence and cleaned by global adaptive thresholding with a split-based approach.
    3. The kernel classification model is gaussian mixture model(GMM).
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
import multiprocessing
import glob
import sys
import ast
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds, calculate_default_transform
import numpy as np
from utils import *
from ancillarydata import *
import dask
import dask.array as da
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from concurrent.futures import ProcessPoolExecutor
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
    out_ras = da.concatenate((in_ras, NDWI[np.newaxis, :, :], NDVI[np.newaxis, :, :]), axis=0)
    #out_ras = da.concatenate((R_ratio[np.newaxis, :, :], G_ratio[np.newaxis, :, :], B_ratio[np.newaxis, :, :],\
    #    NDWI[np.newaxis, :, :], NDVI[np.newaxis, :, :]), axis=0)
    return out_ras

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
    max_components = 5
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
    # Fit the self-supervised GMM
    initial_gmm = fit_gmm_serial(In_features)  # Assume fitgmm function exists
    ini_PDF = get_PDF_Th(0.95, initial_gmm) # Actually, the CDF threshold 0.8 should be fine-tuned.
    ini_prediction = initial_gmm.score_samples(predict_features)
    print(ini_prediction.shape)
    predict_initial = np.where(ini_prediction > ini_PDF, 1, 0)
    return predict_initial

def process_chunk(chunk, gmm, threshold):
    prediction = gmm.score_samples(chunk)
    return np.where(prediction > threshold, 1, 0)

def single_GMM_parallel(WI, PW, num_PW, chunk_size=4096):
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
    # Step 1: Initial GMM Fit
    PW_indices = get_random_pixels(PW.compute(), [1], num_PW)
    arr_PW_indices = np.array(PW_indices)
    In_features = WI.compute()[:, arr_PW_indices[:, 0], arr_PW_indices[:, 1]].T
    predict_features = WI.compute().reshape((WI.shape[0], -1)).T
    
    # Fit the self-supervised GMM
    initial_gmm = fit_gmm_serial(In_features)  # Assume fitgmm function exists
    ini_PDF = get_PDF_Th(0.87, initial_gmm)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = []
        for i in range(0, predict_features.shape[0], chunk_size):
            chunk = predict_features[i:i+chunk_size]
            results.append(executor.submit(process_chunk, chunk, initial_gmm, ini_PDF))
        
        combined_results = np.concatenate([result.result() for result in results])
    
    return combined_results

def write_predict_nparray(outputfile, predictions,non_valid_mask, ref_src, shapes):
    # Reshape the predicted target variable to match the shape of the `values_arr` mask
    #with dask.config.set(**{'array.slicing.split_large_chunks': False}):
    predicted_mask = np.reshape(predictions, shapes)
    ref_src.update(dtype=rasterio.uint8, nodata=255, count=1)
    predicted_mask_write = np.where(non_valid_mask, ref_src['nodata'], predicted_mask).astype(ref_src['dtype'])
    print(f"Writing output to {outputfile}")
    write_output(outputfile, predicted_mask_write, ref_src)

def quantile_based_th(in_ras,non_valid_mask,thPD=30):
    # thPD: quantile threshold for sample cleaning
    # return a quantile based threshold
    cal_ras = in_ras[non_valid_mask != True]
    flat_arr = cal_ras.flatten()
    quantile = np.percentile(flat_arr, thPD)
    '''tempory solution for Planet'''
    return quantile.compute()

def SWD(input_optical,input_cloud,outputfile,info_dict={'index':'ALL','cloud_band':[0],'cloud_value':[0]},PW_threshold=25,geoextent=None): 
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
    #rc_gplcc = dask.delayed(get_ancillary)(bounds_wgs84,ref_src, 'getGPLCC')
    #wop_raw, LCC_raw = dask.compute(rc_wop, rc_gplcc)
    wop_raw = dask.compute(rc_wop)
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

    #LCC_raw = da.where(non_valid_mask,0,LCC_raw)
    wop_raw = da.where(non_valid_mask,0,wop_raw)

    # Apply the mask to each band using masked arrays
    #masked_image = np.ma.masked_array(WI, mask=np.broadcast_to(non_valid_mask, WI.shape))
    print("Image size:", WI.shape[1:])
    print("Start sample cleaning...")
    # NWDI and NDVI
    I_negative_ndwi = WI[-2, :, :]
    Threshold_negative_ndwi = SBA(I_negative_ndwi,non_valid_mask)
    I_ndvi = WI[-1, :, :]
    Threshold_ndvi = SBA(I_ndvi,non_valid_mask)
    
    print(f"Histogram_upper_bound: negtive_NDWI {Threshold_negative_ndwi}", f"NDVI {Threshold_ndvi}")
    
    NIR_thless = da.where( (I_negative_ndwi <= Threshold_negative_ndwi) | (I_ndvi <= Threshold_ndvi), 1, 0)
    PW = da.where(wop_raw>=PW_threshold,1,0)
    PW = da.where(NIR_thless==1, PW, 0).squeeze()
    num_PW = da.count_nonzero(PW==1).compute()
    if num_PW<=1:
        raise ValueError(f"Not enough persistent water in the image, try enlarge the input spatial domain")
    
    num_PW = int(num_PW)
    print(f"Number of training samples: {num_PW}")
    WI = da.concatenate((I_negative_ndwi[np.newaxis, :, :], I_ndvi[np.newaxis, :, :]), axis=0)
    
    WI = da.where(non_valid_mask, 0, WI)
    print("Input feature size: ", WI.shape)
    
    prediction = single_GMM_parallel(WI, PW, num_PW)
    shapes = (1,WI.shape[1],WI.shape[2])
    write_predict_nparray(outputfile, prediction, non_valid_mask, ref_src, shapes)
  
if __name__ == "__main__":
    SWD(*sys.argv[1:])