import os
import numpy as np
from EM_threshold import *

def tile_EM(tiles,hand_candidates):
    '''
    Thresholding from tiles using Expectation Maximization
    '''
    selected_tiles = select_backscatter_tiles(tiles, hand_candidates, True)
    tiles = np.log10(tiles) + 30. 
    scaling = 256 / (np.mean(tiles) + 3 * np.std(tiles))
    gaussian_threshold = determine_em_threshold_Otsu(tiles[selected_tiles, :, :], scaling)
    threshold_db = 10. * (gaussian_threshold - 30.)
    return threshold_db

def bound_adj_optical(I_PW, threshold_db, ENL, maxP, minP, I_min_image):
    I_min = np.min(I_PW)
    I_max = np.power(10, threshold_db/10.)
    if I_min >= I_max:
        print('The minimum value of the available water pixels is larger than the maximum value of the available water pixels.')
        I_min = I_min_image
    Iu,Id = I_max,I_min
    ENLa = ENL
    return Iu,Id,ENLa

def normalize_data(data, method="minmax"):
    if method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
    elif method == "zscore":
        mean_val = np.mean(data)
        std_dev = np.std(data)
        normalized_data = (data - mean_val) / std_dev
    else:
        raise ValueError("Invalid normalization method. Choose 'minmax' or 'zscore'.")
    return normalized_data

def SBA(I_raw, mask_array, tile_shape = (400,400)):
    '''
    Split based approach for gobal thresholding, used as the upper bound of persistent water
    '''
    I = np.ma.masked_array(I_raw, mask=mask_array)
    #wo = wo_raw.compute()
    tiles = tile_array(I, tile_shape=tile_shape, pad_value=np.nan)
    #hand_tiles = tile_array_overlap(array_wo, tile_shape=tile_shape, pad_value=np.nan)
    whole_threshold = tile_EM(tiles, np.arange(tiles.shape[0]))
    whole_threshold = np.power(10, whole_threshold/10.)
    return whole_threshold