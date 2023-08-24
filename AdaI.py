import os
import dask
import dask.array as da
import numpy as np
from scipy.stats.distributions import chi2
from EM_threshold import *
from scipy.ndimage import median_filter
from scipy import ndimage

def to_db(arr):
    result = 10 * da.log10(arr)
    return da.where(result==-np.inf, np.nan, result)

def tail_adjustment(Iu,Id,I,maxP=0.99,minP=0.001):
    '''
    Adjust the tailing threshold to make sure the tailing area is within the range of maxP and minP
    '''
    ratio=1
    scaling=1
    Iu0=Iu
    while ratio>1-maxP:
        Iu=scaling*Iu0
        scaling = scaling+0.5
        mainArea = da.count_nonzero((I<Iu) & (I>Id))
        tailingArea = da.count_nonzero((I>=Iu) & (I<5*Iu))
        ratio=tailingArea/(tailingArea+mainArea)
    ratio=1
    scaling=1
    Id0=Id
    while ratio>1-minP:
        Id1=Id0/scaling
        scaling = scaling+0.5
        mainArea = da.count_nonzero((I<Iu) & (I>Id))
        tailingArea = da.count_nonzero((I>Id1/5 ) & (I<Id))
        ratio=tailingArea/(tailingArea+mainArea)
        Id=Id1
    return Iu,Id

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

def tile_EM_withselect(tiles,selected_tiles):
    '''
    Thresholding from tiles that meets the bimodal test
    using Expectation Maximization
    '''
    tiles = np.log10(tiles) + 30. 
    scaling = 256 / (np.mean(tiles) + 3 * np.std(tiles))
    gaussian_threshold = determine_em_threshold_Otsu(tiles[selected_tiles, :, :], scaling)
    threshold_db = 10. * (gaussian_threshold - 30.)
    return threshold_db

def bound_adj(I_PW, threshold_db, ENL, maxP, minP):
    I_min = np.min(I_PW)
    I_max = np.power(10, threshold_db/10.)
    if I_min >= I_max:
        raise ValueError('The minimum value of the available water pixels is larger than the maximum value of the available water pixels.')
    h, bins = np.histogram(I_PW, bins=500, range=[I_min,I_max])
    Ip = bins[np.argmax(h)]
    xu = chi2.ppf(maxP, df=2*ENL)
    xd = chi2.ppf(minP, df=2*ENL)
    x = np.linspace(xd, xu, 500)
    pd=chi2.pdf(x,2*ENL)
    xp = x[pd.argmax()]
    Iu=(xu/xp*Ip)
    Id=(xd/xp*Ip)
    I_PW_chi2 = I_PW[(I_PW<=I_max) & (I_PW>=I_min)]
    Iu,Id = tail_adjustment(Iu,Id,I_PW_chi2)
    if Iu >= I_max:
        Iu = I_max
    #if Id <= I_min:
    #    Id = I_min
    Id = I_min
    avai_I = I_PW_chi2[(I_PW_chi2<Iu) & (I_PW_chi2>Id)]
    ENLa = np.mean(avai_I)**2/np.var(avai_I)
    return Iu,Id,ENLa

def bound_adj_optical(I_PW, threshold_db, ENL, maxP, minP, I_min_image):
    I_min = np.min(I_PW)
    I_max = np.power(10, threshold_db/10.)
    if I_min >= I_max:
        print('The minimum value of the available water pixels is larger than the maximum value of the available water pixels.')
        I_min = I_min_image
    Iu,Id = I_max,I_min
    ENLa = ENL
    return Iu,Id,ENLa

def PW_buffer(WO_mask,thre_PW,tile_shape=(80,80)):
    bu_size = int(tile_shape[0]//10)
    PW_mask = np.where(WO_mask>=thre_PW,1,0)
    buffer_mask = ndimage.binary_dilation(PW_mask, iterations=bu_size)
    return buffer_mask

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
    
def AdaI_PW_adj(I_raw,wo_raw,thProbSeeds,ENL=7.1,PW_ratio = [0.1,0.6], nodata=0, tile_shape = (80,80), max_db_threshold=1000.0,
                 maxP = 0.99, minP = 0.0001, max_arr = 0.1, min_arr = -9999, L_water_tiles = 15):
    '''
    New version of findRange, which achive the Adaptive Initialization of Liquid water.
    '''
    I = I_raw
    I_min_image = np.min(I)
    wo = wo_raw.compute()
    non_mask = I<=0
    mean_value = np.mean(I[~non_mask])
    I[non_mask] = mean_value
    #I = median_filter(I, size=(5,5))
    I[non_mask]=0
    hand_candidates = []
    array = np.ma.masked_values(I, nodata)
    array_wo = np.where(I>min_arr, wo, 0)
    thProbSeeds = thProbSeeds + 2
    threshold_db = None
    array_tseed = None
    selected_tiles = []
    hand_tiles = tile_array_overlap(array_wo, tile_shape=tile_shape, pad_value=np.nan)
    Max_TP = thProbSeeds
    Min_TP = 0
    Pass_pre = False
    Pass_now = False
    TH_db_pre = None
    while threshold_db == None and thProbSeeds >= 2:
        thProbSeeds = thProbSeeds - 2
        thProbSeeds = (Max_TP + Min_TP) // 2
        if thProbSeeds <= 0:
            thProbSeeds = 1
        PW_bu_area = PW_buffer(wo,thProbSeeds,tile_shape=(80,80))
        #hand_buffer_tiles = tile_array_overlap(PW_bu_area, tile_shape=tile_shape, pad_value=np.nan)
        #hand_candidates = select_water_tiles_buffer(hand_buffer_tiles, thProbSeeds, PW_ratio)
        hand_candidates = select_water_tiles(hand_tiles, thProbSeeds, PW_ratio)
        if len(hand_candidates) >= L_water_tiles:
            #PW_bu_area = PW_buffer(wo,thProbSeeds,tile_shape=(80,80))
            array_tseed = np.ma.masked_where(PW_bu_area == 0, array)
            #array_tseed = np.ma.masked_where(wo<thProbSeeds, array)
            tiles = tile_array_overlap(array_tseed, tile_shape=tile_shape, pad_value=np.nan)
            tiles = np.ma.masked_less_equal(tiles, min_arr)
            selected_tiles = select_tiles(tiles, hand_candidates,lower_median=False)
            if len(selected_tiles) >= 5:
                threshold_db = tile_EM_withselect(tiles,selected_tiles)
                Pass_now = True
            else:
                Pass_now = False
        if Pass_now ==  True:
            Min_TP = thProbSeeds
            TH_db_now = threshold_db
        else:
            Max_TP = thProbSeeds
        if Pass_pre == True and Pass_now == False:
            if Max_TP - Min_TP <= 10:
                threshold_db = TH_db_pre
                thProbSeeds = Min_TP
                break
            else:
                Max_TP = thProbSeeds
        elif Pass_pre == False and Pass_now == True:
            TH_db_pre = TH_db_now
    print(f'Found {len(selected_tiles)} water tiles at Persistent Water Seeds = {thProbSeeds}')
    I_PW = I[(wo>=thProbSeeds) & (I>min_arr)]
    #if array_tseed is None:
    #    PW_bu_area = PW_buffer(wo,thProbSeeds,tile_shape=(80,80))
    #    array_tseed = np.ma.masked_where(PW_bu_area == 0, array)
    #    array = array_tseed
    #else:
    #    array = array_tseed
    #overall tiles
    tiles = tile_array(array, tile_shape=tile_shape, pad_value=np.nan)
    tiles = np.ma.masked_less_equal(tiles, min_arr)
    #tiles = np.ma.masked_greater_equal(tiles, max_arr)
    back_up_threshold = tile_EM(tiles,np.arange(tiles.shape[0]))
    if back_up_threshold > max_db_threshold:
        print(f'Back up too high! Using maximum threshold {max_db_threshold} db')
        back_up_threshold = max_db_threshold
    back_up_Iu, back_up_Id, back_up_ENLa = bound_adj_optical(I_PW, back_up_threshold, ENL, maxP, minP,I_min_image)
    if threshold_db is not None:
        print(f'Main upbound {threshold_db} db, back up upbound {back_up_threshold} db')
        if threshold_db > max_db_threshold:
            print(f'Threshold too high! Using maximum threshold {max_db_threshold} db')
            threshold_db = max_db_threshold
        else:
            print(f'Using threshold {threshold_db} db')
        Iu,Id,ENLa = bound_adj_optical(I_PW, threshold_db, ENL, maxP, minP,I_min_image)
    else:
        print(f'Can not initiliaze the main bounds, Using back up threshold {back_up_threshold} db')
        Iu,Id,ENLa = back_up_Iu, back_up_Id, back_up_ENLa
    print(f'Iu: {Iu}, Id: {Id}, ENLa: {ENLa}, back up Iu: {back_up_Iu}, back up Id: {back_up_Id}, back up ENLa: {back_up_ENLa}')
    # Iu,Id,ENLa,thProbSeeds,back_up_Iu, back_up_Id, back_up_ENLa, PW_bu_area
    return Iu

def AdaI(I_raw, wo_raw, thProbSeeds, PW_ratio = [0.1,0.6], nodata=0, tile_shape = (80,80), max_db_threshold=100.0,
                 maxP = 0.99, minP = 0.0001, max_arr = 0.1, min_arr = 0, L_water_tiles = 15):
    '''
    New version of findRange, which achive the Adaptive Initialization of Liquid water.
    '''
    I = I_raw
    #wo = wo_raw.compute()
    tiles = tile_array(I, tile_shape=tile_shape, pad_value=np.nan)
    #hand_tiles = tile_array_overlap(array_wo, tile_shape=tile_shape, pad_value=np.nan)
    whole_threshold = tile_EM(tiles, np.arange(tiles.shape[0]))
    whole_threshold = np.power(10, whole_threshold/10.)
    return whole_threshold