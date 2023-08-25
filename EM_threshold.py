from typing import Literal, Optional, Tuple, Union
import numpy as np
from skimage.filters import threshold_otsu

def select_water_tiles_buffer(tiles: Union[np.ndarray, np.ma.MaskedArray],
                      hand_threshold: float, hand_fraction: list) -> np.ndarray:
    if np.allclose(tiles, 0.0):
        return []
    tile_indexes = np.arange(tiles.shape[0])
    if np.count_nonzero(tiles>=hand_threshold) == 0:
        return []
    else:
        percent_valid_pixels = np.sum(tiles, axis=(1, 2)) / (tiles.shape[1] * tiles.shape[2])
        return tile_indexes[(percent_valid_pixels >= hand_fraction[0]) & (percent_valid_pixels <= hand_fraction[1])]

def select_water_tiles(tiles: Union[np.ndarray, np.ma.MaskedArray],
                      hand_threshold: float, hand_fraction: list) -> np.ndarray:
    if np.allclose(tiles, 0.0):
        raise ValueError(f'All pixels in scene have a Persistent Water occurrence of {0.0} (all water); '
                         f'scene is not a good candidate for water mapping.')

    tile_indexes = np.arange(tiles.shape[0])

    tiles = np.ma.masked_greater_equal(tiles, hand_threshold)
    if np.count_nonzero(tiles>=hand_threshold) == 0:
        return []
    else:
        percent_valid_pixels = np.sum(tiles.mask, axis=(1, 2)) / (tiles.shape[1] * tiles.shape[2])
        return tile_indexes[(percent_valid_pixels >= hand_fraction[0]) & (percent_valid_pixels <= hand_fraction[1])]

def mean_of_subtiles(tiles: np.ndarray) -> np.ndarray:
    sub_tile_shape = (tiles.shape[1] // 8, tiles.shape[2] // 8)
    sub_tiles_mean = np.zeros((tiles.shape[0], 64))
    for ii, tile in enumerate(tiles):
        sub_tiles = tile_array(tile.filled(0), tile_shape=sub_tile_shape)
        sub_tiles_mean[ii, :] = sub_tiles.mean(axis=(1, 2))
    return sub_tiles_mean

def mean_of_subtiles_dask(tiles: np.ndarray) -> np.ndarray:
    sub_tile_shape = (tiles.shape[1] // 8, tiles.shape[2] // 8)
    tiles_dask = da.from_array(tiles, chunks=(1, None, None))
    sub_tiles_mean = tiles_dask.map_blocks(
        lambda tile: tile_array(tile.filled(0), tile_shape=sub_tile_shape).mean(axis=(1, 2)),
        dtype=tiles.dtype,
        chunks=(1, 64)
    ).compute()
    return sub_tiles_mean

def select_backscatter_tiles(backscatter_tiles: np.ndarray, hand_candidates: np.ndarray, lower_median = False) -> np.ndarray:
    '''
    Tile selection for the back up threshold.
    '''
    tile_indexes = np.arange(backscatter_tiles.shape[0])
    sub_tile_means = mean_of_subtiles(backscatter_tiles)
    sub_tile_means_std = sub_tile_means.std(axis=1)
    tile_medians = np.ma.median(backscatter_tiles, axis=(1, 2))
    tile_variance = sub_tile_means_std / tile_medians

    low_mean_threshold = np.ma.median(tile_medians[hand_candidates])
    low_mean_candidates = tile_indexes[tile_medians < low_mean_threshold]
    if lower_median:
        potential_candidates = np.intersect1d(hand_candidates, low_mean_candidates)
    else:
        potential_candidates = np.intersect1d(hand_candidates, tile_indexes)
    for variance_threshold in np.nanpercentile(tile_variance.filled(np.nan), np.arange(1, 96)[::-1]):
        variance_candidates = tile_indexes[tile_variance > variance_threshold]
        selected = np.intersect1d(variance_candidates, potential_candidates)
        sort_index = np.argsort(sub_tile_means_std[selected])[::-1]
        if len(selected) >= 15:
            #print(f'Found 10 tiles with variance > {variance_threshold} and median backscatter < {low_mean_threshold}')
            return selected[sort_index][:10]
    return np.array([])

def tile_array(array: Union[np.ndarray, np.ma.MaskedArray], tile_shape: Tuple[int, int] = (200, 200),
               pad_value: float = None) -> Union[np.ndarray, np.ma.MaskedArray]:
    '''
    Split an array into tiles of a given shape.
    Non overlapping version
    '''
    array_rows, array_columns = array.shape
    tile_rows, tile_columns = tile_shape

    rpad = -array_rows % tile_rows
    cpad = -array_columns % tile_columns

    if (rpad or cpad) and pad_value is None:
        raise ValueError(f'Cannot evenly tile a {array.shape} array into ({tile_rows},{tile_columns}) tiles')

    if rpad or cpad:
        padded_array = np.pad(array, ((0, rpad), (0, cpad)), constant_values=pad_value)
        if isinstance(array, np.ma.MaskedArray):
            mask = np.pad(array.mask, ((0, rpad), (0, cpad)), constant_values=True)
            padded_array = np.ma.MaskedArray(padded_array, mask=mask)
    else:
        padded_array = array

    tile_list = []
    for rows in np.vsplit(padded_array, range(tile_rows, array_rows, tile_rows)):
        tile_list.extend(np.hsplit(rows, range(tile_columns, array_columns, tile_columns)))

    dstack = np.ma.dstack if isinstance(array, np.ma.MaskedArray) else np.dstack
    tiled = np.moveaxis(dstack(tile_list), -1, 0)

    return tiled

def determine_em_threshold_Otsu(tiles: np.ndarray, scaling: float) -> float:
    thresholds = []
    for ii in range(tiles.shape[0]):
        test_tile = (np.around(tiles[ii, :, :] * scaling)).astype(int)
        thresholds.append(thre_otsu(test_tile) / scaling)
        #np.median(np.sort(thresholds)) np.max(np.sort(thresholds)) np.percentile(thresholds,25)
    return np.median(thresholds)

def thre_otsu(tile):
    threshold = threshold_otsu(tile)
    return threshold

def tile_array_overlap(array: np.ndarray, tile_shape: Tuple[int, int] = (80, 80), overlap: float = 0.2, pad_value: float = None) -> np.ndarray:
    """Tile a 2D numpy array with overlapping

    Args:
        array: 2D array to tile
        tile_shape: the shape of each tile
        overlap: overlap between tiles, expressed as a fraction of tile size

    Returns:
        the tiled array
    """
    array_rows, array_columns = array.shape
    tile_rows, tile_columns = tile_shape

    # Calculate the number of tiles along each axis
    num_tiles_x = int(np.ceil(array_rows / (tile_rows - overlap * tile_rows)))
    num_tiles_y = int(np.ceil(array_columns / (tile_columns - overlap * tile_columns)))

    # Calculate the padding needed to evenly tile the array
    rpad = num_tiles_x * tile_rows - array_rows
    cpad = num_tiles_y * tile_columns - array_columns
    
    if (rpad or cpad) and pad_value is None:
        raise ValueError(f'Cannot evenly tile a {array.shape} array into ({tile_rows},{tile_columns}) tiles')

    # Pad the array
    padded_array = np.pad(array, ((0, rpad), (0, cpad)), constant_values=0)

    # Initialize an empty array to store the tiles
    tiles = np.zeros((num_tiles_x * num_tiles_y, tile_rows, tile_columns))

    # Extract tiles
    tile_index = 0
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            x_start = int(i * (tile_rows - overlap * tile_rows))
            y_start = int(j * (tile_columns - overlap * tile_columns))
            x_end = x_start + tile_rows
            y_end = y_start + tile_columns
            tiles[tile_index] = padded_array[x_start:x_end, y_start:y_end]
            tile_index += 1

    return tiles
    
def compute_ad(means, stds):
    '''
    Compute the Ashman D (AD) coefficient using the means and standard deviations of the Gaussian components.
    '''
    delta_mean = means[0] - means[1]
    std_sum = stds[0]**2 + stds[1]**2
    ad_coefficient = delta_mean / np.sqrt(0.5 * std_sum)
    return ad_coefficient