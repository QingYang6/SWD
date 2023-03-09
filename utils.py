import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling, transform_bounds, calculate_default_transform
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import box

def mergelist(rasters_to_merge):
    """
    merge a list of rasters into memory file
    
    Parameters
    ----------
    rasters_to_merge: a list of rasters to be merged
    Returns
    -------
    out_mem: merged virtual memory file
    """
    # Open each raster using rasterio
    src_files_to_mosaic = [rasterio.open(raster) for raster in rasters_to_merge]
    # Merge the rasters using rasterio.merge.merge()
    mosaic, out_trans = merge(src_files_to_mosaic)
    # Create a MemoryFile to save the merged raster
    out_mem = MemoryFile()
    # Write the merged raster to the MemoryFile using rasterio
    with rasterio.open(out_mem, 'w', driver='GTiff', height=mosaic.shape[1],
                    width=mosaic.shape[2], count=mosaic.shape[0], dtype=mosaic.dtype,
                    crs=src_files_to_mosaic[0].crs, transform=out_trans) as dest:
        dest.write(mosaic)
    # Seek to the beginning of the MemoryFile to read the data
    out_mem.seek(0)
    return out_mem

def reproject_clip(in_ras, ref):
    """
    Parameters
    ----------
    in_rast: input raster file
    ref: ref src
    Returns
    -------
        generator of reprojected data for merging
    """
    with rasterio.open(in_ras) as src:
        in_ras_matrix = src.read(1)
        ref.update({'dtype': src.dtypes[0], 'count': src.count})
        out_mem = MemoryFile()
        with out_mem.open(**ref) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=in_ras_matrix,
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref['transform'],
                    dst_crs=ref['crs'],
                    resampling=rasterio.warp.Resampling.nearest,
                    src_nodata=src.nodata,
                    dst_nodata=src.nodata,
                    copy_src_overviews=True)
        # Seek to the beginning of the MemoryFile to read the data
        out_mem.seek(0)
    return out_mem

def cropfile(input_file,bounds):
    """
    crop a raster based on new bounds into memory file
    
    Parameters
    ----------
    input_file: path to the input raster file
    bounds:
    Returns
    -------
    out_mem: croped virtual memory file
    """
    # Define new_bounds
    new_bounds = box(*bounds)
    # Open the raster file
    with rasterio.open(input_file) as src:
        # Crop the raster
        out_image, out_transform = mask(src, [new_bounds], crop=True)
        # Update the metadata for the cropped raster
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})
    # Save the cropped raster to a MemoryFile
    out_mem = MemoryFile()
    with rasterio.open(out_mem, "w", **out_meta) as dest:
        dest.write(out_image)
    # Rewind the MemoryFile pointer to the beginning
    out_mem.seek(0)
    return out_mem

def is_wgs84(tif_path):
    with rasterio.open(tif_path) as src:
        return src.crs.to_epsg() == 4326

def reproject_to_wgs84(tif_path):
    out_mem = MemoryFile()
    with rasterio.open(tif_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, 'EPSG:4326', src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': 'EPSG:4326',
            'transform': transform,
            'width': width,
            'height': height
        })
        with out_mem.open(**kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs='EPSG:4326',
                    resampling=Resampling.nearest)
    return out_mem