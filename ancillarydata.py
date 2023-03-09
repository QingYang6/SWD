import numpy as np
import rasterio
from osgeo import ogr

def glo30list(bounds:tuple) -> list:
    """
    Get the intersect copernicus glo30 data file list based on bounds and offical geojson
    """
    # Open the geojson file
    DEM_GEOJSON = '/vsicurl/https://asf-dem-west.s3.amazonaws.com/v2/cop30-2021.geojson'
    dataSource = ogr.Open(DEM_GEOJSON)
    layer = dataSource.GetLayer()
    # Create an OGR geometry object from the bounds
    # Create an empty polygon
    extent_geom = ogr.Geometry(ogr.wkbPolygon)
    # Add the vertices of the bounding box
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(bounds[0], bounds[1])
    ring.AddPoint(bounds[2], bounds[1])
    ring.AddPoint(bounds[2], bounds[3])
    ring.AddPoint(bounds[0], bounds[3])
    ring.AddPoint(bounds[0], bounds[1])
    extent_geom.AddGeometry(ring)
    # Loop over features and find intersecting features
    elements = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if extent_geom.Intersects(geom):
            file_path = feature.GetField('file_path')
            print(file_path)
            elements.append(file_path)
    # Close the data source
    dataSource = None
    return elements

def getWO(bounds:tuple) -> list:
    """
    Get water occurence data from ESA, based on the geobounds of reference tif or src.
    Resample and clip with rasterio.
    Returns
    -------
        list of webfiles
    """
    
    Hcrop_Ex = bounds
    maxLoESA = int(np.ceil(Hcrop_Ex[2] / 10) * 10)
    minLoESA = int(np.floor(Hcrop_Ex[0] / 10) * 10)
    maxLaESA = int(np.ceil(Hcrop_Ex[3] / 10) * 10 + 10)
    minLaESA = int(np.ceil(Hcrop_Ex[1] / 10) * 10)
    
    if maxLoESA > 0 and minLoESA > 0:
        lons = [str(e) + 'E' for e in range(minLoESA, maxLoESA, 10)]
    elif maxLoESA < 0 and minLoESA < 0:
        lons = [str(w) + 'W' for w in np.absolute(range(minLoESA, maxLoESA, 10))]
    else:
        lons = [str(w) + 'W' for w in np.absolute(range(minLoESA, 0, 10))]
        lons.extend([str(e) + 'E' for e in range(0, maxLoESA, 10)])

    if maxLaESA > 0 and minLaESA > 0:
        lats = [str(n) + 'N' for n in range(minLaESA, maxLaESA, 10)]
    elif maxLaESA < 0 and minLaESA < 0:
        lats = [str(s) + 'S' for s in np.absolute(range(minLaESA, maxLaESA, 10))]
    else:
        lats = [str(s) + 'S' for s in np.absolute(range(minLaESA, 0, 10))]
        lats.extend([str(n) + 'N' for n in range(0, maxLaESA, 10)])
    
    ESArevision = '1_3_2020'
    ESAdatasets = ['occurrence']
    
    url_tmpl, file_tmpl, _ = templatesESAwater(ESArevision)
    elements=[]
    for ds_name in ESAdatasets:
        for lon in lons:
            for lat in lats:
                filename = file_tmpl.format(ds=ds_name, lon=lon, lat=lat)
                url = url_tmpl.format(ds=ds_name, file=filename)
                print(url)
                elements.append(url)
    return elements

def getGPLCC(bounds:tuple) -> list:
    """
    Get land cover data from GongPeng LCC, based on the geobounds of reference tif or src.
    Resample and clip with rasterio.
    Returns
    -------
        list of webfiles
    """
    Hcrop_Ex = bounds
    LCCmainURL = "http://data.ess.tsinghua.edu.cn/data/fromglc10_2017v01/fromglc10v01"
    LCCfileregex = "_LA_LO.tif"
    maxLoLCC = np.ceil(Hcrop_Ex[2])
    minLoLCC = np.floor(Hcrop_Ex[0])
    maxLaLCC = np.ceil(Hcrop_Ex[3])
    minLaLCC = np.floor(Hcrop_Ex[1])
    Lo_arange = np.arange(minLoLCC - 1, maxLoLCC + 1, 1)
    La_arange = np.arange(minLaLCC - 1, maxLaLCC + 1, 1)
    Lo_arangeGPLCC = Lo_arange[Lo_arange % 2 == 0]
    La_arangeGPLCC = La_arange[La_arange % 2 == 0]
    elements = []
    for Lo in Lo_arangeGPLCC:
        for La in La_arangeGPLCC:
            LCCfileName = LCCfileregex.replace("LO", str(int(Lo)))
            LCCfileName = LCCfileName.replace("LA", str(int(La)))
            url = LCCmainURL+LCCfileName
            print(url)
            elements.append(url)
    return elements
    
def templatesESAwater(revision):
    """
    Return url templates for ESA data download.
    Parameters
    ----------
    revision: version of the data
    Returns
    -------
        url and filename templates as well as padding
    """
    REVISIONS = ['1_0', '1_1', '1_1_2019', '1_3_2020']
    v10, v11, v11_2019, v13_2020 = REVISIONS
    url_tmpl = 'http://storage.googleapis.com/global-surface-water/downloads'
    file_tmpl = '{ds}_{lon}_{lat}'
    if revision == v10:
        padding = 15
    elif revision == v11:
        url_tmpl += '2'
        file_tmpl += '_v' + v11
        padding = 20
    elif revision == v11_2019:
        url_tmpl += '2019v2'
        file_tmpl += 'v' + v11_2019
        padding = 24
    elif revision == v13_2020:
        url_tmpl += '2020'
        file_tmpl += 'v' + v13_2020
        padding = 24
    url_tmpl += '/{ds}/{file}'
    file_tmpl += '.tif'
    return (url_tmpl, file_tmpl, padding)