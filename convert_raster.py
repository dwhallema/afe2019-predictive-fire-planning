#!/usr/bin/env python
# coding: utf-8

# # GDAL raster format conversion and compression

# Author: Dennis W. Hallema
# 
# Description: Script for raster to raster conversion and compression with GDAL.
# 
# Depends: `gdal`
# 
# Disclaimer: Use at your own risk. The authors cannot assure the reliability or suitability of these materials for a particular purpose. The act of distribution shall not constitute any such warranty, and no responsibility is assumed for a user's application of these materials or related materials.

# In[ ]:


# Import modules
from osgeo import gdal
gdal.UseExceptions()
gdal.AllRegister()


# In[ ]:


# Raster files to be converted
rasters = [
    'data/barrier.asc',
    'data/costdist.asc',
    'data/flatdist.asc',
    'data/ridgedist.asc',
    'data/valleydist.asc',
    'data/roaddist.asc',
    'data/ros01.asc',
    'data/rtc01.asc',
    'data/sdi01.asc',
    'data/brt_resp_float.asc'
    'data/brt_resp2.asc'
]


# In[ ]:


# Convert raster files
for i, raster in enumerate(rasters):
    ds = gdal.Open(raster)
    t_options = gdal.TranslateOptions(gdal.ParseCommandLine("-of Gtiff -co COMPRESS=LZW"))
    ds_out = raster.replace('.asc','.tif')
    print('Writing {}'.format(ds_out))
    ds = gdal.Translate(ds_out, ds, options=t_options)
    ds = None
print('Done')

