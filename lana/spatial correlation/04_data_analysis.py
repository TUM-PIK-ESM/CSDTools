#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
from statsmodels.tsa.seasonal import STL
from scipy.stats import linregress, kendalltau

import os
import sys

# parameters
years_HLU    = slice(np.datetime64("2000-01"), np.datetime64("2020-12"))
stl_seasonal = 7
sws          = 5*12 # size of sliding windows in months
max_dist     = 100 # maximum distance to define neighborhood [in km]



# Path to original data (as downloaded)
p = 'path_to_AMSRE/LPRM-AMSR_E_L3_D_SOILM3_V002_*.nc'
# p = '/p/tmp/lanabl/AMSR2/LPRM-AMSR2_L3_D_SOILM3_V001_*.nc4'
p_mask = 'data/cells_to_include.nc'
p_dist = 'data/dist.nc'
    
mask = xr.open_dataarray(p_mask)
dist = xr.open_dataarray(p_dist)

vi = "AMSRE_bandC"

# Paths to save results
path_monthly_data   = "data/monthly_data_%s.nc"%vi
path_csd_indicators = "data/csd_indicator_%s.nc"%vi
path_csd_trends     = "data/csd_trends_%s.nc"%vi


###############################################################################################################
# 1. get data
def preprocess_add_date(ds):
    dn = ds.encoding["source"].split("_")[-1]
    ds = ds.assign_coords({"time": np.datetime64(dn[:4]+'-'+dn[4:6], "M")})
    return ds

# variables 
# AMSRE-E: opt_depth_x (X-band VOD)   and opt_depth_c (C-band VOD)
# AMSRE2:  opt_depth_c1 (C1-band VOD), opt_depth_c2 (C2-band VOD) and opt_depth_x (X-band VOD)
data = xr.open_mfdataset(p, preprocess=preprocess_add_date, concat_dim="time", combine="nested").opt_depth_c
data = data.rename({"Longitude":"lon", "Latitude":"lat"})
data = data.groupby("time").mean().sel(time=years_HLU) # monthly mean
data.to_netcdf(path_monthly_data)

data = data.where(mask).stack(lon_lat = ("lon", "lat")).dropna("lon_lat")
mask_stacked = mask.where(mask).stack(lon_lat = ("lon", "lat")).dropna("lon_lat")
data = data.reindex_like(mask_stacked) # make sure that the distance matrix fits!



###############################################################################################################
# 2. STL: Seasonal Trend decomposition using LOESS
# removing seasonality and trends, i.e. keep residual
def STL_resid(d):
    d.flags.writeable = True      # to resolve "ValueError: buffer source array is read-only"
    return STL(d, seasonal = stl_seasonal, period = 12).fit().resid

def apply_STL_resid(d, time_dim='time'):
    return xr.apply_ufunc(
        STL_resid, 
        d,
        input_core_dims=[[time_dim]],   # dimension not to "broadcast" (=loop) over
        output_core_dims=[[time_dim]],  # dimension that the output will have
        keep_attrs=True,
        dask='parallelized',            
        vectorize=True,                 
        output_dtypes=[float]           
        )
data = data.chunk({"time":-1, "lon_lat":10})
data = apply_STL_resid(data).compute()



###############################################################################################################
# 3. CSD indicators
def get_ac1(data):
    n = data.shape[1]
    return np.diagonal(np.corrcoef(np.append(data[1:, :], data[:-1,:], axis=1), rowvar=False)[:n,n:])

def get_spc(data):
    n = data.shape[1]
    return np.corrcoef(data, data, rowvar=False)[:n, n:]

def csd_on_sw(data, sws, distance_matrix):
    # first dim must be time and may not contain nans
    ntime  = data.shape[0]
    ncells = data.shape[1]
    ac1 = np.zeros((ntime, ncells))
    var = np.zeros((ntime, ncells))
    spC = np.zeros((ntime, ncells))

    ac1[:,:] = np.nan
    var[:,:] = np.nan
    spC[:,:] = np.nan

    for itime in range(sws, ntime+1):
        dtmp = data[itime-sws:itime, :]
        # AC1
        ac1[itime-1,:] = get_ac1(dtmp)
        # Var
        var[itime-1,:]  = np.var(dtmp, axis=0)
        # Spatial Correlations, dims = ("time", "lon_lat")
        spC[itime-1,:] = np.nanmean(get_spc(dtmp), where=distance_matrix, axis=0)
    return np.array([ac1, var, spC])


def apply_csd_on_sw(d, sws, distance_matrix, time_dim='time'):
    d.load()
    sdata = xr.apply_ufunc(
        csd_on_sw, 
        d,
        input_core_dims =[[time_dim, "lon_lat"]],  
        output_core_dims=[["indicator", time_dim, "lon_lat"]],
        vectorize=True,                
	    kwargs = {"sws":sws, "distance_matrix": distance_matrix}
    )
    return sdata.assign_coords({"indicator": ["AC1", "Variance", "SpatialCorrelation"]})


# Calculate CSD indicators to data
data = data.chunk({"time":-1, "lon_lat":-1})
csd_inds = apply_csd_on_sw(
    data.transpose("time", "lon_lat").dropna("time", "all"),
    sws = sws,
    distance_matrix = (dist<=max_dist).values,
    time_dim = "time"
)
csd_inds.unstack().reindex_like(mask).to_netcdf(path_csd_indicators)



###############################################################################################################
# 4. Assess change: Trends & Kendall's Tau
# Linear trend
def linear_trend(d):
    return linregress(range(len(d)), d)[0] # Kendall's Tau: kendalltau(d, range(len(d)))

def apply_trend(d, time_dim='time'):
    d.load()
    tdata = xr.apply_ufunc(
        linear_trend, 
        d,
        input_core_dims=[[time_dim]], 
        output_core_dims=[[]],
        vectorize=True,               
        )
    return tdata

data = data.chunk({"time":-1, "lon_lat":10})
trends = apply_trend(csd_inds.dropna("time", "all"))
trends.unstack().reindex_like(mask).to_netcdf(path_csd_trends)
