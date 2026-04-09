"""
This script contains the wrapper functions to:
- generate fourier surrogates
- calculate a p value from the surrogate distribution (after using get_trends.py to make trend files)
"""

import numpy as np
import xarray as xr
import scipy.stats as st

## FOURIER SURROGATES

def fourier_surrogates(ts, ns):
    ## generate ns fourier surrogates from timeseries ts
    ts_fourier  = np.fft.rfft(ts)
    random_phases = np.exp(np.random.uniform(0, 2 * np.pi, (ns, ts.shape[0] // 2 + 1)) * 1.0j)
    ts_fourier_new = ts_fourier * random_phases
    new_ts = np.real(np.fft.irfft(ts_fourier_new))
    return new_ts

def fourier_wrapper(ts,nsmp):
    ts = ts - runmean(ts,w=50)
    return np.transpose(fourier_surrogates(ts,nsmp))

## P-VALUE CALCULATION

def f(x):
    if not np.isnan(x[-1]):
        x=x[~np.isnan(x)]
        return st.percentileofscore(x, x[-1])
    else:
        return np.nan

## after creating the surrogates with the above code, the CSD indicator calculation functions need to be run on all these surrogates, and then the trend calculated for the CSD indicators
## once these trends are available, a p-values can be calculated with the code below (which is fitted for large datasets, but can be used for individual time series)

surr_trends = xr.open_dataset('surr_trends.nc').trend
data_trends = xr.open_dataset('data_trends.nc').trend

# make a combined dataset of the data and surrogate trends
d = data_trends.expand_dims(dim='surrogates', axis=0)
d['surrogates']=[int(len(surr_trends['surrogates'].values))]
arr = xr.concat((surr_trends,d),dim='surrogates')

ax = 3 # axis along which to apply the percentile calculation
ppvs = 1- np.apply_along_axis(f, ax, arr)/100
