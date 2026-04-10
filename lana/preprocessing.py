##  Packages and Paths
from packages import *
from params import *
from helpers_paths.general import *


# def day(dt):   return dt.astype('datetime64[D]') - dt.astype('datetime64[M]') + 1
# def month(dt): return dt.astype('datetime64[M]') - dt.astype('datetime64[Y]') + 1
# months30 = [4,6,9,11]
# months31 = [1,3,5,7,8,10,12]
# def last_date(dt):
#     if (day(dt) != 28 or month(dt) != 2) and (day(dt) != 30 or month(dt) not in months30) and (day(dt) != 31 or month(dt) not in months31):
#         print("- original last date: ", dt)
#         print("- last date used:     ", dt-day(dt))
#         return dt-day(dt)
#     else: 
#         print("- original and last date used: ", dt)
#         return dt   
            
def reduce_time_resolution(data, timeRes=timeRes):
    data = data.chunk(time=-1, lon=70, lat=-1)
    data["time"] = [np.datetime64(t) for t in data.time.values]
    
    if timeRes=="original":
        # binsize = int(np.ceil(len(data.time)/data.notnull().sum("time").max().values))
        nbins = int(data.notnull().sum("time").max().values)
        times_new = data.time.groupby_bins("time", nbins).mean(skipna=True)
        data = data.groupby_bins("time", nbins).min()
        # data = data.notnull().groupby_bins("time", nbins).sum()
        data["time_bins"] = times_new
        data = data.rename({"time_bins": "time"})
        print("Reduced to mean of time-bins\nnbins (calculated as int(data.notnull().sum('time').max().values)): ", 
              nbins)
    else: 
        # # Reduce data to monthly data 
        # last_day = last_date(data.time.values[-1])
        if last_year == 2024: 
            try: data = data.sel(time=slice(None, np.datetime64("2024-11-30")))
            except: print("last original timestep is: ", data.time.values[-1], ", which is also included.")
        data['time'] = data.time.dt.strftime("%Y-%m")
        # except: data['time'] = data.time.y("%Y-%m")
        if len(data.time) != len(np.unique(data.time)):  
            if timeRes == "monthlyMean":  data = data.groupby('time').mean(dim='time', keep_attrs = True)
            elif timeRes == "monthlyMax": data = data.groupby('time').max( dim='time', keep_attrs = True)
            else: print("Unknown time resolution!")
    
    data["time"] = [np.datetime64(t) for t in data.time.values]
    # data = data.reindex(time = np.sort(data.time))
    return data



def interpolate_missing_time(data, nMaxInterpolation):
    if nMaxInterpolation > 0:
        # interpolate if interpolateNans
        data = data.where(data.isel(time=0).notnull())
        data = data.where(data.isel(time=-1).notnull())
        datanan = np.isnan(data).sum("time")
        nInterpolations = datanan.isin(range(1,nMaxInterpolation+1)).sum().values
        if nInterpolations > 0:
            print("\n>>>>> Interpolation necessary for %i cells (maximum %i time steps) <<<<<\n"%(nInterpolations, nMaxInterpolation))
            print("   -> no interpolation for %i cells"%int((datanan==0).sum().values))
            [print("   -> # timesteps %2i: %i interpolations"%(i, int((datanan==i).sum().values))) for i in range(1, nMaxInterpolation+1)]
            data = data.where(datanan.isin(range(nMaxInterpolation+1)))
            data = data.chunk({"time":-1}).interpolate_na("time")
            data = data.where(np.isnan(data).sum("time")==0)
    return data


def normalize_and_select_cells(data, mask=None, normalize_original=normalize_original, along=None):
    if not mask is None:
        # if "HLU" in mask.variables and "EF" in mask.variables and "Hansen" in mask.variables:	
            # mask = (mask.HLU <= maxHLU) * (mask.EF >= minForest) * (mask.Hansen <= maxHansen)
        data = data.where(mask)
    if normalize_original and along is None:
        data = data-data.mean()
        data = data/data.std()
    elif normalize_original:
        data = data-data.mean(along)
        data = data/data.std(along)
    return data


## ACTUAL Functions for STL: 
# seasonal trend decomposition from statsmodels 
# to be applied on each grid cell
# such that the EWS are based on the detrended residual

def STLresid(d, period=12):
    # calculates the residual after STL on one single np.array
    d.flags.writeable = True # the key to resolve "ValueError: buffer source array is read-only"
    return STL(d, seasonal = stl_seasonal, period = period).fit().resid

def STLparallel(d, time_dim='time', timeRes=timeRes):
    # applies STLresid to a dask-chunked xr.DataArray
    if "monthly" in timeRes:
        period = 12
    else:
        nyears = (d[time_dim].values[-1]-d[time_dim].values[0])/np.timedelta64(365, "D")
        period = len(d[time_dim]) // nyears
    print("period: ", period)
    return xr.apply_ufunc(
        STLresid, 
        d,
        input_core_dims=[[time_dim]],   # dimension not to "broadcast" (=loop) over
        output_core_dims=[[time_dim]],  # dimension that the output will have
        keep_attrs=True,
        dask='parallelized',            # as STLresid is not dask-conform, parallelize it
        vectorize=True,                 # vectorize to apply it on single lat-lon-combinations
        output_dtypes=[float],          # with dask="parallelized" we have to provide that
        kwargs = {"period": period}
        )



# Running mean + harmonic fitting for deseasonalization and detrending, as an alternative to STL (for comparison)
def harmonic_fit(ser, order=3):
    harm_freq = list(range(1, order+1))
    
    # Check if we have enough valid data
    if ser.notna().sum() < order * 2 + 3:  # Minimum observations needed (observations >= n_predictors)
        # Not enough data, return NaN series
        return pd.Series(np.nan, index=ser.index)
    
    x, y = ser.index, ser.values
    
    x = np.array([(x - pd.Timestamp('1970-01-01')).days for x in x]) / 365.25
    x_rad = x * 2 * np.pi #Convert days to radians for harmonic fitting
    
    #Create empty array to hold the independents
    nr_indep = order*2 + 2
    indep = np.empty((y.shape[0], nr_indep))
    
    #Add constant for intercept and then time
    indep[:,0] = 1
    indep[:,1] = x_rad
    
    #Now create the harmonic variables
    i = 2
    for freq in harm_freq:
        cos = np.cos(x_rad * freq)
        sin = np.sin(x_rad * freq)
        indep[:,i] = cos
        i = i + 1
        indep[:,i] = sin
        i = i + 1
        
    model = sm.OLS(y, indep, missing='drop').fit()
    coefs = model.params
    fitted = []
    for t in range(x_rad.shape[0]):
        data = indep[t,:]
        harm_term = np.nansum(coefs*data)
        fitted.append(harm_term)
    fitted = np.array(fitted)
    return pd.Series(ser.values - fitted, index=ser.index)

def runmean(x, w):
    n = x.shape[0]
    xs = np.zeros_like(x)
    for i in range(w // 2):
        xs[i] = np.nanmean(x[: i + w // 2 + 1])
    for i in range(n - w // 2, n):
        xs[i] = np.nanmean(x[i - w // 2 + 1:])
    for i in range(w // 2, n - w // 2):
        xs[i] = np.nanmean(x[i - w // 2 : i + w // 2 + 1])
    return x - xs


def deseason_detrend_harmonic(values, time_index, yrs=5, yl=23):
    """
    Apply deseasonalization and detrending using harmonic fitting.
    
    Parameters:
    -----------
    values : array-like
        Time series values
    time_index : pd.DatetimeIndex
        Time index for the series
    yrs : int
        Years parameter for running mean
    yl : int
        Subperiod length for running mean
    
    Returns:
    --------
    array : residuals after harmonic detrending
    """
    rm_offline = pd.Series(runmean(values, yrs*yl), index=time_index)
    deseason_rolling = harmonic_fit(rm_offline, order=3)
    return deseason_rolling.values


def apply_deseason_detrend_harmonic(d, time_dim='time', timeRes=timeRes):
    """
    Applies deseason_detrend_harmonic to a dask-chunked xr.DataArray.
    
    Works on each time series (for each lat-lon combination).
    """
    time_index = d.time.to_index()
    
    return xr.apply_ufunc(
        deseason_detrend_harmonic, 
        d,
        input_core_dims=[[time_dim]],   # dimension not to "broadcast" (=loop) over
        output_core_dims=[[time_dim]],  # dimension that the output will have
        keep_attrs=True,
        dask='parallelized',            # as function is not dask-conform, parallelize it
        vectorize=True,                 # vectorize to apply it on single lat-lon-combinations
        output_dtypes=[float],          # with dask="parallelized" we have to provide that
        kwargs={"time_index": time_index, "yrs": 5, "yl": 23}
    )
