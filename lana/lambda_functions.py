
from packages import *

from params import *
sys.path.append(os.path.dirname(os.getcwd()))

from helpers_methods.spatial_resolution import *
from helpers_methods.preprocessing import *
from helpers_methods.noise import *

warnings.simplefilter('ignore', np.RankWarning)


########################################################
# DETREND
########################################################
def remove_linear_trend_1d(xw):
    if np.isnan(xw.mean()): return [np.nan]*len(xw)
    xn = np.arange(xw.shape[0])
    lg = linregress(xn,xw)
    return xw - lg.slope * xn - lg.intercept

def remove_linear_trend(xw, axis=0):
    return np.apply_along_axis(remove_linear_trend_1d, arr=xw, axis=axis)

def apply_remove_linear_trend(data, time_dim='time'):
    return xr.apply_ufunc(remove_linear_trend, data.transpose(time_dim,...), dask="parallelized")



########################################################
# AC1
########################################################
def ac1_np_1d(data):
    if data.shape[0]==1: return np.nan
    return sm.tsa.acf(data,nlags=1)[1]

def apply_ac1_np(data, axis=0):
    return np.apply_along_axis(ac1_np_1d, arr=data, axis=axis)



########################################################
# GLSAR
########################################################
# Lambda from Generalized Least Squares with AR covariance structure (should allow for colored noise)
# https://github.com/niklasboers/AMOC_EWS/blob/main/EWS_functions.py
def glsar_1d(xw):
    if np.isnan(xw.mean()): return [np.nan, np.nan]
    try:
        dxw = xw[1:] - xw[:-1]
        xw = sm.add_constant(xw)
        model = sm.GLSAR(dxw, xw[:-1], rho=1).iterative_fit(maxiter=10)
        return np.array([model.params[1], model.resid.std()])
    except: return [np.nan, np.nan]

def apply_glsar(xw, axis=1):
    return np.apply_along_axis(glsar_1d, arr=xw[0], axis=axis-1)



########################################################
# Linear Regression
########################################################
def ols_1d(data):
    if data.shape[0]==1: return [np.nan, np.nan]
    dx = data[1:]-data[:-1]
    slope, intercept, _,_,_ = linregress(dx, data[:-1])
    return np.array([slope, np.std(data[:-1] - slope*dx - intercept)])
    noise, model = get_noise_OLS(data)
    return np.array([model.params[0], noise.std()])

def apply_ols(data, axis=1):
    return np.apply_along_axis(ols_1d, arr=data[0], axis=axis-1)



########################################################
# Kramers-Moyal
########################################################
def lambda_KM_1d(km1, edges): 
    return -1*linregress(edges, km1).slope

def sigma_KM_1d(km2): 
    return np.nanmean(np.sqrt(2*km2[km2>=0]))

def estimate_lambda_sigma_KM_1d(kme12):
    if kme12.shape[0] != 3: print("KM to estimates problem!!!!")
    return np.array([lambda_KM_1d(km1=kme12[1,:], edges=kme12[0,:]), sigma_KM_1d(kme12[2,:])])

def km_series_centered_data_1d(data, Delta_t = Delta_t, bin_number = bin_number, bw=None, centerkmfraction = centerkmfraction, **kwargs):
    if data.shape[0]==1: return [np.nan, np.nan]
    if np.isnan(data).any(): return [np.nan, np.nan]
    return_estimates = kwargs["kwargs"]["return_estimates"]
    if bw is None: bw = 14*np.std(data)/bin_number
    ### Stable state is 0 since we do it on de-trended data!
    ### Get estimations of Kramers-Moyal coefficients and keep only the specified fraction around the stable state
    kmc, edges = km(data, bw = bw, bins = np.array([bin_number]), powers = [[0],[1],[2],[3],[4]])
    edges = edges[0]
    # get the inner fraction only
    center_index = np.searchsorted(edges,0)
    left_index  = center_index - round(bin_number*centerkmfraction/2)
    right_index = center_index + round(bin_number*centerkmfraction/2)
    left_index  = max(0,left_index)
    right_index = min(bin_number-1, right_index)
    # get KM1, KM2, KM4
    kmc = kmc[[1,2,4], :]#, left_index:right_index]
    # correct the KM coefficients
    kmc[0] = kmc[0]/Delta_t
    kmc[1] = kmc[1]/Delta_t - Delta_t*np.square(kmc[0])/2
    kmc[1][kmc[1]<0] = 0
    if not return_estimates:
        kmc[2] = kmc[2]/Delta_t
        edges[:left_index]   = np.nan
        edges[right_index:]  = np.nan
        kmc[:, :left_index]  = np.nan
        kmc[:, right_index:] = np.nan
        tmp = np.append(np.append(kmc, [kmc[2]/kmc[1]], axis=0), [edges], axis=0) # KM1, KM2, KM4, edges
        return tmp
        # return np.append(np.append(kmc, [kmc[2]/kmc[1]], axis=0), [edges], axis=0) # KM1, KM2, KM4, edges
    kmc   = kmc[:, left_index:right_index]
    edges = edges[left_index:right_index]
    return estimate_lambda_sigma_KM_1d(np.append([edges], kmc[:2,:], axis=0))

def km_series_centered_data(data, Delta_t = Delta_t, bin_number = bin_number, bw=None, centerkmfraction = centerkmfraction, axis=0):
    kwargs = dict(Delta_t = Delta_t, bin_number = bin_number, bw=bw, centerkmfraction = centerkmfraction, return_estimates=False)
    return np.apply_along_axis(km_series_centered_data_1d, arr=data, axis=axis, kwargs=kwargs)


def km_estimates_centered_data(data, Delta_t = Delta_t, bin_number = bin_number, bw=None, centerkmfraction = centerkmfraction, axis=0):
    kwargs = dict(Delta_t = Delta_t, bin_number = bin_number, bw=bw, centerkmfraction = centerkmfraction, return_estimates=True)
    return np.apply_along_axis(km_series_centered_data_1d, arr=data[0], axis=axis-1, kwargs=kwargs)


def apply_KM_series(d, time_dim='time'):
    # KM-coefficients from timeseries
    d.load()
    KMdata =  xr.apply_ufunc(
        km_series_centered_data_1d, 
        d,
        input_core_dims=[[time_dim]],   # dimension not to "broadcast" (=loop) over
        output_core_dims=[['KMCoefficient', "space"]],
        vectorize=True,                 # vectorize to apply it on single lat-lon-combinations
        kwargs = {"kwargs":{"return_estimates":False}}
        )
    return KMdata.assign_coords({"KMCoefficient": ["KM1", "KM2", "KM4", "ThetaRatio", "edges"]})


def apply_KM_estimates(d, time_dim='time'):
    # lambda & sigma estimates from timeseries (via KM)
    d = d.expand_dims({"estimate": 2}).transpose("estimate", time_dim, ...) # for methods that return estimates of lambda & sigma
    return d.reduce(km_estimates_centered_data, axis=1).assign_coords({"estimate": ["lambda", "sigma"]})


########################################################
# Get lambdas from CSD indicators
#######################################################
def lambda_from_regression(a): 
    return np.log(a+1)

def lambda_from_AC1(ac1, Delta_t=Delta_t):
    return np.log(ac1) / Delta_t

def lambda_from_var_continuous_sigma(var, sigma):
    return np.square(sigma)/(2*var)

def lambda_from_var_discrete_sigma(var, sigma):
    sigma_squared = np.square(sigma)
    sigma_squared = np.where(sigma_squared < var, sigma_squared, np.nan)
    return 0.5* np.log(1- sigma_squared/var)

# def continuous_sigma_from_discrete(sigma, lam, Delta_t=Delta_t):
#     sigma_continuous_squared = 2 * lam * np.square(sigma) / (1-np.exp(-2*lam*Delta_t))
#     sigma_continuous_squared = np.where(sigma_continuous_squared>0, sigma_continuous_squared, np.nan)
#     return np.sqrt(sigma_continuous_squared)

def continuous_sigma_from_discrete(sigma, lam, Delta_t=Delta_t):
    sigma_continuous_squared = -2 * lam * np.square(sigma) / (1-np.exp(2*lam*Delta_t))
    sigma_continuous_squared = np.where(sigma_continuous_squared>0, sigma_continuous_squared, np.nan)
    return np.sqrt(sigma_continuous_squared)


########################################################
# all indicators
########################################################

def calculate_indicators(droll):
    # Mapping to OU parameter notation:
    # AC1      = empirical autocorrelation at lag 1
    # lamAC1   = λ from AC1 formula
    # a_GLS    = regression slope a_r from GLSAR
    # lamGLS   = λ from regression (GLSAR)
    # a_LR     = regression slope a_r from OLS
    # lamLR    = λ from regression (OLS)
    # lamKM    = λ from Kramers–Moyal drift slope
    # dsigGLS  = σ_discrete from GLSAR residuals
    # sigGLS   = σ_continuous from σ_discrete + λ
    # dsigLR   = σ_discrete from OLS residuals
    # sigLR    = σ_continuous from σ_discrete + λ
    # sigKM    = σ_continuous from KM2

    droll = apply_remove_linear_trend(droll, time_dim="tw")
    droll = droll.transpose("tw", ...)

    # AC1
    ac1 = droll.reduce(apply_ac1_np, axis=0).compute()
    lamAC1 = lambda_from_AC1(ac1)
    

    # variance for lambdas from var
    var = droll.var("tw")


    # Linear Regression
    droll = droll.expand_dims({"estimate": 2}).transpose("estimate", "tw", ...) # for methods that return estimates of lambda & sigma
    lr = droll.reduce(apply_ols, axis=1).assign_coords({"estimate": ["a_reg", "sigma_discrete"]})

    alr = lr.sel(estimate="a_reg", drop=True)
    lamLR = lambda_from_regression(alr)
    dsigLR = lr.sel(estimate="sigma_discrete", drop=True)
    sigLR  = continuous_sigma_from_discrete(sigma=dsigLR, lam=lamLR)
    
    
    # GLSAR
    lgl = droll.reduce(apply_glsar, axis=1).assign_coords({"estimate": ["a_reg", "sigma_discrete"]})
    algl = lgl.sel(estimate="a_reg", drop=True)
    lamGLS = lambda_from_regression(algl)
    dsigGLS = lgl.sel(estimate="sigma_discrete", drop=True)
    sigGLS = continuous_sigma_from_discrete(sigma=dsigGLS, lam=lamGLS)


    # Kramers-Moyal
    dkm = droll.reduce(km_estimates_centered_data, axis=1).assign_coords({"estimate": ["lambda", "sigma"]})
    lamKM = dkm.sel(estimate="lambda", drop=True)
    sigKM = dkm.sel(estimate="sigma", drop=True)

    # lambda from variances different corrections
    lamVar = -1./(2*var) # that is actually not correct, but if we do "lambda from var discrete sigma", we will only get nans
    print("lamVar is calculated as 1./(2*var), so it is not corrected for the variance being discrete!")
    lamVLR = lambda_from_var_discrete_sigma(var, dsigLR)
    lamVKM = lambda_from_var_continuous_sigma(var, sigKM)

    return xr.Dataset({
        'AC1':    ac1, 
        'lamAC1': lamAC1,
        'Var':    var,
        'lamVar': lamVar,
        'lamVLR': lamVLR,
        'lamVKM': lamVKM,
        'a_GLS':  algl, 
        'lamGLS': lamGLS, 
        'a_LR':   alr, 
        'lamLR':  lamLR, 
        'lamKM':  lamKM, 
        'dsigGLS':dsigGLS, 
        'sigGLS': xr.DataArray(sigGLS, dims=dsigGLS.dims, coords=dsigGLS.coords),
        'dsigLR': dsigLR, 
        'sigLR':  xr.DataArray(sigLR, dims=dsigLR.dims, coords=dsigLR.coords),
        'sigKM':  sigKM
    }).compute().to_array("estimate")


def calculate_indicators_on_sw(data, sws=sliding_window_size):
    time_orig = data.time
    data = data.rolling(time=sws, min_periods=sws).construct("tw").dropna("time", "any")
    # data = data.chunk({"time":-1, "ll": 1000})
    data = calculate_indicators(data)
    return data.reindex(time=time_orig, copy=False)


def calculate_indicators_on_entireTS(data):
    # on entire time series
    return calculate_indicators(data.rename({"time": "tw"}).transpose("tw", ...))






########################################################
# all indicators SMALL
########################################################

def calculate_indicators_reduced(droll):

    droll = apply_remove_linear_trend(droll, time_dim="tw")

    droll = droll.expand_dims({"estimate": 2}).transpose("estimate", "tw", ...) # for methods that return estimates of lambda & sigma
    # GLSAR
    lgl = droll.reduce(apply_glsar, axis=1).assign_coords({"estimate": ["a_reg", "sigma_discrete"]})
    algl = lgl.sel(estimate="a_reg", drop=True)
    lamGLS = lambda_from_regression(algl)
    dsigGLS = lgl.sel(estimate="sigma_discrete", drop=True)
    sigGLS = continuous_sigma_from_discrete(sigma=dsigGLS, lam=lamGLS)

    return xr.Dataset({
        'a_GLS':  algl, 
        'lamGLS': -lamGLS, 
        'dsigGLS': dsigGLS, 
        'sigGLS': xr.DataArray(sigGLS, dims=dsigGLS.dims, coords=dsigGLS.coords),
    }).compute().to_array("estimate")


def calculate_indicators_on_sw_reduced(data, sws=sliding_window_size):
    time_orig = data.time
    data = data.rolling(time=sws, min_periods=sws).construct("tw").dropna("time", "any")
    # data = data.chunk({"time":-1, "ll": 1000})
    data = calculate_indicators_reduced(data)
    return data.reindex(time=time_orig, copy=False)


def calculate_indicators_on_entireTS_reduced(data):
    # on entire time series
    return calculate_indicators_reduced(data.rename({"time": "tw"}).transpose("tw", ...))







########################################################
# SPATIAL CORRELATION
########################################################

def get_spc(data):
	return np.corrcoef(data, rowvar=False)

def calculate_spc(data, distance_matrix):
	# Calculate Spatial Correlations
	return np.nanmean(get_spc(data), where=distance_matrix, axis=0)

def apply_spc(data, distance_matrix, time_dim='time'):
	data.load()
	spc_data = xr.apply_ufunc(
		calculate_spc, 
		data,
		input_core_dims=[[time_dim, "ll"]],
		output_core_dims=[["ll"]],
		vectorize=True,
		kwargs={"distance_matrix": distance_matrix.values}
	)
	return spc_data