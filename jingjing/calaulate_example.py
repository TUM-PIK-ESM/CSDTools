##example
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from rasterio.windows import Window
import statsmodels.api as sm
from scipy import optimize
import statsmodels.tsa.stattools
import statsmodels.api as sm
import statsmodels.tsa.seasonal
from Functions import  forwardFilteringM, Model, window_spec, calculate_var, calculate_acor1, lambda_acs, lambda_psd, run_fit_a_ar1
    
# Randomly generate a time series and noise.
np.random.seed(42)
t = np.linspace(0, 1, 2000)
trend = 0.03 + (0.8 - 0.03) * t * 0.3
noise = np.random.normal(0, 0.1, 2000)
ts = trend + noise
ts = np.clip(ts, 0.03, 0.8)

### set parameters
win = 500
dd = 0.98
rseas = []
vid = 2

ar_list1 = []
var_list1 = []
acs_list1 = []
psd_list1 = []
ac1_Glsar_list1 = [] 
rosa_list1 = []
dlm_list1 = []

x = ts
w = noise

ar_tmp = np.full_like(x, fill_value=np.nan, dtype=float)
var_tmp = ar_tmp.copy()
acs_tmp = ar_tmp.copy()
psd_tmp = ar_tmp.copy()

for start in range(x.size-win):
    end = start + win
    win_data = statsmodels.tsa.tsatools.detrend(x[start:end], order=2)

    var = calculate_var(win_data)
    ac1 = calculate_acor1(win_data)
     
    ### acs & psd need multiply -1
    data1 = lambda_acs(win_data, initial=[1, 1], relevant_lags=3)
    acs = -data1

    data2 = lambda_psd(win_data, initial=[1, 1, 1])
    psd = -data2

    ar_tmp[start+win] = ac1
    var_tmp[start+win] = var
    acs_tmp[start+win] = acs
    psd_tmp[start+win] = psd

rosa, _ = window_spec(x,w,win,1)
ac1_Glsar = run_fit_a_ar1(x, win)

x_dlm = x - np.nanmean(x)
Y = x_dlm[1:]
XX = np.column_stack(([x_dlm[:-1], w[:-1]]))
M = Model(Y,XX,rseas,dd)
FF = forwardFilteringM(M)
dlm = FF.get('sm')[vid,:] 
dlm[:win] = np.nan
dlm = np.log(dlm) 
ac1_Glsar = ac1_Glsar 

ar_list1.append(ar_tmp)
var_list1.append(var_tmp)
acs_list1.append(acs_tmp)
psd_list1.append(psd_tmp)
rosa_list1.append(-rosa)
dlm_list1.append(dlm)
ac1_Glsar_list1.append(ac1_Glsar)