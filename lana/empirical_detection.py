from packages import *
from params import *


### Code for Transition Detection and Recovery Fitting ###
p0 = [-1, -0.1]
bounds = ([-np.inf,-np.inf], [0, 0]) # [min a, min b], [max a, max b]

def exp_fit(x, a, b): 
    # we define recovery negatively, thus -b
    return a*np.exp(b*x)

def exp_jac(x, a, b):
    # jacobian (partial derivates) for faster and more stable fitting
    jac = np.array([np.exp(b*x), a*x*np.exp(b*x)]).T
    return jac

def get_rsq(ser, popt):
    # rsquared statistic: how well does the model explain the variation in the data?
    residuals = ser - exp_fit(np.arange(ser.shape[0]), *popt)
    ss_res = np.nansum(residuals**2)
    ss_tot = np.nansum((ser - np.nanmean(ser))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def transition(x, win_size, pct=99, ret=False):
    # The idea:
    # Slide a window through the time series
    # At each step, compare the mean of the first and second half of the window (a rough derivative).
    # Smooth results to reduce noise (savgol_filter).
    # Keep only the largest changes (above a percentile threshold).
    # Find peaks in the change signal (transition points).
    # Estimate widths of these transitions.
    # return:
    # 1. transition_times: indices where big changes occur
    # 2. widths: how wide the transitions are

    # Calculate the half window size for smoothing
    half_window = int(win_size / 2)

    # Get the length of the input array
    ln = x.shape[0]

    # 1. Create rough "derivative"
    # Slide a window through the time series
    # At each step, compare the mean of the first and second half of the window (average change at this point, i.e. a rough derivative), with
    # Large positive values → sudden drop (before > after).
    # Large negative values → sudden rise.
    av_derivative = [np.nanmean(x[i - half_window : i]) - np.nanmean(x[i : i + half_window]) for i in range(half_window, ln-half_window)] 
    # Apply a Savitzky-Golay filter for peak-preserving smoothing
    av_derivative = np.array(av_derivative)
    av_derivative = savgol_filter(av_derivative, 7, 1, deriv=0)
    # make it the same length as the input
    av_derivative = np.array([np.nan]*half_window + list(av_derivative) + [np.nan] * half_window)

    # Create a copy of the smoothed derivative array
    av_derivative_masked = av_derivative.copy()

    # Set values below a certain percentile threshold to 0
    av_derivative_masked[av_derivative < np.nanpercentile(av_derivative, pct)] = 0 

    # Find relative maxima in the masked derivative array
    transition_times = argrelmax(av_derivative_masked, order=1)[0] 

    return transition_times

    

def fit_to_transition(transition_date, detrended, raw, dtime, n_local_min=8, return_for_plot=False):
    # detrended is the detrended and deseasoned data
    # raw is the raw time series
    # dtime is the an array of the dates in both, detrended and raw (must be the same!)
    """
    Fit an exponential recovery to data after a transition.

    Parameters
    ----------
    transition_date : np.datetime64
        Date of detected transition.
    detrended : pandas.Series
        Detrended, deseasoned data.
    raw : pandas.Series
        Raw data.
    dtime : np.ndarray of datetime64
        Dates corresponding to detrended and raw (must be aligned).
    n_local_min : int
        Number of points to search for first local minimum.
    return_for_plot : bool
        If True, returns extra info for plotting/debugging.

    Returns
    -------
    tuple
        raw_drop, fit_b, rsq, submn, subvar, prevmn, prevvar, ks, pval
    """
    # Safety: Ensure alignment
    if not (len(dtime) == len(detrended) == len(raw)):
        raise ValueError("dtime, detrended, and raw must have the same length.")
    
    end_date  = transition_date + np.timedelta64(3*365,'D')
    prev_date = transition_date - np.timedelta64(3*365,'D')

    # Ensure transition is within range for fitting
    if end_date > dtime.max():
        return (0., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1., np.nan)

    # Mask post-transition period for fitting
    fit_mask = (dtime >= transition_date) & (dtime <= end_date)
    fitting     = detrended[fit_mask].copy()
    fitting_raw = raw[fit_mask].copy()
    
    # Handle case with insufficient data
    # if fitting.size < n_local_min:
    if np.invert(np.isnan(fitting)).sum() < n_local_min:
        return (0., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1., np.nan)
    
    #Find the closest local minima to start the fitting from
    armin = np.nanargmin(fitting[:n_local_min])
    fitting_min = fitting[armin:]
    
    #See how far the data drops during the transition (to catch spurious transitions)
    raw_drop = fitting_raw[0] - fitting_raw[armin]
    
    #Get the fit statistics
    try:
        trange = np.arange(fitting_min.shape[0])
        approximate_equilibrium = np.nanmean(fitting_min) # assume that after half of the time, the data is at equilibrium
        # approximate_equilibrium = np.nanmean(fitting_min) # assume that after half of the time, the data is at equilibrium
        fitting_min_eq_0 = fitting_min - approximate_equilibrium
        popt, _ = curve_fit(exp_fit, trange, fitting_min_eq_0, p0=p0, jac=exp_jac, bounds=bounds)
        rsq = get_rsq(fitting_min_eq_0, popt)
        fit = popt[1]
        rel_fit_at_1 = exp_fit(1, *popt) / np.sqrt(np.nanvar(fitting_min_eq_0[len(fitting_min_eq_0)//2:]))
    except Exception as e:
        print("Curve fitting failed:", e)
        fit, rsq, popt, rel_fit_at_1 = np.nan, np.nan, None, np.nan

    if return_for_plot: return armin, popt, trange, fitting_min_eq_0, rsq, fit
    #Get the before/after mean/std
    subser  = raw[fit_mask]

    prevser = raw[(dtime < transition_date) & (dtime >= prev_date)]

    submn,  subvar  = np.nanmean(subser),  np.nanstd(subser)
    prevmn, prevvar = np.nanmean(prevser), np.nanstd(prevser)

    #Add KS Statistic
    try:
        ks, pval = ks_2samp(prevser, subser)
    except:
        ks, pval = np.nan, np.nan
    
    return raw_drop, fit, rsq, submn, subvar, prevmn, prevvar, ks, pval, rel_fit_at_1

    
    
def emp_rr(raw, detr, dtime, raw_drop_min, n_local_min):
    transition_times = transition(raw, win_size=9)
    bestRsq     = 0.
    best = np.zeros(11)
    best[:] = np.nan
    for i, t in enumerate(transition_times):
        tmp = fit_to_transition(
            transition_date = dtime[t], 
            detrended       = detr, 
            raw             = raw,
            dtime           = dtime,
            n_local_min     = n_local_min)
        raw_drop, fit, rsq, _, _, _, _, _, pval, rel_fit_at_1 = tmp
        if (raw_drop >= raw_drop_min and pval > 0.05 and rsq >= bestRsq and fit > -3): 
            best[:-1] = tmp
            best[-1] = t
    return np.array(best)





def empirical_recovery_rates(draw, ddetr, timeRes=timeRes, time_dim='time'):
    print("Using empirical detection methods following Taylor strictly, with $\lambda$ defined as <0")
    print("BUT \n1. fit over 3 years")
    print("2. best instead of average")
    print("3. raw drop relative to std of detrended data")
    draw.load()
    ddetr.load()
    dtime = ddetr.time.values
    raw_drop_min = emp_raw_drop_min_scaling * np.float64(ddetr.std().values)
    n_local_min  = emp_n_local_min
    # n_local_min = 8
    # raw_drop_min = 0.01
    empFit =  xr.apply_ufunc(
        emp_rr, 
        draw.sel(time=dtime), 
        ddetr.sel(time=dtime),
        input_core_dims=[[time_dim], [time_dim]],   # dimension not to "broadcast" (=loop) over
        output_core_dims=[['empiricalRecovery']],#["T"], ["c"]],
        vectorize=True,                 # vectorize to apply it on single lat-lon-combinations
        kwargs = {"dtime": dtime, "raw_drop_min": raw_drop_min, "n_local_min": n_local_min}
        )
    return empFit.assign_coords({"empiricalRecovery": ['raw_drop', 'recoveryRate', 'rSquared', 'submn', 'subvar', 'prevmn', 'prevvar', 'ks', 'pval', 'rel_fit_at_1', 'transTime']})
 
