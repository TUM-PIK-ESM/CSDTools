####Function1: calculate dlm
import scipy.linalg
from scipy.stats import t as tdstr
from scipy.stats import norm
from scipy.interpolate import interp1d
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

class Prior:
    def __init__(self, m, C, S, nu): 
        self.m = m # mean of t-distribution 
        self.C = C # scale matrix of t-distribution
        self.S = S # precision ~ IG(nu/2,S*nu/2)
        self.nu = nu # degree of freedom

class Model:
    def __init__(self,Y,X,rseas,delta):
        self.Y = Y
        self.X = X
        self.rseas = rseas
        dd = np.ones(4)*delta
        self.delta = dd
        ntrend = 2;nregn = X.shape[1]; pseas = len(rseas);nseas = pseas*2;
        m = np.zeros([ntrend+nregn+nseas,1])
        C = scipy.linalg.block_diag(1*np.eye(ntrend),1*np.eye(nregn),1*np.eye(nseas))
        S = np.power(0.2,2); nu = ntrend+nregn+pseas;
        pr = Prior(m,C,S,nu)
        self.prior = pr
        
        
def forwardFilteringM(Model):
    # All the parameters estimated here correspond Eqs. 13-16 and the related ones in the Supplementary Information of Liu et al. (2019)
    # notation in the code -> notation in Liu et al., 2019: 
    # m -> m_t; C -> C_t^{**}; nu -> n_t; 
    # a -> a_t; R -> R_t^{**}; F -> F_t; e -> e_t; y -> y_t; Q -> q_t^{**}; f -> f_t; S -> s_t = d_t/n_t
    
    Y = Model.Y
    X = Model.X
    rseas = Model.rseas
    delta = Model.delta
    Prior = Model.prior
    period = 365.25/16
    deltrend = delta[0];delregn = delta[1];delseas = delta[2];delvar = delta[3]
    Ftrend = np.array([[1],[0]]);ntrend = len(Ftrend); Gtrend = np.array([[1,1],[0,1]]);itrend = np.arange(0,ntrend)
    nregn = X.shape[1];Fregn = np.zeros([nregn,1]);Gregn=np.eye(nregn);iregn = np.arange(ntrend,ntrend+nregn)
    pseas = len(rseas);nseas = pseas*2;iseas = np.arange(ntrend+nregn,ntrend+nregn+nseas)
    Fseas = np.tile([[1],[0]],[pseas,1]);Gseas = np.zeros([nseas,nseas]);
    for j in range(pseas):
        c = np.cos(2*np.pi*rseas[j]/period);
        s = np.sin(2*np.pi*rseas[j]/period);
        i = np.arange(2*j,2*(j+1))
        Gseas[np.reshape(i,[2,1]),i] = [[c,s],[-s,c]]
    F = np.concatenate((Ftrend,Fregn,Fseas),axis=0)
    G = scipy.linalg.block_diag(Gtrend,Gregn,Gseas) 
    m = Prior.m; C = Prior.C; S = Prior.S; nu = Prior.nu

    T = len(Y)
    sm = np.zeros(m.shape)
    sC = np.zeros([C.shape[0],C.shape[1],1])
    sS = np.zeros(1)
    snu = np.zeros(1)
    slik = np.zeros(1)
    for t in range(T):
        a = np.dot(G,m)
        R = np.dot(np.dot(G,C),np.transpose(G))
        R[np.reshape(itrend,[-1,1]),itrend] = R[np.reshape(itrend,[-1,1]),itrend]/deltrend
        R[np.reshape(iregn,[-1,1]),iregn] = R[np.reshape(iregn,[-1,1]),iregn]/delregn
        R[np.reshape(iseas,[-1,1]),iseas] = R[np.reshape(iseas,[-1,1]),iseas]/delseas
        nu = delvar*nu
        F[iregn,0] = X[t,]

        A = np.dot(R,F);Q = np.squeeze(np.dot(np.transpose(F),A)+S); A = A/Q; f = np.squeeze(np.dot(np.transpose(F),a))
        y = Y[t]
        
        if ~np.isnan(y):
            e = y-f; ac = (nu+np.power(e,2)/Q)/(nu+1)
            rQ = np.sqrt(Q)
            mlik = tdstr.pdf(e/rQ,nu)/rQ
            m = a+A*e; C = ac*(R-np.dot(A,np.transpose(A))*Q); nu = nu+1; S = ac*S; 
            # About "S = ac*S" (using the notations in Liu et al. (2019)): 
            # s_t = d_t/n_t = (d_{t-1}+e_t^2/(q_t^{**}/s_t))/n_t = s_{t-1} * (n_{t-1}+e_t^2/(q_t^{**})/n_t = ac * s_{t-1}
        else:
            m = a; C = R;
            if t<T-1:
                X[t+1,0] = f
            mlik = np.nan
        sm = np.concatenate((sm,m),axis=1)
        sC = np.concatenate((sC,np.reshape(C,[C.shape[0],C.shape[1],1])),axis=2)
        snu = np.concatenate((snu,[nu]),axis=0)
        sS = np.concatenate((sS,[S]),axis=0)
        slik = np.concatenate((slik,[mlik]),axis=0)  
    return {'sm':sm, 'sC':sC ,'snu':snu,'slik':slik} 

def computeAnormaly(CLM,AvgCLM,date0):
    deltaT = timedelta(days=16)
    anCLM = np.zeros([1,CLM.shape[1]])
    for i in range(CLM.shape[0]):
        st = date0+deltaT*(i); st = st.timetuple().tm_yday
        et = date0+deltaT*(i+1); et = et.timetuple().tm_yday               
        if et<st:
            window = np.concatenate((np.arange(st,365),np.arange(0,et)))
        else:
            window = np.arange(st,et)
        window[window==365] = 0  # leap year
        anCLM = np.concatenate((anCLM,np.reshape(CLM[i,:]- np.mean(AvgCLM[window,:],axis = 0),[1,CLM.shape[1]])),axis=0)
    return anCLM[1:,:]

def Index_low(nn,date0,percentile):
    intervel = 16
    date0_num = date0.toordinal()
    dd = np.arange(date0,date0+timedelta(days=intervel)*len(nn),timedelta(days=intervel))
    dd_num = np.arange(date0_num,date0_num+intervel*(len(nn)),intervel)
    idq = [i for i in range(len(nn)) if np.isfinite(nn[i])] 
    tt1_num = np.arange(dd_num[idq[0]],dd_num[idq[-1]],1)
    f_itp = interp1d(dd_num[idq], nn[idq],kind = 'linear')
    nn_itp = f_itp(tt1_num)
    
    yday = np.array([date.fromordinal(tt1_num[i]).timetuple().tm_yday for i in range(len(tt1_num))])
    
    ndvi_mean = np.array([np.mean(nn_itp[yday==i]) for i in range(1,366)])
    ndvi_std = np.array([np.std(nn_itp[yday==i]) for i in range(1,366)])
    if len(ndvi_mean)==365:
        ndvi_mean = np.concatenate((ndvi_mean,[ndvi_mean[-1]]),axis = 0)
        ndvi_std = np.concatenate((ndvi_std,[ndvi_std[-1]]),axis = 0)
    
    tt2 = np.arange(dd[0],dd[-1],timedelta(days=1))
    tt2_num = np.arange(dd_num[0],dd_num[-1],1)
    yday2 = np.array([date.fromordinal(tt2_num[i]).timetuple().tm_yday for i in range(len(tt2_num))])
    nv = norm.ppf(1-(1-percentile)/2)
    lowboundary = np.array([ndvi_mean[yday2[i]-1]-nv*ndvi_std[yday2[i]-1] for i in range(len(tt2))])
    
    index_low = [i for i in range(len(dd)) if (~np.isnan(nn[i])) and (nn[i]<lowboundary[tt2_num==dd_num[i]])] 
    return index_low
    
def compute_dlm(x, w, win, vid, rseas, dd, dt, epsilon):
    x_dlm = x - np.nanmean(x)
    Y = x_dlm[1:]
    XX = np.column_stack((x_dlm[:-1], w[:-1]))

    M = Model(Y, XX, rseas, dd)
    FF = forwardFilteringM(M)

    dlm = FF["sm"][vid, :]
    dlm[:win] = np.nan
    dlm = np.log(dlm) / (dt * epsilon)

    return dlm

###Function2:calculate ROSA
import numpy as np
import scipy.integrate 
import tqdm
import scipy.signal
import statsmodels.tsa.stattools
import statsmodels.api as sm
import statsmodels.tsa.seasonal
import scipy.stats as st
from scipy.optimize import curve_fit

def window_spec(x,eta,length,dt,method="ratio"):
    assert x.shape == eta.shape
    assert length < x.size
    
    def fitfunction(f,ls):
        return (1/dt)**2/((2*np.pi*f)**2  + ls**2)
    
    ls = np.full_like(x,np.nan)
    ls_err = ls.copy()
    
    for i in tqdm.trange(ls.size-length):
        detx = statsmodels.tsa.tsatools.detrend(x[i:i+length],order=2)
        f,Sxx = scipy.signal.welch(detx,fs=1/dt)
        f,Sff = scipy.signal.welch(eta[i:i+length],fs=1/dt)#,detrend="linear")
            
        if method == "ratio":
            popt, pcov = scipy.optimize.curve_fit(fitfunction,
                                                  f[1:],
                                                  Sxx[1:]/Sff[1:],
                                                  p0=[1.0],
                                                  bounds=(0.0, np.inf))
            ls[i+length] = popt[0]
            ls_err[i+length] = pcov[0][0]
        else:
            def f2m(L):
                return Sff[1:] * (1/dt)**2/((2*np.pi*f[1:])**2  + L**2) - Sxx[1:]
            opt = scipy.optimize.least_squares(f2m,1.0,bounds=(0.0,np.inf))
            ls[i+length] = opt.x.item()
    return ls,ls_err

### Function3: calculate AC1 & var
from scipy import optimize

def calculate_var(timeseries, **kwargs):
    return sum([x**2 for x in timeseries]) / len(timeseries)

def calculate_acov(timeseries, lag=1):
    return sum([timeseries[i] * timeseries[i + lag] for i in range(len(timeseries) - lag)]) / (len(timeseries) - lag)

def calculate_acor1(timeseries, **kwargs):
    return calculate_acov(timeseries, lag=1)/calculate_var(timeseries)

##Function4:lamda_GLSAR
def run_fit_a_ar1(x, w):
    ## calculate the restoring rate of timeseries x in running windows w
    n = x.shape[0]
    xs = np.zeros_like(x)

    for i in range(w):
        xs[i] = np.nan

    for i in range(n - w // 2, n):
        xs[i] = np.nan

    for i in range(w // 2, n - w // 2):
        xw = x[i - w // 2 : i + w // 2 + 1]
        xw = xw - xw.mean() # variations in the window

        p0, p1 = np.polyfit(np.arange(xw.shape[0]), xw, 1)

        xw = xw - p0 * np.arange(xw.shape[0]) - p1 # remove linear trend

        xw = statsmodels.tsa.tsatools.detrend(xw, order=2)
        x_prev = xw[:-1]

        dxw = xw[1:] - x_prev
        
        idc = np.where(~np.isnan(dxw)) # get non-nan indices in difference

        # use these for regression:
        xw2 = xw[idc]
        dxw2 = dxw[idc] 

        if xw2.size > 10:  
            xw2 = sm.add_constant(xw2)
            model = sm.GLSAR(dxw2, xw2, rho=1)
            results = model.iterative_fit(maxiter=10)
            a = results.params[1]
            xs[i+w//2] = a 
        else:
            xs[i+w//2] = np.nan
            
    return xs

## Function5 & 6: lamda_ACS &  lamda_PSD
def acor_struc(timeseries, relevant_lags=1):
    struc = [calculate_acov(timeseries, lag) / calculate_var(timeseries) for lag in range(relevant_lags)]
    return np.array(struc)

def error_acs(params, observed_ac):
    lambda_ = abs(params[0])
    theta_ = abs(params[1])
    lambda_ = min(lambda_,5)
    theta_ = min(theta_,5)
    if abs(lambda_ - theta_) < 0.01:
        theoretical_ac = np.array([np.exp(-lambda_ * x) * (1 + lambda_ * x) for x in range(len(observed_ac))])
    else:
        theoretical_ac = np.array([(theta_ * np.exp(-lambda_ * x) - lambda_ * np.exp(-theta_ * x)) / (theta_ - lambda_)
                                   for x in range(len(observed_ac))])
    if 0:
        plt.plot(observed_ac)
        plt.plot(theoretical_ac)
        plt.show()
        print(params)
    return sum(abs(observed_ac - theoretical_ac) ** 2)
    
def lambda_acs(timeseries, **kwargs):
    observed_ac = acor_struc(timeseries, kwargs["relevant_lags"])
    params = optimize.fmin(error_acs, kwargs["initial"], args=(observed_ac,), maxiter=100000, disp=False)
    if "return_all_params" in kwargs and kwargs.get("return_all_params"):
        return [min([abs(params[0]), abs(params[1])]), max([abs(params[0]), abs(params[1])])]
    else:
        return min([np.abs(params[0]), abs(params[1])])

def power_spectrum(timeseries):
    fouriertransform = np.fft.rfft(timeseries)[1:round(len(timeseries) / 2)]
    return abs(fouriertransform) ** 2 / len(timeseries)

def frequency_mean(timeseries, band_size=10):
    if band_size > 1:
        return np.array([sum(timeseries[i * band_size:(i + 1) * band_size]) / band_size for i in
                         range(round(len(timeseries) / band_size - 1))])
    else:
        return timeseries

def error_psd(params, n, observed_psd):
    lambda_ = abs(params[0])
    theta_ = abs(params[1])
    kappa_ = abs(params[2])
    lambda_ = min(lambda_,5)
    theta_ = min(theta_,5)
    elambda_ = np.exp(lambda_)
    etheta_ = np.exp(theta_)
    cos_array = [np.cos(2*np.pi*f) for f in np.fft.fftfreq(n)[1:round(n/2)]]
    if abs(lambda_ - theta_) < 0.01:
        theoretical_psd = np.array([kappa_ ** 2 / (4 * lambda_**3) * (-lambda_-cos_array[i]*np.sinh(lambda_) +
                                                                np.cosh(lambda_) * (lambda_*cos_array[i] +
                                                                                      np.sinh(lambda_)))
                                    / (cos_array[i]-np.cosh(lambda_))**2 for i in range(len(cos_array))])
    else:
        theoretical_psd = np.array([kappa_**2/(2*lambda_*theta_*(lambda_**2-theta_**2))*(lambda_*(1 - etheta_ ** -2)
                                                                                         / (1 + etheta_ ** -2 - 2
                                                                                            * etheta_ ** -1
                                                                                            * cos_array[i]) - theta_
                                                                                         * (1 - elambda_ ** -2)
                                                                                         / (1 + elambda_ ** -2 - 2 *
                                                                                            elambda_ ** -1
                                                                                            * cos_array[i]))
                                    for i in range(len(cos_array))])
    theoretical_psd = np.log(frequency_mean(theoretical_psd))
    if 0:
        plt.loglog(np.exp(theoretical_psd))
        plt.loglog(np.exp(observed_psd))
        plt.show()
    return sum(abs(observed_psd - theoretical_psd) ** 2)


def lambda_psd(timeseries, **kwargs):
    observed_psd = np.log(frequency_mean(power_spectrum(timeseries)))
    params = optimize.fmin(error_psd, kwargs["initial"], args=(len(timeseries), observed_psd), maxiter=100000, disp=False)
    if "return_all_params" in kwargs and kwargs.get("return_all_params"):
        return [min([abs(params[0]), abs(params[1])]), max([abs(params[0]), abs(params[1])]),
                abs(params[2])]
    else:
        return min([abs(params[0]), abs(params[1])])
