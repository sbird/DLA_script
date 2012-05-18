"""Function for fitting data with a broken power law using mpfit.
Obtained from http://code.google.com/p/agpy/"""

import numpy as np
import mpfit

def brokenpowerfit(xax, data, err=None, pinit=None,breakpoint=None, quiet=True):
    """
    Fits a broken power law (a line in log-space) to data as a function of x.
    xax and data are assumed to be already in logspace.
    Thin wrapper around mpfit

    Parameters:
        xax - x location of the data
        data - data to fit to
        err - errors of the data; weights the chi-square of the results.
        breakpoint - Initial breakpoint to guess at. If not specified, the median of the data.
        pinit - Initial scale and slope to guess.
            If not specified, uses the measured scale at the breakpoint guess and then
            calculates a simple slope at the breakpoint
        quiet - Will mpfit spit out extra information

    returns: breakpoint, value_at_breakpoint,slope_above_breakpoint,slope_below_breakpoint

    """
    if err is None:
        err = np.ones(data.shape,dtype='float')
    if breakpoint is None:
        breakpoint = np.median(xax)
    if pinit is None:
        breakind=np.argmin(np.abs(breakpoint-xax))
        tind=np.argmin(np.abs(1.1*breakpoint-xax))
        lind=np.argmin(np.abs(0.9*breakpoint-xax))
        uslope = (data[tind]-data[breakind])/(xax[tind]-xax[breakind])
        lslope = (data[lind]-data[breakind])/(xax[lind]-xax[breakind])
        pinit = [data[breakind],lslope,uslope]

    #Non-changing parameters to mpfitfun
    params={'xax':xax,'data':data,'err':err}

    mp = mpfit.mpfit(mpfitfun,xall=[breakpoint,]+pinit,functkw=params,quiet=quiet)

    return mp.params

def broken_fit(p,xax):
    """Evaluate the broken power law fit with the parameters chosen
        Parameters:
        p[0] - breakpoint
        p[1] - scale
        p[2] - lower slope (xax < breakpoint)
        p[3] - higher slope (xax >= breakpoint)
    """
    xdiff=xax-p[0]
    lfit=(p[2]*xdiff+p[1])*(xdiff < 0)
    ufit=(p[3]*xdiff+p[1])*(xdiff >= 0)
    return lfit+ufit

def mpfitfun(p,fjac=None,xax=None,data=None,err=None):
    """This function returns a status flag (0 for success)
    and the weighted deviations between the model and the data"""
    return [0,np.ravel((broken_fit(p,xax)-data)/err)]


def powerfit(xax, data, err=None, pinit=None,breakpoint=None, quiet=True):
    """
    Fits a power law (a line in log-space) to data as a function of x.
    xax and data are assumed to be already in logspace.
    Thin wrapper around mpfit

    Parameters:
        xax - x location of the data
        data - data to fit to
        err - errors of the data; weights the chi-square of the results.
        breakpoint - Point where initial scale is specified.
            If not given, the median of the data.
        pinit - Initial scale and slope to guess.
            If not specified, uses the measured scale at the breakpoint and then
            calculates a simple slope
        quiet - Will mpfit spit out extra information

    returns: breakpoint, value_at_breakpoint,slope

    """
    if err is None:
        err = np.ones(data.shape,dtype='float')
    if breakpoint is None:
        breakpoint = np.median(xax)
    if pinit is None:
        breakind=np.argmin(np.abs(breakpoint-xax))
        tind=np.argmin(np.abs(1.1*breakpoint-xax))
        uslope = (data[tind]-data[breakind])/(0.1*breakpoint)
        pinit = [data[breakind],uslope]

    #Non-changing parameters to mpfitfun
    params={'xax':xax,'data':data,'err':err,'x0':breakpoint}

    mp = mpfit.mpfit(mppowerfun,xall=pinit,functkw=params,quiet=quiet)

    return np.concatenate([[breakpoint,],mp.params])

def mppowerfun(p,fjac=None,xax=None,data=None,err=None,x0=None):
    """This function returns a status flag (0 for success)
    and the weighted deviations between the model and the data
        Parameters:
        p[0] - scale
        p[1] - slope"""
    xdiff=xax-x0
    fit=(p[1]*xdiff+p[0])
    return [0,np.ravel((fit-data)/err)]

