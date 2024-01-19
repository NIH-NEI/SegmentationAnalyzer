import logging

import numpy as np
from scipy.stats import norm, f_oneway, ks_2samp, chisquare, ttest_ind

from src.AnalysisTools import experimentalparams as ep
from src.AnalysisTools.datautils import create3dlist

USEDTREATMENTS = ep.USEDTREATMENTS
USEDWEEKS = ep.USEDWEEKS
TREATMENT_TYPES = ep.TREATMENT_TYPES
WS = ep.WS

def removeoutliers3dlist(alldata, m: float = 2):
    """
    removes outliers outside m standard deviations for 3D lists

    Args:
        alldata: values in 2d list of weeks and treatments.
        m: number of standard deviations included. Data outside this is considered outlier data.

    Returns:
        data with outliers removed
    """
    newdata = create3dlist(USEDTREATMENTS, USEDWEEKS)
    for t, treatment in enumerate(TREATMENT_TYPES):
        for w, weekno in enumerate(WS[:USEDWEEKS]):
            wtdata = alldata[t][w]
            wtarr = np.asarray(wtdata)
            # wtarrp = wtarr[wtarr < 2000]
            wtarrp = removeoutliers(wtarr, m)
            newdata[t][w] = list(wtarrp)
            # a = wtarr[wtarr >= 5000]
            #             print(len(wtdata), len(newdata[t][w]), type(wtdata),type(newdata[t][w]))
    return newdata


def removeoutliers(data1darray, m: float = 2):
    """
    removes outliers outside m standard deviations for 1d arrays

    Args:
        data1darray: 1d array
        m:  number of standard deviations included. Data outside this is considered outlier data.

    Returns:
        data with outliers removed
    """
    newdata = data1darray[abs(data1darray - np.mean(data1darray)) < m * np.std(data1darray)]
    return newdata


# def rmoutliers(alldata, m=2):
#     newdata = create3dlist(usedtreatments, usedweeks)
#     for t, treatment in enumerate(treatment_type):
#         for w, weekno in enumerate(ws[:usedweeks]):
#             wtdata = alldata[t][w]
#             wtarr = np.asarray(wtdata)
#             #             wtarrp = wtarr[wtarr<2000]
#             wtarrp = wtarr[abs(wtarr - np.mean(wtarr)) < m * np.std(
#                 wtarr)]  # Note the abs value --> for both ends
#             newdata[t][w] = list(wtarrp)
#     #             a = wtarr[wtarr>=5000]
#     #             print(len(wtdata), len(newdata[t][w]), type(wtdata),type(newdata[t][w]))
#
#     return (newdata)


def perctosd(percentile: float = 95.452):
    """
    calculate standard deviation based on percent point function.

    Args:
    percentile: percentile

    Returns:
         number of standard deviations
    """
    percentile = percentile / 100
    tail = percentile + (1 - percentile) / 2
    z_crit = norm.ppf(tail)
    return z_crit


# def newmusigma(alldata, percentile):
#     sigma = perctosd(percentile)
#     data = alldata.copy()
#     rmoutlierdata = rmoutliers(data, m=sigma)
#     newmus = np.zeros((usedtreatments, usedweeks))
#     newsigmas = np.zeros((usedtreatments, usedweeks))
#     # for t, treatment in enumerate(treatment_type):
#     #     for w, weekno in enumerate(ws[:usedweeks]):
#     #         wtdata = rmoutlierdata[t][w]
#     #         newmus[t,w] = np.mean(wtdata)
#     #         newsigmas[t,w] = np.std(wtdata)
#     return newmus, newsigmas


def getmusigma(data):
    """
    Get mean and standard deviation of given data

    Args:
        data:

    Returns:
         mu, sigma
    """
    sigma = np.nanstd(data)
    mu = np.nanmean(data)
    return mu, sigma


def getmusigma2d(data):
    """
    Get 2d mean and standard deviation of given data

    Args:
        data:

    Returns:
         mu, sigma
    """
    d = data.reshape((ep.USEDTREATMENTS, ep.USEDWEEKS, -1))
    mus = np.zeros((ep.USEDTREATMENTS, ep.USEDWEEKS))
    sigmas = np.zeros((ep.USEDTREATMENTS, ep.USEDWEEKS))
    for t in range(ep.USEDTREATMENTS):
        for w in range(ep.USEDWEEKS):
            mus[t, w], sigmas[t, w] = getmusigma(d[t, w, :])
    return mus, sigmas


def one_way_anova(listofarrays):
    """
    Return p and f value of one way ANOVA for list of arrays

    Args:
        listofarrays: list of arrays

    Returns:
         ANOVA f value and p value
    """
    assert len(listofarrays) >= 2, f" list must contain 2 or more samples. Currently{len(listofarrays)}"
    try:
        fvalue, pvalue = f_oneway(*listofarrays)
    except Exception as e:
        fvalue, pvalue = np.nan, np.nan
        logging.warning("ANOVA exception:", e)
    return fvalue, pvalue


def kstest(listofarrays):
    """
    Return D stat and p value of one way Kolmogorov-Smirnov test for list of arrays

    Args:
        listofarrays: list of arrays

    Returns:
         Kolmogorov-Smirnov test D stat and p value
    """
    assert len(listofarrays) == 2, f" list must contain 2 samples. Currently{len(listofarrays)}"
    try:
        # print("Listofarrays",*listofarrays)
        ksstat, kspvalue = ks_2samp(*listofarrays)
    except Exception as e:
        ksstat, kspvalue = np.nan, np.nan
        print("listofarrays: ", listofarrays)
        logging.warning("KStest exception:", e)
    return ksstat, kspvalue


def chisquaretest(listofarrays):
    """
    Return chi squared and p value of one way chi-squared test for list of arrays

    Args:
        listofarrays: list of arrays

    Returns:
         chi squared and p value
    """
    
    assert len(listofarrays) == 2, f" list must contain 2 samples. Currently{len(listofarrays)}"
    try:
        chisq, chipvalue = chisquare(*listofarrays)
    except Exception as e:
        chisq, chipvalue = np.nan, np.nan
        logging.warning("ChiSquared exception:", e)

    return chisq, chipvalue


def ttest(listofarrays):
    assert len(listofarrays) == 2, f" list must contain 2 samples. Currently{len(listofarrays)}"
    try:
        f, p = ttest_ind(*listofarrays)
    except Exception as e:
        f, p = np.nan, np.nan
        logging.warning("Ttest exception:", e)
    return f, p


def stackbyabstractionlevel(stackdata, abstraction, fixeddims=6):
    """
    Get stack data averaged along abstraction level
        stackdata: stack data
        abstraction: abstraction level
        fixeddims: fixed value
    Returns:
        stackdata
    """
    dims = stackdata.ndim
    axes = (0, 1, 2, 3, 4, 5, 6)[:dims]  # to account for cell vs organelle dimensions
    if abstraction:
        abstraction = abstraction + dims - fixeddims  # to account for cell vs organelle dimensions
        stackdata = np.nanmean(stackdata, axis=axes[dims - abstraction:dims])
        # print("Abstraction axes", axes[dims - abstraction:dims], stackdata.shape, dims, abstraction)
    return stackdata


def removestackoutliers(stackdata: np.ndarray, abstraction: int = 0, m: float = 2):
    """
    expected dimensions of stackdata: ((usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxorganellepercell))

    Args:
        stackdata: stackdata
        m: number of standard deviations to keep

    Returns:
        stackdata with removed outliers
    """

    stackdata = stackbyabstractionlevel(stackdata, abstraction)
    nstackdata = stackdata.copy()  # a separate copy to avoid mutating stackdata
    s = stackdata.shape
    for treatment in range(s[0]):
        for week in range(s[1]):
            selectedarray = stackdata[treatment, week].copy()
            stdev = np.nanstd(selectedarray)
            mean = np.nanmean(selectedarray)
            condition = np.abs(selectedarray - mean) < m * stdev

            # nooutlierarray = selectedarray.copy()
            selectedarray[~condition] = np.nan
            nstackdata[treatment, week] = selectedarray
            # print(f"treatment: {treatment}, week: {week}, nstackdata dimensions: {nstackdata.shape}, nstackdata [t,w] dimensions: {nstackdata[treatment, week].shape}, original array dimensions: {selectedarray.shape}")
    return nstackdata


if __name__ == "__main__":
    n1, n2, n3, n4, n5 = 5, 6, 1, 12, 234
    exshape = (n1, n2, n3, n4, n5)
    selectedarray = np.random.random(n1 * n2 * n3 * n4 * n5).reshape(exshape)
    mean = np.mean(selectedarray)
    stdev = np.std(selectedarray)
    # nonoutliers = np.any(abs(selectedarray.flatten() - np.mean(selectedarray.flatten())) < 2 * np.std(selectedarray.flatten()))

    condition = np.abs(selectedarray - mean) < 1 * stdev
    newarray = selectedarray.copy()
    newarray[~condition] = np.nan

