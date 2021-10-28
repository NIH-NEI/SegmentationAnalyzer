import numpy as np
from scipy.stats import norm

from src.AnalysisTools import experimentalparams as ep
from src.AnalysisTools.datautils import create3dlist

USEDTREATMENTS = ep.USEDTREATMENTS
USEDWEEKS = ep.USEDWEEKS
TREATMENT_TYPES = ep.TREATMENT_TYPES
WS = ep.WS


def removeoutliers3dlist(alldata, m: float = 2):
    """
    removes outliers outside of m standard deviations for 3D lists
    :param alldata:
    :param m:
    :return:
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
    removes outliers outside of m standard deviations for 1d arrays
    :param data1darray: 1d array
    :param m: number of standard deviations
    :return:
    """
    newdata = data1darray[abs(data1darray - np.mean(data1darray)) < m * np.std(data1darray)]
    return newdata


def perctosd(percentile: float = 95.452):
    """
    calculate standard deviation based on percent point function
    :param percentile: percentile
    :return: number of standard deviations
    """
    percentile = percentile / 100
    tail = percentile + (1 - percentile) / 2
    z_crit = norm.ppf(tail)
    return z_crit
