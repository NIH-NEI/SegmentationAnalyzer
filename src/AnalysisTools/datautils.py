from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

from src.AnalysisTools import experimentalparams
from src.AnalysisTools import types


def create3dlist(len2: int = experimentalparams.USEDTREATMENTS, len1: int = experimentalparams.USEDWEEKS):
    """
    Create 3d lists with dimensions len1 and len2
    :param len2: number of lists of lists
    :param len1: size of each list
    :return: list of (lists of size len1)
    """
    return [[[] for w in range(len1)] for t in range(len2)]


def createlistof3dlists(n: int = 7, len2: int = experimentalparams.USEDTREATMENTS, len1: int = experimentalparams.USEDWEEKS):
    """

    :param n: number of lists of lists
    :param len2: number of lists of lists
    :param len1: size of each list
    :return: n list of (lists of size len1)
    """
    listof3dlists = []
    for i in range(n):
        listof3dlists.append(create3dlist(len2, len1))
    return listof3dlists


def checkfinite(vals: types.ArrayLike, debug: bool = False):
    """
    Checks all values contained in input data structures for finite or non finite. If any value is not finite, returns False.
    :param vals: list, tuple, ndarray or set of values (any dimensions)
    :param debug: for debugging
    :return: Boolean value True = all values are finite; False = atleast 1 nonfinite encountered
    """
    arefinite = True
    isfinite = True
    ignoretypes = types.ArrayLike
    for val in vals:
        if not isinstance(val, ignoretypes):
            isfinite = np.isfinite(val)
            if not isfinite:
                if debug:
                    print("nonfinite encountered: ", val)
                return False
        else:
            isfinite = checkfinite(val)
        arefinite = isfinite and arefinite
    return arefinite


def generatedataframe(stackdata, propertyname: str = "Propertyname"):
    """
    Converts stack based data into a pandas dataframe using multi-indexing.
    :param stackdata: data divided into stacks
    :param propertyname: name of property
    :return: Multiindexed Dataframe
    """
    y = boolean_indexing(stackdata)
    names = ["Treatment", "Week", "id"]
    labeldata = np.arange(y.shape[-1])
    index = pd.MultiIndex.from_product(
        [experimentalparams.TREATMENT_TYPES, experimentalparams.WS[:experimentalparams.USEDWEEKS], labeldata],
        names=names)
    return pd.DataFrame({propertyname: y.flatten()}, index=index).reset_index()


def generatedataframeind(stackdata, propertyname: str = "Property", useboolean: bool = False):
    """
    TODO: confirm difference
    :param stackdata:
    :param propertyname:
    :param useboolean:
    :return:
    """
    y = boolean_indexing(stackdata)
    shape = y.shape
    labelids = np.arange(shape[-1])
    names = ["Treatment", "Week", "ID"]
    # labeldata = np.arange(shape[-1])
    index = pd.MultiIndex.from_product([experimentalparams.USEDTREATMENTS, experimentalparams.USEDWEEKS, labelids], names=names)
    df = pd.DataFrame({propertyname: y.flatten()}, index=index).reset_index()
    return df


def boolean_indexing(listoflists, fillval=np.nan) -> np.ndarray:  #
    """

    :param listoflists: list of list data structure
    :param fillval: fill empty cells with given value
    :return: boolean indexed array
    """
    lens = np.asarray([[len(inner) for inner in outer] for outer in listoflists])
    mask = np.array([inlens[:, None] > np.arange(lens.max()) for inlens in lens])
    out = np.full(mask.shape, fillval)
    print(lens, mask.shape, out.shape)

    out[mask] = np.concatenate(np.concatenate(listoflists))
    return out


def getFileListContainingString(folder, s) -> list:
    """
    returns a list of filenames in selected folder containing the string s
    :param folder: path to folder
    :param s: substring
    :return: list of filenames containing s
    """
    return [f for f in listdir(folder) if isfile(join(folder, f)) if f.__contains__(s)]


def orderfilesbybasenames(dnafnames, actinfnames, GFPfnames, debug=False) -> tuple:
    """
    returns ordered list of filenames for DNA, Actin and GFP channel. This is to ensure the code is robust to any unintentional shuffling of files.
    :param dnafnames: list of filenames - DNA Channel.
    :param actinfnames: list of filenames - Actin Channel.
    :param GFPfnames: list of filenames - GFP Channel.
    :param debug: Prints the basename values for each channel for debugging.
    :return: ordered lists of the 3 channels with corresponding filenames at same list index.
    """
    dnafiles, actinfiles, GFPfiles = [], [], []
    print(len(dnafiles), len(actinfiles), len(actinfiles))
    print(len(dnafnames), len(actinfnames), len(GFPfnames))

    for df in dnafnames:
        basedname = "_".join(df.split("_")[:-2])
        if debug:
            print(basedname)
        for af in actinfnames:
            baseaname = "_".join(af.split("_")[:-3])
            if debug:
                print(baseaname)
            if basedname == baseaname:
                for Tf in GFPfnames:
                    baselname = "_".join(Tf.split("_")[:-1])  # check
                    if debug:
                        print(GFPfnames)
                    if basedname == baselname:
                        #                     print(baselname, basedname, baseaname)
                        dnafiles.append(df)
                        actinfiles.append(af)
                        GFPfiles.append(Tf)
                        baseaname = ""
                        basedname = ""
                        baselname = ""
                        break
    print(len(dnafiles), len(actinfiles), len(GFPfiles))
    return dnafiles, actinfiles, GFPfiles
