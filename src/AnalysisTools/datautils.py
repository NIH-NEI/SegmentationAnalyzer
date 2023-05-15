import pandas as pd
from os import listdir
from os.path import isfile, join
from .experimentalparams import *

# tn = 4
"""
tn = 4 for TOM
tn = 4 for FBL
3 for LAMP
3 for sec
"""


def create3dlist(len2: int = USEDTREATMENTS, len1: int = USEDWEEKS) -> list:
    """
    Create 3d lists with dimensions len1 and len2

    :param len2: number of lists of lists
    :param len1: size of each list
    :return: list of (lists of size len1)
    """
    return [[[] for w in range(len1)] for t in range(len2)]


def createlistof3dlists(n: int = 7, len2: int = USEDTREATMENTS, len1: int = USEDWEEKS) -> list:
    """
    Creates n 3d lists with dimensions len1 and len2. Useful when there is need for multiple
    parameters


    :param n: number of lists of lists
    :param len2: number of lists of lists
    :param len1: size of each list
    :return: n list of (lists of size len1)
    """
    listof3dlists = []
    for i in range(n):
        listof3dlists.append(create3dlist(len2, len1))
    return listof3dlists


def checkfinite(vals: ArrayLike, debug: bool = False) -> bool:
    """
    Checks all values contained in input data structures for finite or non finite. If any value is
    not finite, returns False.

    :param vals: list, tuple, ndarray or set of values (any dimensions)
    :param debug: for debugging
    :return: Boolean value True = all values are finite; False = atleast 1 nonfinite encountered
    """
    arefinite, isfinite = True, True
    ignoretypes = ArrayLike
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
    print("CHECK2: ", arefinite)
    return arefinite


def checkfinitetemp(vals: ArrayLike, debug: bool = False) -> bool:
    """
    TODO: update (not currently in use), all iterables
    :param vals:
    :param debug:
    :return:
    """
    arefinite, isfinite = True, True
    # for val in vals:
    #     # print(val, vals)
    #     isfinite = np.isfinite(val)
    #     if not isfinite:
    #         if debug:
    #             print("nonfinite encountered: ", val)
    #         return False
    #     arefinite = isfinite and arefinite
    # # print("CHECK2: ",arefinite)
    return arefinite


def generatedataframe(stackdata: np.ndarray, propertyname: str = "Propertyname") -> pd.DataFrame:
    """
    Converts stack based data into a pandas dataframe using multi-indexing.
    :param stackdata: data divided into stacks
    :param propertyname: name of property
    :return: Multiindexed Dataframe
    """
    print("gdf: ", stackdata.shape, flush=True)
    y = boolean_indexing(stackdata)
    names = ["Treatment", "Week", "ID"]
    labeldata = np.arange(y.shape[-1])
    index = pd.MultiIndex.from_product([TREATMENT_TYPES, WS[:USEDWEEKS], labeldata], names=names)
    return pd.DataFrame({propertyname: y.flatten()}, index=index).reset_index()


def generateindexeddataframe(stackdata: np.ndarray, propertyname: str = "Property", usedchannels="channel",
                             basendim=7) -> pd.DataFrame:
    """

    :param stackdata:
    :param propertyname:
    :param usedchannels:
    :param basendim:
    :return:
    """
    stackdims = stackdata.ndim
    abstraction = basendim - stackdims
    # print(stackdims, abstraction, basendim)
    if not isinstance(usedchannels, list):
        usedchannels = [usedchannels]
    # y = boolean_indexing(stackdata)
    labelids = np.arange(stackdata.shape[-1])  # this may be different for each organelle

    names = ["Treatment", "Week", "Channel", "Well", "FOV", "Cell_ID", "Organelle_ID"]  #
    namevalues = [TREATMENT_TYPES, WS[:USEDWEEKS], usedchannels, WELLS, FIELDSOFVIEW,
                  list(np.arange(MAX_CELLS_PER_STACK)), labelids]

    if abstraction:
        names = names[:-abstraction]
        namevalues = namevalues[:-abstraction]
    index = pd.MultiIndex.from_product(namevalues, names=names)
    # print("INDEX\n",index)
    indexedstack = pd.DataFrame({propertyname: stackdata.flatten()}, index=index).reset_index()
    return indexedstack


def generatedataframeind(stackdata: np.ndarray, propertyname: str = "Property",
                         useboolean: bool = False) -> pd.DataFrame:
    """
    :param stackdata: data divided into stacks
    :param propertyname: name of property
    :param useboolean:
    :return:
    """
    y = boolean_indexing(stackdata)
    labelids = np.arange(y.shape[-1])
    names = ["Treatment", "Week", "ID"]
    # labeldata = np.arange(shape[-1])
    index = pd.MultiIndex.from_product([USEDTREATMENTS, USEDWEEKS, labelids], names=names)
    df = pd.DataFrame({propertyname: y.flatten()}, index=index).reset_index()
    return df


def expandToNdim(stackdata: np.ndarray, setdims: int = 7, mindims: int = 4, maxdims: int = 7) -> np.ndarray:
    dims = stackdata.ndim
    print(dims)
    if dims < setdims:
        for i in range(setdims - dims):
            stackdata = np.expand_dims(stackdata, axis=-1)
    assert (stackdata.ndim == setdims)
    return stackdata


def boolean_indexing(listoflists, fillval=np.nan) -> np.ndarray:  #
    """
    Generates a boolean indexed array with dimensions based on the maximum datapoints (per week per
    treatment). This is required to address the issue that there may be different number of
    datapoints for each category. This is done on a list of list to convert it to a ndarray. empty values are filled with nans.

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
    returns a list of filenames in selected folder containing the string s.

    :param folder: path to folder
    :param s: substring
    :return: list of filenames containing s
    """
    return [f for f in listdir(folder) if isfile(join(folder, f)) if f.__contains__(s)]


def orderfilesbybasenames(dnafnames, actinfnames, GFPfnames, debug=False) -> tuple:
    """
    returns ordered list of filenames for DNA, Actin and GFP channel. This is to ensure the code is
    robust to any unintentional shuffling of files.

    :param dnafnames: list of filenames - DNA Channel.
    :param actinfnames: list of filenames - Actin Channel.
    :param GFPfnames: list of filenames - GFP Channel.
    :param debug: Prints the basename values for each channel for debugging.
    :return: ordered lists of the 3 channels with corresponding filenames at same list index.
    """
    dnafiles, actinfiles, GFPfiles = [], [], []
    # tn=4
    if debug:
        print(len(dnafiles), len(actinfiles), len(actinfiles))
        print(len(dnafnames), len(actinfnames), len(GFPfnames))
        print(dnafnames)
        print(actinfnames)
        print(GFPfnames)
    for df in dnafnames:
        basedname = "_".join(df.split("_")[:-2])
        if debug:
            print("DNA", basedname)
        for af in actinfnames:
            baseaname = "_".join(af.split("_")[:-2])
            if debug:
                print("ACTIN:", baseaname)
            if basedname == baseaname:
                for Tf in GFPfnames:
                    ############################# temporary for inlcuding lamp1 case  TODO:simplify in final version
                    Tftemp = Tf
                    if Tf.__contains__("s2") and not Tf.__contains__("_s2"):
                        Tftemp = Tf.replace("s2", "_s2")
                    baselname = "_".join(Tftemp.split("_")[:3])  # check
                    ############################# temporary for inlcuding lamp1 case  TODO:simplify in final version
                    if debug:
                        print("GFP:", baselname)
                    if basedname == baselname:
                        dnafiles.append(df)
                        actinfiles.append(af)
                        GFPfiles.append(Tf)
                        baseaname, basedname, baselname = "", "", ""
                        break

    no_dnafiles = len(dnafiles)
    no_actinfiles = len(actinfiles)
    no_gfpfiles = len(GFPfiles)
    print(no_dnafiles, no_actinfiles, no_gfpfiles)
    assert no_dnafiles == no_actinfiles == no_gfpfiles, "Number of Actin, DNA and GFP segmentation stacks do not match"
    return dnafiles, actinfiles, GFPfiles, no_dnafiles


def getwr_3channel(df, af, lf, debug=False):
    """
    Checks names of files and ensures the files correspond to each other. Returns week and replicate
    information for calculation purposes.

    :param df: dna file names
    :param af: actin file names
    :param lf: gfp channel file names
    :return: week id, replicate id, week no. replicate number, common base string
    """

    lf = lf.replace("s2", "")  # NOTE: temporary for lamp1
    basestringdna = "_".join(df.split("_")[:-2])
    basestringactin = "_".join(af.split("_")[:-2])
    basesstringgfp = "_".join(lf.split("_")[:3])
    if debug:
        print(basestringdna, basestringactin, basesstringgfp)
    assert basestringdna == basestringactin == basesstringgfp, f"unequal string lengths {basesstringgfp}, {basestringdna}, {basestringactin}"
    s1, r, fov = basestringdna.split("_")
    w = s1.split("-")[1]
    w_ = WS.index(w)
    r_ = int(r[1:]) - 2  # r goes from 2 to 11 - change it to 0-9
    fov_ = int(fov[-1:]) - 1  # fov goes from 1 to 6 - change it to 0-5
    return w, r, w_, r_, fov, fov_, basestringdna


def array_nan_equal(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.array_equal(a[m], b[m])
