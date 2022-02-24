import json
import pickle
import traceback

import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from tifffile import imread
from src.AnalysisTools import datautils
from src.AnalysisTools import types


#
# def aimg_simple(name):
#     seg = AICSImage(name)
#     seg = 1 * (seg.data > 0)
#     return np.squeeze(seg)
#
#
# def aimgproc(name):
#     seg = AICSImage(name)
#     seg = seg.data.astype(np.int16)
#     seg = seg.max() - seg  # invert
#     seg = seg.squeeze()
#     seg = seg // seg.max()
#     return seg


def opensegmentedstack(name: types.PathLike,
                       whiteonblack: types.SegmentationLike = "default", debug: bool = False):
    """
    Handles simple binary segmentations as well as 3 color segmentations used in project.

    TODO: test
    opens segmented stacks. whiteonblack
    :param debug:
    :param name:
    :param binary: segmented image with only two unique values. For 3 color images
    :param whiteonblack: default indicates if image is whiteonblack or blackonwhite. The defaults may be different for another dataset
    :return:
    """
    binary = None
    seg = imread(name)
    uniquenos = np.unique(seg)
    if len(uniquenos) == 2:
        binary = True
    elif len(uniquenos) == 3:
        binary = False
    try:
        if debug:
            print(f"opening: {name}, binary: {binary},", end="")
        if binary:
            seg = AICSImage(name)
            if whiteonblack == "default":
                whiteonblack = True
            seg = 1 * (seg.data > 0)
        else:
            seg = imread(name)
            # print(type(seg), seg.shape)
            if whiteonblack == "default":
                whiteonblack = False
            seg = seg.astype(np.int16)
            seg = seg.max() - seg  # invert
            seg = seg.squeeze()
            seg = seg // seg.max()
        seg = seg.squeeze()
        if debug:
            print(f" whiteonblack: {whiteonblack} ||\t DONE, shape = ", seg.shape)
        return seg
    except Exception as e:
        print(e, traceback.format_exc())


def saveproperty(stack, filepath=None, type="pickle"):
    """

    :param filepath:
    :param type:
    :return:
    """
    success = False
    try:
        if type == "npz":
            # filepath = filepath + ".npz"
            np.savez(filepath, stack)
        elif type == "pickle":
            f = open(f"{filepath}.pkl", "wb")
            pickle.dump(stack, f)
        elif type == "json":
            f = open(f"{filepath}.pkl", "w")
            json.dump(stack, f)
        elif type == "csv":
            pd.DataFrame(stack).to_csv(f"{filepath}.csv")
    except Exception as e:
        print("exception: ", e)
    return success


def loadproperty(fpath):
    loadedarray = None
    if fpath.endswith(".npz"):
        loadedfile = np.load(fpath)
        loadedarray = loadedfile[loadedfile.files[0]]
    return loadedarray

def checksavedfileintegrity(loadedstackdata, savestackdata, ftype="npz"):
    loaded_equals_saved = False
    if ftype == "npz":
        # loadedstackdata = loaded[loaded.files[0]]
        loaded_equals_saved = datautils.array_nan_equal(loadedstackdata, savestackdata)
    return loaded_equals_saved

def convertfromnpz(npzpath, targetdir = None, totype = "csv"):
    from src.AnalysisTools import experimentalparams as ep
    import os
    # dims = (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells) # depends on organelle

    isconverted = False
    if totype == "csv":
        GFPchannel, organelletype, propertyname, strsigma = npzpath[:-4].split("/")[-1].split("_")

        loadedstackdata = loadproperty(npzpath)
        array3d = loadedstackdata.reshape((ep.USEDTREATMENTS, ep.USEDWEEKS,-1))
        array3ddf = datautils.generatedataframe(array3d, propertyname)
        csvname = os.path.join(targetdir,"".join([npzpath[:-4], ".csv"]))
        check = array3ddf.to_csv(csvname)
        if check is None:
            isconverted = True

    return isconverted

if __name__ == "__main__":
    import os
    from os.path import join, isfile
    convertfromdir = 'C:/Users/satheps/PycharmProjects/Results/2022/Feb18/TOM/results_all/npz/'
    targetdir = 'C:/Users/satheps/PycharmProjects/Results/2022/Feb25/TOM/csv_feb18/'
    files = os.listdir(convertfromdir)

    datafiles = [f for f in files if isfile(join(convertfromdir, f)) if f.__contains__('.npz')]
    for datafile in datafiles:
        filepath = join(convertfromdir,datafile)
        convertfromnpz(npzpath=filepath, targetdir= targetdir)
    # n1, n2, n3, n4, n5 = 12, 42, 15, 10, 3
    # exshape = (n1,n2,n3,n4,n5)
    # testmat = np.arange(n1*n2*n3*n4*n5).reshape(exshape)
    # fpath = "C:/Users/satheps/PycharmProjects/Results/2022/Jan21/savetest/testmat.npz"
    #
    # saveproperty(testmat, filepath=fpath, type = "npz")
    # loaded = loadproperty(fpath)
    # # print(loaded.shape)
    # # print(loaded == testmat)
    # print((not False in (loaded['arr_0']==testmat)))
    # print(loaded.files, loaded[loaded.files[0]].shape )
