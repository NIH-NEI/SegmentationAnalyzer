import json
import pickle
import traceback

import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from skimage.measure import label as skilbl
# from aicsimageio.writers import OmeTiffWriter
from tifffile import imread

from src.AnalysisTools import datautils
from src.AnalysisTools import types
from src.stackio import labelledcsvhandler


def opensegmentedstack(name: types.PathLike, whiteonblack: types.SegmentationLike = "default", debug: bool = False):
    """
    Handles simple binary segmentations as well as 3 color segmentations used in project.
    opens segmented stacks. whiteonblack

    :param name:
    :param whiteonblack: default indicates if image is whiteonblack or blackonwhite. The defaults may be different for
     another dataset. Can be True, False or "default"
    :param debug: outputs additional information
    :return:
    """

    if name.endswith("csv"):
        labelseg = labelledcsvhandler.csvtoids(name)  # assume default dimensions
        return labelseg
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

def getlabelledstack(img, debug= False):
    """
    If stack has only two values (foreground and background), attempts to generate instance labels and returns labelled
    image. Do not put this in recursion.

    :param img: Segmented or already labelled image.
    :param debug: outputs additional information.
    :return: labelled image.
    """
    lblimg, objcount = None, None
    if len(np.unique(img)) <= 2:
        lblimg, _ = skilbl(img, return_num=True)
        if debug:
            print(len(np.unique(img)))
    else:
        lblimg = img.copy()
    return lblimg

def read_get_labelledstacks(Actinfilepath, DNAfilepath, debug=False):
    """
    Open segmented files for coregistered channels and returns labelled stacks.

    :param Actinfilepath:
    :param DNAfilepath:
    :param debug:
    :return:
    """

    img_ACTIN = opensegmentedstack(Actinfilepath)  # binary=False
    img_DNA = opensegmentedstack(DNAfilepath)  # binary=False
    labelactin = getlabelledstack(img_ACTIN, debug=debug)
    labeldna = getlabelledstack(img_DNA, debug=debug)
    # print("TEST: all unique values should be equal", np.unique(img_GFP), np.unique(img_ACTIN), np.unique(img_DNA))
    return labelactin, labeldna


def saveproperty(stack, filepath=None, type="npz"):
    """

    :param stack:
    :param filepath:
    :param type:
    :return:
    """
    success = False
    try:
        if type == "npz":
            # filepath = filepath + ".npz"
            np.savez(filepath, stack)
            success = True
        elif type == "pickle":
            f = open(f"{filepath}.pkl", "wb")
            pickle.dump(stack, f)
            success = True
        elif type == "json":
            f = open(f"{filepath}.pkl", "w")
            json.dump(stack, f)
            success = True
        elif type == "csv":
            pd.DataFrame(stack).to_csv(f"{filepath}.csv")
            success = True
    except Exception as e:
        print("exception: ", e)
    return success


def loadproperty(fpath):
    """
    Loads property

    :param fpath:
    :return:
    """
    loadedarray = None
    if fpath.endswith(".npz"):
        loadedfile = np.load(fpath)
        loadedarray = loadedfile[loadedfile.files[0]]
    return loadedarray


def checksavedfileintegrity(loadedstackdata, savestackdata, ftype="npz"):
    """

    :param loadedstackdata:
    :param savestackdata:
    :param ftype:
    :return:
    """
    loaded_equals_saved = False
    if ftype == "npz":
        # loadedstackdata = loaded[loaded.files[0]]
        loaded_equals_saved = datautils.array_nan_equal(loadedstackdata, savestackdata)
    return loaded_equals_saved


def convertfromnpz(npzpath, targetdir=None, totype="csv", save=True):
    """

    :param npzpath:
    :param targetdir:
    :param totype:
    :param save: save a file
    :return:
    """
    from src.AnalysisTools import experimentalparams as ep
    import os
    # dims = (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells) # depends on organelle

    isconverted = False
    if totype == "csv":
        print(f"saving as {totype}")
        GFPchannel, organelletype, propertyname, strsigma = npzpath[:-4].split("/")[-1].split("_")

        loadedstackdata = loadproperty(npzpath)
        array3d = loadedstackdata.reshape((ep.USEDTREATMENTS, ep.USEDWEEKS, -1))
        array3ddf = datautils.generatedataframe(array3d, propertyname)
        if save:
            csvfilename = f"{GFPchannel}_{organelletype}_{propertyname}.csv"
            csvpath = os.path.join(targetdir, csvfilename)
            check = array3ddf.to_csv(csvpath)
        else:
            return array3ddf  # TODO : Support for merging various properties in one file
        if check is None:
            isconverted = True

    return isconverted


def convertfromnpz_allproperties(npzfolderpath, targetdir=None, totype="csv", organelle="Cell", save=True):
    """

    :param npzfolderpath:
    :param targetdir:
    :param totype:
    :param organelle:
    :param save:
    :return:
    """
    import os
    datafiles = [join(npzfolderpath, f) for f in files if isfile(join(npzfolderpath, f)) if f.__contains__('.npz')]

    # dims = (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells) # depends on organelle
    isconverted = False
    array3ddfs = None
    firstloop = True
    if totype == "csv":
        for datafile in datafiles:
            GFPchannel, organelletype, propertyname, strsigma = datafile[:-4].split("/")[-1].split("_")
            if organelletype == organelle:
                print(datafile, ":::", GFPchannel, organelletype, propertyname, strsigma)
                loadedstackdata = loadproperty(datafile)
                array3ddf = datautils.generateindexeddataframe(loadedstackdata, propertyname)
                # print(array3ddf)
                if firstloop:
                    array3ddfs = array3ddf.copy()
                    firstloop = False
                else:
                    newpropcol = list(array3ddf.columns)[-1]
                    print(newpropcol)
                    if newpropcol == "Centroid":  # ignore centroid for now
                        continue
                    else:
                        array3ddfs[newpropcol] = array3ddf[newpropcol]
                # print(type(list(array3ddf.columns)[-1]))
        # print(array3ddfs)
        # exit()
        if save:
            print(f"saving as {totype}")
            csvfilename = f"{GFPchannel}_{organelle}.csv"
            csvpath = os.path.join(targetdir, csvfilename)
            check = array3ddfs.to_csv(csvpath)
        else:
            return array3ddfs  # TODO : Support for merging various properties in one file
        if check is None:
            isconverted = True

    return isconverted


if __name__ == "__main__":

    import os
    from os.path import join, isfile

    convertfromdir = 'C:/Users/satheps/PycharmProjects/Results/2022/Feb18/TOM/results_all/npz/'
    targetdir = 'C:/Users/satheps/PycharmProjects/Results/2022/Mar18/combinecsv/'
    files = os.listdir(convertfromdir)
    #####################################################################################
    convertfromnpz_allproperties(npzfolderpath=convertfromdir, targetdir=targetdir)
    #####################################################################################
    # datafiles = [f for f in files if isfile(join(convertfromdir, f)) if f.__contains__('.npz')]
    # for datafile in datafiles:
    #     filepath = join(convertfromdir,datafile)
    #     a = convertfromnpz(npzpath=filepath, targetdir= targetdir)
    #     print(a)
    # #     exit(0)
    #####################################################################################
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
    #####################################################################################
