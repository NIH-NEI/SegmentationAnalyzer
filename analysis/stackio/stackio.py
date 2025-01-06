import json
import pickle
import time
import traceback
from os.path import join, isfile
import pandas as pd
from aicsimageio import AICSImage
from skimage.measure import label as skilbl
# from aicsimageio.writers import OmeTiffWriter
from tifffile import imread
from analysis.AnalysisTools import datautils
from analysis.AnalysisTools.dtypes import *
from analysis.stackio import labelledcsvhandler


def opensegmentedstack(name: PathLike, whiteonblack: SegmentationLike = "default", debug: bool = False):
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
            if whiteonblack == True:
                seg = 1 * (seg.data > 0)
            else:
                seg = (1 - 1 * (seg.data > 0))
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


def getlabelledstack(img, debug=False):
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


def read_get_segmented_stacks(Actinfilepath, DNAfilepath, debug=False):
    """
    Open segmented files for coregistered channels and returns segmented stacks.

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
    Converts npz files to csv
    :param npzpath: path to npz file
    :param targetdir: target directory for saving converted files
    :param totype: file format of file to be saved (currently limited to csv)
    :param save: save a file
    :return:
    """
    from analysis.AnalysisTools import experimentalparams as ep
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
            return array3ddf
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
    files = os.listdir(npzfolderpath)
    datafiles = [join(npzfolderpath, f) for f in files if isfile(join(npzfolderpath, f)) if f.__contains__('.npz')]
    # dims = (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells) # depends on organelle
    isconverted = False
    array3ddfs = None
    firstloop = True
    if totype == "csv":
        try:
            for datafile in datafiles:
                try:
                    GFPchannel, organelletype, propertyname, strsigma = os.path.basename(datafile[:-4]).split("_")
                except:
                    try:
                        GFPchannel, organelletype, propertyname = os.path.basename(datafile[:-4]).split("_")
                    except:
                        print(f"Could not resolve file name: {datafile[:-4]}")
                if propertyname == "Mean Volume" or propertyname == "Count per cell" or propertyname.__contains__(
                        "distance to wall") or propertyname.__contains__("z-distance d"):
                    organelletype = "Cell"  # TEMP use in cell data
                proptypes = [None]
                usepropname = propertyname
                if propertyname == "Centroid":
                    proptypes = ["X", "Y", "Z"]
                elif propertyname == "Orientation":
                    proptypes = ["r", "\u03B8 (theta)", "\u03C6 (phi)"]
                    continue  # dont need for calculations
                if organelletype == organelle:
                    # if propertyname == "Centroid" or propertyname == "Orientation":
                    #     # continue
                    print(datafile, organelle, flush=True)

                    loadedstackdata = loadproperty(datafile)
                    if propertyname == "Mean Volume":  # TEMP
                        loadedstackdata = loadedstackdata.mean(axis=-1)
                    # print("DONE:", datafile, proptypes)

                    for proptype in proptypes:
                        if proptype is not None:
                            usepropname = f"{propertyname}_{proptype}"
                            # print(loadedstackdata.shape())
                            # exit()
                            usestackdata = loadedstackdata[..., proptypes.index(proptype)]
                        else:
                            usestackdata = loadedstackdata
                        print(datafile, ":::", GFPchannel, organelletype, usepropname, proptype,
                              usestackdata.shape, loadedstackdata.shape)
                        array3ddf = datautils.generateindexeddataframe(usestackdata, usepropname)

                        if firstloop:
                            array3ddfs = array3ddf.copy().dropna(axis='index')
                            firstloop = False
                        else:
                            newpropcol = list(array3ddf.columns)[-1]
                            # print(newpropcol)
                            # if newpropcol == "Centroid":  # ignore centroid for now
                            #     continue
                            # else:
                            array3ddfs[newpropcol] = array3ddf[newpropcol].dropna(axis='index')
                    del loadedstackdata
                    del usestackdata
                    del array3ddf
            if save:
                print(f"saving as {totype}")
                csvfilename = f"{GFPchannel}_{organelle}.csv"
                if not os.path.exists(targetdir):
                    os.mkdir(targetdir)
                csvpath = os.path.join(targetdir, csvfilename)
                print(array3ddfs.shape)
                # array3ddfs_na = array3ddfs.dropna().reset_index(drop=True)
                # print(array3ddfs_na.shape)
                # del array3ddfs
                check = array3ddfs.to_csv(csvpath)
            else:
                return array3ddfs
            if check is None:
                isconverted = True
        except Exception as ex_csv:
            print(f"Exception encountered in convertfromnpz {ex_csv} @ {datafile}")
            print(traceback.format_exc())

    return isconverted


if __name__ == "__main__":
    import os, sys, re


    # savepath_tmm20 = "../results/"
    # subdirs = os.listdir(savepath_tmm20)
    # flist_plt = [f for f in os.listdir(savepath_tmm20) if f.__contains__('.npz')]
    # print(flist_plt)
    # line_names = sorted(['1085A1', '1085A2', '1097F1', '48B1', '48B2', 'BBS10B1', 'BBS16B2', 'D3C', 'LCA5A1', 'LCA5A2',
    #                      'LCA5B2', 'TJP11'])
    # transwells = ['1', '2']
    # FOVs = ['1', '2']
    # maxcells = 150
    # max_org_per_cell = 250
    # sigma = 2
    # organelles = ["Cell", "TOM"]
    # for f in flist_plt:
    #     fpath = os.path.join(savepath_tmm20, f)
    #     tpath = os.path.join(savepath_tmm20, f.replace('.npz','.csv'))
    #     organelle, propertyname, _ = re.split(r'[_.]', f)
    #     # stackdata = loadproperty(fpath)
    #     loadedstackdata = loadproperty(fpath)
    #     dims = loadedstackdata.ndim
    #     labelids = np.arange(loadedstackdata.shape[-1])  # this may be different for each organelle
    #     names = ["Line name", "transwell no.", "FOV no.", "cell id", "organelle id"][:dims]  #
    #     namevalues = [line_names, transwells, FOVs, list(np.arange(maxcells)), labelids][:dims]
    #
    #     index = pd.MultiIndex.from_product(namevalues, names=names)
    #     # print("INDEX\n",index)
    #     indexedstack = pd.DataFrame({propertyname: loadedstackdata.flatten()}, index=index).reset_index()
    #     csvfilename = f"{organelle}_{propertyname}.csv"
    #     csvpath = os.path.join(savepath_tmm20, csvfilename)
    #     check = indexedstack.dropna().to_csv(csvpath)
    """f
    convertfromdir = '..data/../TOM/results_all/npz/'
    targetdir = '..data/combinecsv/'
    
    #####################################################################################
    convertfromnpz_allproperties(npzfolderpath=convertfromdir, targetdir=targetdir)
    #####################################################################################
    """
    # LOOP t
    dirlist = '../CTNNB1_20230522/GJA/'
    # targetdir = 'D:/WORK/NIH_new_work/Final_calculations/'
    subdirs = os.listdir(dirlist)
    print(subdirs)
    # subdirs = [dirr for dirr in subdirs if os.path.isdir(os.path.abspath(dirr))]
    print(os.path.abspath(subdirs[0]))
    for subdir in subdirs:  # subdirectory for all npz files

        organelles = ["Cell", "DNA", subdir]
        # organelles = ["Cell"]
        convertfromdir = os.path.join(dirlist, subdir, "calcs/")
        targetdir = os.path.join(dirlist, subdir, "csvs/")
        if not os.path.exists(targetdir):
            os.mkdir(targetdir)
        for organelle in organelles:
            try:
                if organelle == "MYH":  # To address discrepancy in directory name for organelle
                    organelle = "MYH1"
                convertfromnpz_allproperties(npzfolderpath=convertfromdir, targetdir=targetdir, organelle=organelle)
            except Exception as ee:
                tb = sys.exc_info()[2]
                ee.with_traceback(tb)
                print("MISSED:", ee, convertfromdir, subdir, organelle)
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
    # fpath = "..data/../savetest/testmat.npz"
    #
    # saveproperty(testmat, filepath=fpath, type = "npz")
    # loaded = loadproperty(fpath)
    # # print(loaded.shape)
    # # print(loaded == testmat)
    # print((not False in (loaded['arr_0']==testmat)))
    # print(loaded.files, loaded[loaded.files[0]].shape )
    #####################################################################################
