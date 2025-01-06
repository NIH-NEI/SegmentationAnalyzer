import os

import numpy as np
import pandas
import tifffile


def csvtoids(csvname, shape=None, save=False, returnvals=True, debug=False):
    """
    Converts compressed csv segmentations to ndarray and saves and/or returns it
    Args:
        csvname: csv name
        shape: dimensions of stack
        save: Boolean
        returnvals: Returns values if True
        debug: verbosity for debugging

    Returns:
        array if returnvals set to True

    """
    if shape is None:
        shape = (27, 1076, 1276)
    csvpath = os.path.abspath(csvname)
    if debug:
        print(f'Input segmentation CSV: {csvpath}')
        print(f"csv to tif dimensions: {shape}")
    mdata = np.zeros(shape=shape, dtype=np.uint16)
    csvdf = pandas.read_csv(csvpath)
    for index, row in csvdf.iterrows():
        mdata[row["Frame"], row["y"], row["xL"]:row["xR"] + 1] = row["ID"]
    if save:
        basedir, fn = os.path.split(csvpath)
        bn, ext = os.path.splitext(fn)
        tifpath = os.path.join(basedir, bn + '_ids_py.tif')
        for ch in ('_DNA', '_Actin'):
            idx = bn.find(ch)
            if idx >= 0:
                bn = bn[:idx]
                break
        tifffile.imwrite(tifpath, mdata, compress=6, photometric='minisblack')
    if returnvals:
        return mdata


if __name__ == '__main__':
    csvname = ""
    # tifpath = "../Results/2022/Mar18/csvtest/sec61/exptif/"
    shape = (27, 1078, 1278)
    # args = [None, csvname, shape]

    # csvtoids(csvname, shape)
    path1 = "..data/../csvtest/sec61/P1-W1-SEC_G02_F001_Actin_RPE_ids_py.tif"
    path2 = "..data/../csvtest/sec61/P1-W1-SEC_G02_F001_Actin_RPE_ids.tif"
    tif1 = tifffile.imread(path1)
    tif2 = tifffile.imread(path2)
    tif1bw = tif1 > 0
    tif2bw = tif2 > 0
    mask = (tif1 == tif2)
    maskbw = (tif1bw == tif2bw)
    marr = np.ones_like(tif1) * 255
    print(tif1.shape, tif2.shape, mask.shape, marr.shape)
    marr[maskbw] = 0
    # marr[mask] = 0
    maskpath = "..data/../csvtest/sec61/P1-W1-SEC_G02_F001_Actin_RPE_eq.tif"
    tifffile.imwrite(maskpath, marr, compress=6, photometric='minisblack')
    debug = False
    if debug:
        print(np.unique(marr))
        print(np.unique(tif2))
        print(np.unique(tif1))
        print(np.unique(tif2))
        print(np.unique(tif1 & tif2))
