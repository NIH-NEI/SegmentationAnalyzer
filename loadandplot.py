import os
import traceback
from concurrent.futures import ProcessPoolExecutor
# import sys
# import warnings
# import argparse
from os.path import isfile, join

import numpy as np
from src.stackio import stackio
from src.AnalysisTools import experimentalparams as ep
from src.Visualization import plotter


# filename = f"{otype}_{propnames[i]}_{strsigma}.npz"
# calcfolder = '../Results/2022/Jan21/TOM_stack_18img/segmented/calcs/'
# savefolder = '../Results/2022/Jan21/TOM_stack_18img/segmented/plots/'
# calcfolder = '../Results/2022/Jan28/TOM/testset_thr/'
# calcfolder =  '../Results/2022/Feb4/TOM/all/'


def plotRPEproperties(stackdata, savefolder, organelletype, propertyname, strsigma, logplot):
    properties3d = ["Centroid", "Orientation"]
    centroiddims = ["Z", "X", "Y"]
    orientationdims = ["r", "\u03B8 (theta)", "\u03C6 (phi)"]
    centroidunits = ["microns", "microns", "microns"]
    orientationunits = ["microns", "degrees", "degrees"]
    propertybydim = [centroiddims, orientationdims]
    unitsbydim = [centroidunits, orientationunits]
    if logplot:
        print("using logarithmic plots")
    if propertyname in properties3d:
        propid = properties3d.index(propertyname)
        for pid, pdim in enumerate(propertybydim[propid]):
            dimpropertyname = f"{propertyname}_{pdim}"
            dimstackdata = stackdata[..., pid]
            units = f"{unitsbydim[pid]}"
            print(f"Plotting {propertyname} {pid}: {dimpropertyname}:{dimstackdata.shape}")
            plotter.violinstripplot(stackdata=dimstackdata, channel=organelletype, propname=dimpropertyname,
                                    units=units, savepath=savefolder, savesigma=strsigma, uselog=logplot)

    else:
        units = ep.getunits(propertyname)
        plotter.violinstripplot(stackdata=stackdata, channel=organelletype, propname=propertyname,
                                units=units, savepath=savefolder, savesigma=strsigma, uselog=logplot)


def loadandplot(calcfolder='../Results/2022/Feb4/TOM/newpropstest_add1/',
                savefolder='../Results/2022/Feb4/TOM/newpropstest_add1/plots/', sigma=None):
    # calcfolder = '../Results/2022/Feb11/TOM/results_test2/npz/'
    # calcfolder = '../Results/2022/Feb18/TOM/results_wsphericity/npz/'
    # savefolder = '../Results/2022/Feb18/TOM/results_wsphericity/plots/'
    calcfolder = '../Results/2022/Feb18/TOM/results_all/npz/'
    # savefolder = '../Results/2022/Feb18/TOM/results_all/plots/'
    # savefolder = '../Results/2022/Feb18/TOM/results_all/plots_withoutliers/'
    savefolder = '../Results/2022/Feb25/TOM/results_all/plots_chkstat/'

    files = os.listdir(calcfolder)
    datafiles = [f for f in files if isfile(join(calcfolder, f)) if f.__contains__('.npz')]
    print(datafiles)

    num_processes = 8
    executor = ProcessPoolExecutor(num_processes)
    processes = []

    # for datafile in reversed(datafiles):
    try:
        for datafile in datafiles:
            GFPchannel, organelletype, propertyname, strsigma = datafile[:-4].split("_")
            stackdata = stackio.loadproperty(join(calcfolder, datafile))
            # stackdata = loadedfile[loadedfile.files[0]]
            # print(datafile)
            # if not propertyname.__contains__("normalized"):
            #     continue
            print(GFPchannel, organelletype, propertyname, float(strsigma))
            print(stackdata.shape, len(np.unique(stackdata)), np.count_nonzero(~np.isnan(stackdata)))
            print("zeros: ", np.count_nonzero(stackdata == 0), end=" |")
            print("negatives: ", np.count_nonzero(stackdata < 0), end=" |")
            print("positives: ", np.count_nonzero(stackdata > 0), end=" |")
            print("np.nan: ", len(stackdata[np.isnan(stackdata)]))
            for uselog in [True, False]:
                # for uselog in [True]:
                # executor.submit(ShapeMetrics.calculate_multiorganelle_properties, GFPObjects)
                processes.append(executor.submit(plotRPEproperties, stackdata, savefolder, organelletype, propertyname,
                                                 float(strsigma), uselog))
        for process in processes:
            process.result()
    except Exception as e:
        print(e, traceback.format_exc())


if __name__ == "__main__":
    print("in main")
    loadandplot()
# add arguments based approach