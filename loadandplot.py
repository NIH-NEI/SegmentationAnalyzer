import os
import traceback
from concurrent.futures import ProcessPoolExecutor
# import warnings
# import argparse
from os.path import isfile, join
from src.AnalysisTools import statcalcs

import click
import numpy as np
from src.AnalysisTools import experimentalparams as ep
from src.Visualization import plotter
from src.stackio import stackio


def plotRPEproperties(stackdata, savefolder, organelletype, propertyname, percentile, logplot, vplot=True, pplot=False):
    try:
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
                units = f"{unitsbydim[propid][pid]}"
                print(f"Plotting {propertyname} {pid}: {dimpropertyname}:{dimstackdata.shape}")
                if vplot:
                    plotter.violinstripplot(stackdata=dimstackdata, channel=organelletype, propname=dimpropertyname,
                                            units=units, savepath=savefolder, percentile_include=percentile, uselog=logplot)
                if pplot:
                    plotter.stdboxplot(stackdata=dimstackdata, channel=organelletype, propname=dimpropertyname,
                                       units=units, savepath=savefolder, percentile_include=percentile, uselog=logplot)

        else:
            units = ep.getunits(propertyname)
            if vplot:
                plotter.violinstripplot(stackdata=stackdata, channel=organelletype, propname=propertyname,
                                        units=units, savepath=savefolder, percentile_include=percentile, uselog=logplot)
            if pplot:
                plotter.stdboxplot(stackdata=stackdata, channel=organelletype, propname=propertyname,
                                   units=units, savepath=savefolder, percentile_include=percentile, uselog=logplot)
    except Exception as e:
        print(f"{organelletype}, {propertyname}\nError:", e, traceback.format_exc())


@click.command(options_metavar="<options>")
@click.option("--calcfolder", default = "../Results/2022/Apr29/lc3b/calcs/",#default="../SegmentationAnalyzer/temp/",
              help="Folder containing calculated shape metrics", metavar="<PathLike>")
@click.option("--savefolder", default = "../Results/2022/June10/lc3b/plots/", metavar="<PathLike>",
              help="Folder where plots should be saved")
# @click.option("--sigma", default=None, metavar="positive float",
#               help="number of standard deviations from mean included in plots")
@click.option("--percentile", default=95.45 , metavar="positive float",
              help="number of standard deviations from mean included in plots")
# @click.option("--debug", default=False, metavar="<Boolean>", help="Show extra information for debugging")
def loadandplot(calcfolder, savefolder, percentile):
    print(calcfolder, savefolder, percentile)
    files = os.listdir(calcfolder)
    datafiles = [f for f in files if isfile(join(calcfolder, f)) if f.__contains__('.npz')]
    print(datafiles)

    num_processes = 8
    executor = ProcessPoolExecutor(num_processes)
    processes = []

    # for datafile in reversed(datafiles):
    try:
        for datafile in datafiles:
            GFPchannel, organelletype, propertyname, _ = datafile[:-4].split("_")
            # final value number is for percentile. This may need to be removed depending on the way files are named.
            stackdata = stackio.loadproperty(join(calcfolder, datafile))
            # print(datafile)
            # if not propertyname.__contains__("Volume"):
            #     continue
            # if not organelletype.__contains__("Cell"):
            #     continue
            print(GFPchannel, organelletype, propertyname, float(percentile))
            print(stackdata.shape, len(np.unique(stackdata)), np.count_nonzero(~np.isnan(stackdata)))
            print("zeros: ", np.count_nonzero(stackdata == 0), end=" |")
            print("negatives: ", np.count_nonzero(stackdata < 0), end=" |")
            print("positives: ", np.count_nonzero(stackdata > 0), end=" |")
            print("np.nan: ", len(stackdata[np.isnan(stackdata)]))
            for uselog in [True, False]:
                # for uselog in [True]:
                # executor.submit(ShapeMetrics.calculate_multiorganelle_properties, GFPObjects)
                processes.append(executor.submit(plotRPEproperties, stackdata, savefolder, organelletype, propertyname,
                                                 float(percentile), uselog))
        for process in processes:
            process.result()
    except Exception as e:
        print(e, traceback.format_exc())


if __name__ == "__main__":
    import sys

    args = sys.argv
    print(f"arguments:{args})")
    loadandplot()
