import numpy as np
import os
from os.path import isfile, join
from concurrent.futures import ProcessPoolExecutor
import traceback
from src.AnalysisTools import experimentalparams as ep
# filename = f"{otype}_{propnames[i]}_{strsigma}.npz"

# calcfolder = '../Results/2022/Jan21/TOM_stack_18img/segmented/calcs/'
# savefolder = '../Results/2022/Jan21/TOM_stack_18img/segmented/plots/'
# calcfolder = '../Results/2022/Jan28/TOM/testset_thr/'
# calcfolder =  '../Results/2022/Feb4/TOM/all/'

from src.Visualization import plotter


def plotRPEproperties(stackdata, savefolder,  organelletype, propertyname, strsigma, logplot):
    properties3d = ["centroid", "orientations"]
    centroiddims = ["Z", "X", "Y"]
    orientationdims = ["r", "\u03B8 (theta)", "\u03C6 (phi)"]
    centroidunits = ["microns", "microns", "microns"]
    orientationunits = ["microns", "degrees", "degrees"]
    propertybydim = [centroiddims, orientationdims]
    unitsbydim = [centroidunits, orientationunits]
    # print("propertyname:::", propertyname)
    units = ep.getunits(propertyname)
    if logplot == True:
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
        plotter.violinstripplot(stackdata=stackdata, channel=organelletype, propname=propertyname,
                                units=units
                                , savepath=savefolder, savesigma=strsigma, uselog=logplot)

def loadandplot(calcfolder = '../Results/2022/Feb4/TOM/newpropstest_add1/', savefolder =   '../Results/2022/Feb4/TOM/newpropstest_add1/plots/',  sigma = None):
    calcfolder = '../Results/2022/Feb11/TOM/results_test1/'
    savefolder = '../Results/2022/Feb11/TOM/results_test1/plots/'

    files = os.listdir(calcfolder)
    datafiles = [f for f in files if isfile(join(calcfolder,f))if f.__contains__('.npz')]
    print(datafiles)

    num_processes = 8
    executor = ProcessPoolExecutor(num_processes)
    processes = []


    # for datafile in reversed(datafiles):
    try:
        for datafile in datafiles:
            GFPchannel, organelletype, propertyname, strsigma = datafile[:-4].split("_")
            loadedfile = np.load(join(calcfolder,datafile))
            stackdata = loadedfile[loadedfile.files[0]]
            # print(datafile)
            print(GFPchannel, organelletype, propertyname, strsigma)
            print(stackdata.shape, len(np.unique(stackdata)),  np.count_nonzero(~np.isnan(stackdata)))
            print("zeros: ",np.count_nonzero(stackdata == 0), end=" |")
            print("negatives: ",np.count_nonzero(stackdata < 0), end=" |")
            print("positives: ",np.count_nonzero(stackdata > 0), end=" |")
            print("np.nan: ",len(stackdata[np.isnan(stackdata)]))
            for uselog in [True, False]:
                # executor.submit(ShapeMetrics.individualcalcs, GFPObjects)
                processes.append(executor.submit(plotRPEproperties, stackdata, savefolder,  organelletype, propertyname, strsigma, uselog))
        for process in processes:
            process.result()
    except Exception as e:
        print(e,  traceback.format_exc())

if __name__=="__main__":
    print("in main")
    loadandplot()

#
# for process in processes:
#     print('in processes')
#     process.result()

# import os
# import sys
# import warnings
# def loadandplot(argv):
#     if len(argv) > 1 and argv[1].startswith('@'):
#         try:
#             arglist = parse_resp_file(sys.argv[1][1:])
#         except Exception as ex:
#             print('Error parsing response file:', str(ex))
#             sys.exit(1)
#     else:
#         arglist = argv[1:]
#
#     import argparse
#     DEFAULT_DIR
#     parser = argparse.ArgumentParser(
#         description='Make Mask R-CNN predictions on DNA or Actin channel of RPE stacks.')
#     parser.add_argument('-d', '--data-dir', required=False,
#                         metavar="/data/directory",
#                         default=DEFAULT_DIR,
#                         help="Directory path for location of npz files")
#     parser.add_argument('rpefile', nargs='+',
#                         help='RPE Meta File .rpe.json (or directory containing .rpe.json)')
#
#     args = parser.parse_args(arglist)
#     args.data_dir = os.path.abspath(os.path.join(DEFAULT_DATA_DIR, args.data_dir))
#
#     assert args.channel in ('all', 'DNA', 'Actin'), 'Channel must be "DNA", "Actin" or "all".'
#
