import numpy as np
import os
from os.path import isfile, join
from concurrent.futures import ProcessPoolExecutor

# filename = f"{otype}_{propnames[i]}_{strsigma}.npz"

# calcfolder = '../Results/2022/Jan21/TOM_stack_18img/segmented/calcs/'
# savefolder = '../Results/2022/Jan21/TOM_stack_18img/segmented/plots/'
# calcfolder = '../Results/2022/Jan28/TOM/testset_thr/'
calcfolder = '../Results/2022/Jan28/TOM/TOM_calcs_test/testset_fixednan/'

savefolder = '../Results/2022/Jan28/TOM/testset_thr/plots/'
from src.Visualization import plotter
files = os.listdir(calcfolder)
# print(files)
# [f for f in listdir(folder) if isfile(join(folder, f)) if f.__contains__(s)]
datafiles = [f for f in files if isfile(join(calcfolder,f))if f.__contains__('.npz')]
print(datafiles)
propnames = ["volume", "xspan", "yspan", "zspan", "miparea", "max feret", "min feret"]
unittype = ["cu. microns", "microns", "microns", "microns", "sq. microns", "microns", "microns"]

num_processes = 8
executor = ProcessPoolExecutor(num_processes)
processes = []

for datafile in reversed(datafiles):
    try:
        organelletype, propertyname, strsigma = datafile[:-4].split("_")
        print(datafile, datafiles.index(datafile), organelletype, propertyname, strsigma)
        loadedfile = np.load(join(calcfolder,datafile))
        stackdata = loadedfile[loadedfile.files[0]]
        print(stackdata.shape, len(np.unique(stackdata)), np.count_nonzero,  np.count_nonzero(~np.isnan(stackdata)))
        # print(stackdata)
        if organelletype == "TOM20":
            uselog = False
            # print("using log")
        else:
            uselog = False
            # continue
        # plotter.violinstripplot(stackdata=stackdata, channel=organelletype, propname=propertyname, units=unittype[propnames.index(propertyname)],
        #                         savepath= savefolder, savesigma=strsigma, uselog=uselog)#, selected_method_type="Stackwise")
    except Exception as e:
        print(e)
#     processes.append((executor.submit(plotter.violinstripplot, stackdata=stackdata, channel=organelletype, propname=propertyname, units=unittype[propnames.index(propertyname)],
#                             savepath= savefolder, savesigma=True)))
#
# for process in processes:
#     print('in processes')
#     process.result()
