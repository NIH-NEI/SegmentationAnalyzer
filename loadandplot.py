import numpy as np
import os
from os.path import isfile, join
from concurrent.futures import ProcessPoolExecutor

# filename = f"{otype}_{propnames[i]}_{strsigma}.npz"

calcfolder = '../Results/2022/Jan21/TOM_stack_18img/segmented/calcs/'
savefolder = '../Results/2022/Jan21/TOM_stack_18img/segmented/plots/'
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
    organelletype, propertyname, strsigma = datafile[:-4].split("_")
    print(datafile, datafiles.index(datafile), organelletype, propertyname, strsigma)
    loadedfile = np.load(join(calcfolder,datafile))
    stackdata = loadedfile[loadedfile.files[0]]
    # print(stackdata.shape, np.unique(stackdata))
    # print(stackdata)

    plotter.violinstripplot(stackdata=stackdata, channel=organelletype, propname=propertyname, units=unittype[propnames.index(propertyname)],
                            savepath= savefolder, savesigma=True)#, selected_method_type="Stackwise")

#     processes.append((executor.submit(plotter.violinstripplot, stackdata=stackdata, channel=organelletype, propname=propertyname, units=unittype[propnames.index(propertyname)],
#                             savepath= savefolder, savesigma=True)))
#
# for process in processes:
#     print('in processes')
#     process.result()
