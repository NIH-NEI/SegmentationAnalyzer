import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from src.stackio import stackio
import seaborn as sns
from src.AnalysisTools import statcalcs, datautils

savepath_tmm20 = "D:/WORK/NIH_new_work/Dominik/results/"
flist_plt = [f for f in os.listdir(savepath_tmm20) if f.__contains__('.npz')]
print(flist_plt)
line_names = sorted(['1085A1', '1085A2', '1097F1', '48B1', '48B2', 'BBS10B1', 'BBS16B2', 'D3C', 'LCA5A1', 'LCA5A2',
                     'LCA5B2', 'TJP11'])
transwells = ['1', '2']
FOVs = ['1', '2']
maxcells = 150
max_org_per_cell = 250
sigma = 2
for f in flist_plt:
    fpath = os.path.join(savepath_tmm20, f)
    organelle, propertyname, _ = re.split(r'[_.]', f)
    stackdata = stackio.loadproperty(fpath)
    print("before", stackdata.shape, flush=True)
    stackdata = statcalcs.removestackoutliers(stackdata, m=sigma, abstraction=0)
    print("after", stackdata.shape, flush=True)

    dims = stackdata.ndim
    # stackdata_indiv = statcalcs.stackbyabstractionlevel(stackdata, abstraction=0, fixeddims=4)
    labelids = np.arange(stackdata.shape[-1])  # this may be different for each organelle
    names = ["Line name", "transwell no.", "FOV no.", "cell id", "organelle id"][:dims]  #
    namevalues = [line_names, transwells, FOVs, list(np.arange(maxcells)), labelids][:dims]

    index = pd.MultiIndex.from_product(namevalues, names=names)
    # print("INDEX\n",index)
    indexedstack = pd.DataFrame({propertyname: stackdata.flatten()}, index=index).reset_index()
    # print(indexedstack)
    # sns.stripplot(stackdata=indexedstack, channel=organelle, propname=propertyname, savepath=savepath_tmm20)
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))  # , sharey=True)
    sns.stripplot(x="Line name", y=propertyname, hue="transwell no.", jitter=0.2, alpha=1, data=indexedstack,
                  dodge=True, linewidth=1, edgecolor='black', zorder=1)
    if sigma is not None:
        plt.savefig(f"{savepath_tmm20}{organelle}_{propertyname}_{sigma}.png")
    else:
        plt.savefig(f"{savepath_tmm20}{organelle}_{propertyname}.png")
    plt.close()
    plt.clf()
