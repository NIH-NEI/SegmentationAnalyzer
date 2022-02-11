import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.AnalysisTools import datautils, experimentalparams, types
from src.AnalysisTools import statcalcs

plt.rcParams["figure.figsize"] = [12, 9]
a4_dims = (11.7, 8.27)

def returnlogbounds(stack):
    minval = np.nanmin(stack)
    maxval = np.nanmax(stack)
    if np.isnan(minval):
        minval = 0
    else:
        minval = np.floor(minval)
    if np.isnan(maxval):
        maxval = 1
    else:
        maxval = np.ceil(maxval)
    return int(minval), int(maxval)

def violinstripplot(stackdata, channel="Cell", propname="", units="",
                    savesigma=None, selected_method_type=None,
                    treatment_types=experimentalparams.TREATMENT_TYPES, savepath="",
                    withstrpplt=True, scaletype="count", uselog=False):
       """
           scaletype can be count, width or area

           # :param individualdata: stackdata with outliers removed.
           :param stackdata: ndarray, has dimensions ((usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxorganellepercell))
           :param channel:
           :param propname:
           :param units:
           :param savesigma:
           :param selected_method_type:method type can be "Individual", "Stackwise", "platewise"
           :param treatment_types:
           :param savepath:
           :param withstrpplt:
           :return:
           """
       method_types = ["Individual", "Stackwise", "Platewise"]
       useall = False
       if selected_method_type == None:
           useall = True
       else:
           assert selected_method_type in method_types

       try:
           alpha_indiv = max(4000/np.count_nonzero(~np.isnan(stackdata)), 0.01)
       except Exception as e:
           alpha_indiv = 0.5
       print(f"individual alpha set to: {alpha_indiv}")
       print(np.count_nonzero(~np.isnan(stackdata)))
       stackdata_rmoutlier_indiv = statcalcs.removestackoutliers(stackdata, m=2, abstraction=0)
       stackdata_rmoutlier_stack = statcalcs.removestackoutliers(stackdata, m=2, abstraction=1)
       stackdata_rmoutlier_well = statcalcs.removestackoutliers(stackdata, m=2, abstraction=2)
       print(np.count_nonzero(~np.isnan(stackdata)), np.count_nonzero(~np.isnan(stackdata_rmoutlier_indiv)),
             np.count_nonzero(~np.isnan(stackdata_rmoutlier_stack)), np.count_nonzero(~np.isnan(stackdata_rmoutlier_well)))

       if uselog:
           stackdata_rmoutlier_indiv = np.log10(stackdata_rmoutlier_indiv)
           stackdata_rmoutlier_stack = np.log10(stackdata_rmoutlier_stack)
           stackdata_rmoutlier_well = np.log10(stackdata_rmoutlier_well)
           # for log range : multiply

           indivlogmin, indivlogmax = returnlogbounds(stackdata_rmoutlier_indiv)
           stacklogmin, stacklogmax = returnlogbounds(stackdata_rmoutlier_stack)
           welllogmin, welllogmax = returnlogbounds(stackdata_rmoutlier_well)

           print("in uselog")
           alllogstacks = [indivlogmin, indivlogmax, stacklogmin, stacklogmax, welllogmin, welllogmax]
           alllogstacknames = ["indivlogmin", "indivlogmax", "stacklogmin", "stacklogmax", "welllogmin", "welllogmax"]
           print("before",indivlogmin, indivlogmax, stacklogmin, stacklogmax, welllogmin, welllogmax)

           print("after:", indivlogmin, indivlogmax, stacklogmin, stacklogmax, welllogmin, welllogmax)
           # logmins = [indivlogmin-1, stacklogmin-1, welllogmin-1]
           logmins = [indivlogmin, stacklogmin,welllogmin]
           # logmaxes = [indivlogmax+1, stacklogmax+1, welllogmax+1]
           logmaxes = [indivlogmax, stacklogmax,welllogmax]
           print(logmins,"\n", logmaxes)
       index_indiv = datautils.generateindexedstack(stackdata_rmoutlier_indiv, propname)
       index_stack = datautils.generateindexedstack(stackdata_rmoutlier_stack, propname)
       index_well = datautils.generateindexedstack(stackdata_rmoutlier_well, propname)

       data = [index_indiv, index_stack, index_well]
       alphas = [alpha_indiv, 0.75, 1]
       violinwidths = [0.8, 0.8, 0.8]
       boxwidths = [0.8, 0.8, 0.8]
       unitstext = ""
       if (units is not None) and (units !=""):
           unitstext = f"(in {units})"
       # return
       for m, method in enumerate(method_types):
           print(method, selected_method_type, useall)
           if (selected_method_type == method) or useall:
               print(f"Plotting {method}")
               fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))  # , sharey=True)
               # inner = box for single boxplot
               vp = sns.violinplot(ax=axs, x="Week", y=propname, hue="Treatment", cut=0, data=data[m], inner=None,
                                   gridsize=100, palette="turbo", split=True, scale=scaletype,
                                   zorder=0, width=violinwidths[m])
               sns.stripplot(ax=axs, x="Week", y=propname, hue="Treatment", jitter=0.2, alpha=alphas[m], data=data[m],
                             dodge=True, edgecolor='black', zorder=1)
               sns.boxplot(ax=axs, x="Week", y=propname, data=data[m], width=boxwidths[m],
                           boxprops={'facecolor': 'None', "zorder": 10},
                           whiskerprops={"zorder": 10}, hue="Treatment", showfliers=False, dodge=1.1)
               handles, labels = vp.get_legend_handles_labels()
               axs.legend(handles[:0], labels[:0])
               axs.set_title(f"{channel}  {propname} ({method})", fontsize=24)

               l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

               # for ax in axs:
               axs.yaxis.grid(True)
               axs.set_xlabel('Weeks', fontsize=18)
               if uselog:
                   from matplotlib import ticker as mticker
                   axs.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.2f}}}$"))
                   axs.yaxis.set_ticks([np.log10(x) for p in range(logmins[m], logmaxes[m]) for x in np.linspace(10**p, 10**(p+1), 10)], minor=True)

               axs.set_ylabel(f"{channel} {propname}{unitstext})", fontsize=18)
               plt.setp(axs, xticks=[week for week in range(experimentalparams.USEDWEEKS)],
                        xticklabels=experimentalparams.WS[:experimentalparams.USEDWEEKS])
               plt.savefig(f"{savepath}_{channel}_{propname}_{method}_weeks{experimentalparams.USEDWEEKS}_{'withstrpplt' if withstrpplt else ''}{'_log' if uselog else ''}{'_s-' + str(savesigma) if savesigma else ''}.png")
               plt.close()
               plt.clf()


if __name__ == "__main__":
    """
    Example of violinstripplot
    """

    teststack = np.random.random((2, 4, 1, 5, 6, 1000)) * 100
    print(teststack.shape, teststack.ndim)
    if teststack.ndim == 6:
        teststack = np.expand_dims(teststack, axis=-1)

    print(teststack.shape, teststack.ndim)
    violinstripplot(stackdata=teststack, channel="example channel", propname="some property",
                            units="", savepath="", savesigma="99", uselog=False)
