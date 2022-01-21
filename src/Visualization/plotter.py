import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.AnalysisTools import datautils, experimentalparams, types
from src.AnalysisTools import statcalcs
plt.rcParams["figure.figsize"] = [12, 9]
a4_dims = (11.7, 8.27)


def generate_plot(data3dlist, propname: str, units: str, savesigma: str = None,
                  savepath: types.PathLike = "", channel="", withstrpplt: bool = False) -> None:
    """

    :param data3dlist: input data. must be a 3d list - with dimensions treatment, week and ID
    :param propname: property name
    :param units: units (string) - e.g. micron
    :param savesigma: string indicating number of standard deviations
    :param savepath: path to saving plots
    :param channel: channel name
    :param withstrpplt: include strip plot with the violinplot if True
    :return:
    """
    datadf = datautils.generatedataframe(data3dlist, propname)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))
    sns.violinplot(ax=ax, x="Week", y=propname, hue="Treatment", cut=0, data=datadf, gridsize=100,
                   split=True,
                   scale="count", )

    ax.set_xlabel('weeks', fontsize=22)
    ax.set_ylabel(f"{propname}{units}", fontsize=18)

    plt.setp(ax, xticks=[y for y in range(experimentalparams.USEDWEEKS)],
             xticklabels=experimentalparams.WS[:experimentalparams.USEDWEEKS])
    plt.savefig(
        f"{savepath}_{channel}_{propname}_weeks{experimentalparams.USEDWEEKS}_{'withstrpplt' if withstrpplt else ''}{'_s-' + savesigma if savesigma else ''}.png")
    plt.close()
    plt.clf()

    # def generate_plot(data3dlist, propname, units, savesigma = None):
    #     fig, axs = plt.subplots(nrows=1, ncols=usedtreatments, figsize=(18, 8),sharey=True)
    #     widths = np.zeros((usedtreatments,usedweeks))
    # ########### normalizing for maximum widths
    # #     for i in range(usedweeks):
    # #         freq1,bin1 = np.histogram(data3dlist[0][i],100)
    # #         freq2,bins2 = np.histogram(data3dlist[1][i],100)
    # #         widths[0,i] = max(freq1)
    # #         widths[1,i] = max(freq2)
    # #     violinwidths = 0.6* (np.max(widths, axis=1)/np.amax(widths))

    # ########### normalizing for w1 widths:
    #     for i in range(usedweeks):
    #         freq1,bin1 = np.histogram(data3dlist[0][i],20)
    #         freq2,bins2 = np.histogram(data3dlist[1][i],20)
    #         widths[0,i] = max(freq1)
    #         widths[1,i] = max(freq2)
    #     maxwidths = (np.max(widths, axis=1)/np.amax(widths))
    #     inv_w1ratios =  np.flip(widths[:,0])/ np.amax(widths[:,0])
    #     violinwidths = maxwidths * inv_w1ratios
    #     print(widths, maxwidths, widths[:,0]/ np.amax(widths[:,0]), violinwidths, maxwidths*violinwidths)
    # #     return

    #     for t, treatment in enumerate(treatment_type):
    #             try:
    #                 wtdata = data3dlist[t].copy()
    #                 print(type(wtdata), len(wtdata),len(wtdata[0]),len(wtdata[1]),len(wtdata[2]),len(wtdata[3]))
    # #                 sns.violinplot(ax=axs[t],data= wtdata, width=violinwidths[t], scale="area", cut=0, inner="box",zorder=2)
    #                 sns.violinplot(ax=axs[t],data= wtdata, scale="width", cut=0, inner="box",zorder=2)

    # #                 axs[t].violinplot(wtdata, widths=violinwidths[t] ,showextrema = False)
    # #                 axs[t].boxplot(wtdata, widths = 0.15, showfliers = False)
    #                 axs[t].set_title(treatment)
    #             except Exception as e:
    #                 print(w,t, e, len(wtdata))
    #     for ax in axs:
    #         ax.yaxis.grid(True)
    #         ax.set_xlabel('weeks',fontsize = 22)
    #         ax.set_ylabel(f" {propname}{units}",fontsize = 22)
    #     plt.setp(axs, xticks=[y + 1  for y in range(len(wtdata))],
    #              xticklabels=ws[:usedweeks])
    # #     plt.show()
    #     plt.savefig(f"{savepath}{channel}_{propname}_weeks{usedweeks}_{'withstrpplt' if withstrpplt else ''}{'_s-'+savesigma if savesigma else ''}.png")
    #     plt.close
    #     plt.clf()
    #     del fig
    #     del axs

    # def boolean_indexing(v, fillval=np.nan):
    #     lens = np.array([len(item) for item in v])
    #     mask = lens[:,None] > np.arange(lens.max())
    #     print(lens,lens[:,None], len(l), mask)

    #     out = np.full(mask.shape,fillval)
    #     print(np.concatenate(v))
    #     out[mask] = np.concatenate(v)
    #     return out
    # l= [[1, 2, 3], [1, 2], [3, 8, 9, 7, 3]]
    # boolean_indexing(l)


# def violinstripplot(data, organelle="Cell", propname="", unit="", sigma=0, savesigma="", savepath="") -> None:
#     """
#     TODO: double check data types before finalizing
#     :param data:
#     :param organelle: Name of organelle
#     :param propname:
#     :param unit:
#     :param sigma:
#     :param savesigma:
#     :return:
#     """
#     try:
#         tempdata = data.copy()
#         weekly_data_conform = statcalcs.removeoutliers3dlist(tempdata, m=sigma)
#         inddata = datautils.generatedataframe(weekly_data_conform, propname)
#
#         violinstripplotind(inddata, stackdata, channel="Cell", propname=propname, units=f"(in {unit})", savesigma=savesigma, savepath=savepath)
#         # weekly_data_conform, propname=f"{organelle} {propname}", units=, savesigma=savesigma)
#         return 0
#     except Exception as e:
#         print(e)
#         return 1


def violinstripplot(stackdata, channel="Cell", propname="", units="",
                    savesigma=None, selected_method_type=None,
                    treatment_types=experimentalparams.TREATMENT_TYPES, savepath="",
                    withstrpplt=True, scaletype="count"):
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
    method_types = ["Individual",  "Stackwise","Platewise"]
    useall = False
    if selected_method_type == None:
        useall = True
    else:
        assert selected_method_type in method_types
    basendim = 7
    if stackdata.ndim == 6:
        basendim = 6
        stackdata = np.expand_dims(stackdata,axis=-1)
    assert (stackdata.ndim == 7)

    # inddatadf = datautils.generatedataframeind(individualdata, propname)
    # stackdatadf = datautils.generatedataframe(stackdata, propname)
    # stackdata_rmoutlier = statcalcs.removestackoutliers(stackdata)
    stackdata_rmoutlier = stackdata
    print(len(np.unique(stackdata)), len(np.unique(stackdata_rmoutlier)))

    # index_indiv = statcalcs.removestackoutliers(stackdata,m=2)
    # index_stack= statcalcs.removestackoutliers(index_stack0,m=2)
    # index_well= statcalcs.removestackoutliers(index_well0,m=2)
    index_indiv = datautils.generateindexedstack(stackdata_rmoutlier, propname, abstraction=0, basendim=basendim)
    index_stack = datautils.generateindexedstack(stackdata_rmoutlier, propname, abstraction=1, basendim=basendim)
    index_well = datautils.generateindexedstack(stackdata_rmoutlier, propname, abstraction=2, basendim=basendim)

    # index_indiv = index_indiv[~np.isnan(index_indiv)]  # Remove the NaNs
    # index_stack = index_stack[~np.isnan(index_stack)]  # Remove the NaNs
    # index_well = index_well[~np.isnan(index_well)]  # Remove the NaNs

    # platedf = datautils.generatedataframe(stackdata)
    data = [index_indiv,  index_stack, index_well]
    # alphas = [0.01, 0.5, 1]
    alphas = [1, 1, 1]
    for m, method in enumerate(method_types):
        print(m, method, selected_method_type, useall)
        if (selected_method_type == method) or useall:
            print(f"Plotting {method}")
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))#, sharey=True)
            # for t, treatment in enumerate(treatment_types):

            vp = sns.violinplot(ax=axs, x="Week", y=propname, hue="Treatment", cut=0,
                                data=data[m], inner=None, gridsize=100, palette="turbo",
                                split=True, scale=scaletype, zorder=0, ) #  dropna=True, width=violinwidths[t], inner="box"
            sns.stripplot(ax=axs, x="Week", y=propname, hue="Treatment", jitter=0.2,
                          alpha=alphas[m], data=data[m], dodge=True, edgecolor='black',
                          zorder=1)
            sns.boxplot(ax=axs, x="Week", y=propname, data=data[m], width=.8,
                        boxprops={'facecolor': 'None', "zorder": 10},
                        whiskerprops={"zorder": 10},
                        hue="Treatment", showfliers=False, dodge=True)
            handles, labels = vp.get_legend_handles_labels()
            axs.legend(handles[:0], labels[:0])
            axs.set_title(f"{channel} {method}", fontsize=24)

            l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            # for ax in axs:
            axs.yaxis.grid(True)
            axs.set_xlabel('Weeks', fontsize=18)
            axs.set_ylabel(f"{channel} {propname}(in {units})", fontsize=18)
            #     ax.set_xlabel('weeks',fontsize = 22)
            #     ax.set_ylabel(f"{propname}{units}",fontsize = 18)

            plt.setp(axs, xticks=[y for y in range(experimentalparams.USEDWEEKS)],
                     xticklabels=experimentalparams.WS[:experimentalparams.USEDWEEKS])
            # plt.show()
            plt.savefig(
                f"{savepath}_{channel}_{propname}_{method}_weeks{experimentalparams.USEDWEEKS}_{'withstrpplt' if withstrpplt else ''}{'_s-' + str(savesigma) if savesigma else ''}.png")
            plt.close()
            plt.clf()


if __name__ == "__main__":
    """
    TODO: an example of plotting
    """

    teststack = np.random.random((2, 4, 1, 5, 6, 1000)) * 100
    print(teststack.shape, teststack.ndim)
    if teststack.ndim == 6:
        teststack = np.expand_dims(teststack,axis=-1)

    print(teststack.shape, teststack.ndim)
    # violinstripplot(stackdata=teststack, channel="testchannel", propname="testproperty", units="units", savesigma=True,
    #                 selected_method_type="Stackwise")
