import matplotlib.pyplot as plt
import seaborn as sns

from src.AnalysisTools import datautils, experimentalparams, statcalcs

plt.rcParams["figure.figsize"] = [12, 9]


def generate_plot(data3dlist, propname, units, savesigma=None, savepath="", channel="", withstrpplt=False):
    datadf = datautils.generatedataframe(data3dlist, propname)
    #     print(datadf)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))
    sns.violinplot(ax=ax, x="Week", y=propname, hue="Treatment", cut=0, data=datadf, gridsize=100, split=True,
                   scale="count", )

    ax.set_xlabel('weeks', fontsize=22)
    ax.set_ylabel(f"{propname}{units}", fontsize=18)

    plt.setp(ax, xticks=[y for y in range(experimentalparams.USEDWEEKS)],
             xticklabels=experimentalparams.WS[:experimentalparams.USEDWEEKS])
    #     plt.show()
    plt.savefig(
        f"{savepath}_{channel}_{propname}_weeks{experimentalparams.USEDWEEKS}_{'withstrpplt' if withstrpplt else ''}{'_s-' + savesigma if savesigma else ''}.png")
    plt.close()
    plt.clf()

    # a4_dims = (11.7, 8.27)

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


def violinstripplot(data, organelle="Cell", propname="", unit="", sigma=0, savesigma="", savepath=""):
    """
    TODO: check data types
    :param data:
    :param organelle:
    :param propname:
    :param unit:
    :param sigma:
    :param savesigma:
    :return:
    """
    try:
        tempdata = data.copy()
        weekly_data_conform = statcalcs.removeoutliers3dlist(tempdata, m=sigma)
        violinstripplot(inddata, stackdata, channel="Cell", propname=propname, units=f"(in {unit})", savesigma=savesigma, savepath=savepath)
        # weekly_data_conform, propname=f"{organelle} {propname}", units=, savesigma=savesigma)
        return 0
    except Exception as e:
        print(e)
        return 1


def violinstripplot(individualdata, stackdata, channel="Cell", propname="", units="", savesigma=None, method_types=["Individual", "Stackwise"], savepath="", withstrpplt=True):
    inddatadf = datautils.generatedataframeind(individualdata, propname)
    datadf = datautils.generatedataframe(stackdata, propname)
    data = [inddatadf, datadf]
    alphas = [0.01, 0.5]
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 8), sharey=True)
    for m, method in enumerate(method_types):
        vp = sns.violinplot(ax=axs[m], x="Week", y=propname, hue="Treatment", cut=0, data=data[m], inner=None, gridsize=100, palette="turbo", split=True, scale="count", zorder=0)
        sns.stripplot(ax=axs[m], x="Week", y=propname, hue="Treatment", jitter=0.2, alpha=alphas[m], data=data[m], dodge=True, edgecolor='black', zorder=1)
        sns.boxplot(ax=axs[m], x="Week", y=propname, data=data[m], width=.8, boxprops={'facecolor': 'None', "zorder": 10}, whiskerprops={"zorder": 10}, hue="Treatment", showfliers=False, dodge=True)
        handles, labels = vp.get_legend_handles_labels()
        axs[m].legend(handles[:0], labels[:0])
        axs[m].set_title(method_types[m], fontsize=24)
    l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    for ax in axs:
        ax.yaxis.grid(True)
        ax.set_xlabel('Weeks', fontsize=18)
        ax.set_ylabel(f"{channel} {propname}{units}", fontsize=18)
    #     ax.set_xlabel('weeks',fontsize = 22)
    #     ax.set_ylabel(f"{propname}{units}",fontsize = 18)

    plt.setp(ax, xticks=[y for y in range(experimentalparams.USEDWEEKS)], xticklabels=experimentalparams.WS[:experimentalparams.USEDWEEKS])
    #     plt.show()
    plt.savefig(f"{savepath}_{channel}_{propname}_weeks{experimentalparams.USEDWEEKS}_{'withstrpplt' if withstrpplt else ''}{'_s-' + savesigma if savesigma else ''}.png")
    plt.close()
    plt.clf()
