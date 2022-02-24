import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.AnalysisTools import datautils, experimentalparams
from src.AnalysisTools import statcalcs

# from scipy.stats import f_oneway,ks_2samp
plt.rcParams["figure.figsize"] = [12, 9]
a4_dims = (11.7, 8.27)


def returnlogbounds(stack, getintegers=True):  # non int doesnt work since we need range()
    minval = np.nanmin(stack)
    maxval = np.nanmax(stack)
    if np.isnan(minval):
        minval = 0
    else:
        if getintegers:
            minval = np.floor(minval)
        else:
            minval = 0.9 * minval
    if np.isnan(maxval):
        maxval = 1
    else:
        if getintegers:
            maxval = np.ceil(maxval)
        else:
            maxval = 1.1 * maxval
    return int(minval), int(maxval)


def plotstattests(data, testname, savepath, channel, propertyname, percentile=90, stattype=None, commonparameter=""):
    """

    :param data:
    :param testname:
    :param savepath:
    :param channel:
    :param propertyname:
    :param percentile:
    :param stattype:
    :param commonparameter:
    :return:
    """
    if stattype is None:
        stattype = ['fvalue', 'pvalue']
    try:
        usedweeks = experimentalparams.USEDWEEKS
        usedtreatments = experimentalparams.USEDTREATMENTS
        weeks = experimentalparams.WS
        treatment_type = experimentalparams.TREATMENT_TYPES
        usevals = len(data[0])
        if commonparameter == "treatments":
            xticklabels = treatment_type[:usedtreatments]
            comparisonof = "weeks"
        elif commonparameter == "weeks":
            xticklabels = weeks[:usedweeks]
            comparisonof = "treatments"
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
        fig.tight_layout(pad=5.0)

        for a, ax in enumerate(axs):
            if a == 1:
                ax.axhline(y=0.05, color='r', linestyle='-')
            xvals = [1, 2, 3, 4]  # +xgaps[i]
            ax.plot(xvals[:usevals], data[a], linestyle="", marker="s", label=f"{testname}_{stattype[a]}_{percentile}")
            ax.set_title(f"{stattype[a]}", fontsize=18)
            ax.set_xlabel(commonparameter, fontsize=16)
            ax.set_ylabel(stattype[a], fontsize=22)

            # ax.legend(percentile, bbox_to_anchor=(1.05, 1),loc='upper left')
        plt.setp(axs, xticks=[y + 1 for y in range(len(xticklabels))], xticklabels=xticklabels)
        plt.suptitle(f"{testname}(compare {comparisonof}) : {propertyname}({channel})", fontsize=20)
        plt.savefig(f"{savepath}stattest_{testname}_{channel}_{propertyname}_weeks{usedweeks}.png", bbox_inches="tight")
        plt.close()
        plt.clf()
    except Exception as e:
        logging.warning("Statistical Plotting error: ", e)



def stat_tests(stack, savepath="", channel="", propertyname="", percentile=90, generateplot = True):
    """

    :param stack:
    :param savepath:
    :param channel:
    :param propertyname:
    :param percentile:
    :return:
    """
    weeks = experimentalparams.WS
    treatments = experimentalparams.TREATMENT_TYPES
    usedtreatments = experimentalparams.USEDTREATMENTS
    usedweeks = experimentalparams.USEDWEEKS

    stack_wt = stack.reshape((usedtreatments, usedweeks, -1))
    anova_fs_weekly, anova_ps_weekly = [], []
    t_ps_treatmentwise, t_fs_treatmentwise = [], []
    ks_ps_treatmentwise, ks_fs_treatmentwise = [], []
    # chisq_ps_treatmentwise, chisq_fs_treatmentwise = [], []

    ######################################################################
    for w, week in enumerate(weeks[:usedweeks]):
        # default length should be 2
        weeklist = [stack_wt[j, w][~np.isnan(stack_wt[j, w])].flatten() for j in range(usedtreatments)]
        print(f"channel: {channel}, property:{propertyname}, weeklist{w}", list(range(usedtreatments)),
                      [wk.ndim for wk in weeklist])

        # chisq, pvalue_chisq = statcalcs.chisquaretest(weeklist)
        # chisq_ps_treatmentwise.append(pvalue_chisq)
        # chisq_fs_treatmentwise.append(chisq)

        fvalue_ks, pvalue_ks = statcalcs.kstest(weeklist)
        ks_ps_treatmentwise.append(pvalue_ks)
        ks_fs_treatmentwise.append(fvalue_ks)

        fvalue_t, pvalue_t = statcalcs.ttest(weeklist)
        t_ps_treatmentwise.append(pvalue_t)
        t_fs_treatmentwise.append(fvalue_t)
    ######################################################################
    for t, treatment in enumerate(treatments):
        treatmentlist = [stack_wt[t, i][~np.isnan(stack_wt[t, i])].flatten() for i in range(usedweeks)]
        print(f"channel: {channel}, property:{propertyname}, anovalist{t}", list(range(usedweeks)),
                      [tr.ndim for tr in treatmentlist])
        fvalue_anova, pvalue_anova = statcalcs.one_way_anova(treatmentlist)
        anova_fs_weekly.append(fvalue_anova)
        anova_ps_weekly.append(pvalue_anova)
    ######################################################################
    data_anova = [anova_fs_weekly, anova_ps_weekly]
    # data_chisq = [chisq_fs_treatmentwise, chisq_ps_treatmentwise]
    data_ks = [ks_fs_treatmentwise, ks_ps_treatmentwise]
    data_t = [t_fs_treatmentwise, t_ps_treatmentwise]
    if generateplot:
        plotstattests(data_anova, testname="ANOVA", savepath=savepath, channel=channel, propertyname=propertyname,
                      percentile=percentile, stattype=['f-value', 'p-value'], commonparameter="treatments")
        # plotstattests(data_chisq, testname="Chi-squared test", savepath=savepath, channel=channel,
        #               propertyname=propertyname, percentile=percentile, stattype=['chi squared', 'p-value'],
        #               commonparameter="weeks")
        plotstattests(data_ks, testname="Kolmogorov-Smirnov test", savepath=savepath, channel=channel,
                      propertyname=propertyname, percentile=percentile, stattype=['D value', 'p-value'],
                      commonparameter="weeks")
        plotstattests(data_t, testname="t-test", savepath=savepath, channel=channel, propertyname=propertyname,
                      percentile=percentile, stattype=['f-value', 'p-value'], commonparameter="weeks")


def violinstripplot(stackdata, channel="Cell", propname="", units="", savesigma=None, selected_method_type=None,
                    savepath="", withstrpplt=True, scaletype="count", uselog=False, statplots=True):
    """
        scaletype can be count, width or area

        :param stackdata: ndarray, has dimensions ((usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxorganellepercell))
        :param channel:
        :param propname:
        :param units:
        :param savesigma:
        :param selected_method_type:method type can be "Individual", "Stackwise", "platewise"
        :param savepath:
        :param withstrpplt:
        :param scaletype:
        :param uselog:
        :param statplots:
        :return:
        """
    withoutliers = True
    method_types = ["Individual", "Stackwise", "Platewise"]
    useall = False
    if selected_method_type == None:
        useall = True
    else:
        assert selected_method_type in method_types
    ####################################################################################################
    print(f"Datapoints in Original Stackdata {np.count_nonzero(~np.isnan(stackdata))}")
    stackdata_indiv = statcalcs.stackbyabstractionlevel(stackdata, abstraction=0)
    stackdata_stack = statcalcs.stackbyabstractionlevel(stackdata, abstraction=1)
    stackdata_well = statcalcs.stackbyabstractionlevel(stackdata, abstraction=2)

    stackdata_rmoutlier_indiv = statcalcs.removestackoutliers(stackdata, m=2, abstraction=0)
    stackdata_rmoutlier_stack = statcalcs.removestackoutliers(stackdata, m=2, abstraction=1)
    stackdata_rmoutlier_well = statcalcs.removestackoutliers(stackdata, m=2, abstraction=2)

    print(f"{np.count_nonzero(~np.isnan(stackdata))}, {np.count_nonzero(~np.isnan(stackdata_rmoutlier_indiv))},{np.count_nonzero(~np.isnan(stackdata_rmoutlier_stack))}, {np.count_nonzero(~np.isnan(stackdata_rmoutlier_well))}")
    ####################################################################################################
    if uselog:
        stackdata_rmoutlier_indiv = np.log10(stackdata_rmoutlier_indiv)
        stackdata_rmoutlier_stack = np.log10(stackdata_rmoutlier_stack)
        stackdata_rmoutlier_well = np.log10(stackdata_rmoutlier_well)
        # for log range : multiply

        indivlogmin, indivlogmax = returnlogbounds(stackdata_rmoutlier_indiv)
        stacklogmin, stacklogmax = returnlogbounds(stackdata_rmoutlier_stack)
        welllogmin, welllogmax = returnlogbounds(stackdata_rmoutlier_well)
        logmins = [indivlogmin, stacklogmin, welllogmin]
        logmaxes = [indivlogmax, stacklogmax, welllogmax]
    ####################################################################################################

    originaldata = [stackdata_indiv, stackdata_stack, stackdata_well]
    rmoutlierdata = [stackdata_rmoutlier_indiv, stackdata_rmoutlier_stack, stackdata_rmoutlier_well]
    if withoutliers:
        usedata = originaldata
    else:
        usedata = rmoutlierdata
    ####################################################################################################
    index_indiv = datautils.generateindexedstack(usedata[0], propname)
    index_stack = datautils.generateindexedstack(usedata[1], propname)
    index_well = datautils.generateindexedstack(usedata[2], propname)
    data = [index_indiv, index_stack, index_well]
    ####################################################################################################
    # alphas = [alpha_indiv, 0.75, 1]
    violinwidths = [0.8, 0.8, 0.8]
    boxwidths = [0.8, 0.8, 0.8]
    unitstext = ""
    if (units is not None) and (units != ""):
        unitstext = f"(in {units})"
    # return
    for m, method in enumerate(method_types):
        if (selected_method_type == method) or useall:
            linewidth = None
            logging.info(f"Plotting {method}")
            try:

                datacount = np.count_nonzero(~np.isnan(rmoutlierdata[m]))
                alpha = min(1, max(4000 / datacount, 0.01))
                gridsize = min(100, datacount)
                if datacount < 100:
                    linewidth = 1
                logging.info("plotinfo", selected_method_type, useall, datacount, alpha, gridsize, linewidth, flush=True)
            except:
                alpha = 1
                gridsize = 100
                logging.warning("plotinfo exception:", selected_method_type, useall, datacount, alpha, gridsize, linewidth,
                      flush=True)
            if statplots and not uselog:
                # test for statistical null hypothesis
                try:
                    stat_tests(usedata[m], savepath=savepath, channel=channel, propertyname=f"{propname} ({method})")
                except Exception as e:
                    logging.warning("statexception", e)

            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))  # , sharey=True)
            # inner = box for single boxplot
            vp = sns.violinplot(ax=axs, x="Week", y=propname, hue="Treatment", cut=0, data=data[m], inner=None,
                                gridsize=gridsize, palette="turbo", split=True, scale=scaletype,
                                zorder=0, width=violinwidths[m])
            sns.stripplot(ax=axs, x="Week", y=propname, hue="Treatment", jitter=0.2, alpha=alpha, data=data[m],
                          dodge=True, linewidth=linewidth, edgecolor='black', zorder=1)
            sns.boxplot(ax=axs, x="Week", y=propname, data=data[m], width=boxwidths[m],
                        boxprops={'facecolor': 'None', "zorder": 10}, whiskerprops={"zorder": 10}, hue="Treatment",
                        showfliers=False, dodge=1.1)
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
                axs.yaxis.set_ticks([np.log10(x) for p in range(logmins[m], logmaxes[m]) for x in
                                     np.linspace(10 ** p, 10 ** (p + 1), 5)], minor=True)

            axs.set_ylabel(f"{channel} {propname}{unitstext}", fontsize=18)
            plt.setp(axs, xticks=[week for week in range(experimentalparams.USEDWEEKS)],
                     xticklabels=experimentalparams.WS[:experimentalparams.USEDWEEKS])
            plt.savefig(
                f"{savepath}_{channel}_{propname}_{method}_weeks{experimentalparams.USEDWEEKS}_{'withstrpplt' if withstrpplt else ''}{'_log' if uselog else ''}{'_s-' + str(savesigma) if savesigma else ''}.png")
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
