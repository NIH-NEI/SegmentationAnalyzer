"""
TODO:
    1. fix SegmentationAnalyzer environment and replace current one.
    2. create a jupyter notebook for tutorial
"""

# import metadataWriter
import datetime
import traceback
from concurrent.futures import ProcessPoolExecutor
from os.path import join

import numpy as np
import pandas as pd
from scipy.ndimage.measurements import label, find_objects

from src.AnalysisTools import experimentalparams, datautils, ShapeMetrics
from src.Visualization import plotter
from src.stackio import stackio

if __name__ == "__main__":
    doindividualcalcs = True
    channel = "TOM20"
    ###############################################
    usedweeks = experimentalparams.USEDWEEKS
    usedtreatments = experimentalparams.USEDTREATMENTS
    treatment_type = experimentalparams.TREATMENT_TYPES
    ###############################################

    segmented_ch_folder = '../Results/2021/july23/TOM20segmented/'

    dnafnames = datautils.getFileListContainingString(segmented_ch_folder, 'DNA_RPE.tif')
    actinfnames = datautils.getFileListContainingString(segmented_ch_folder, 'Actin_RPE_.tif')
    GFPfnames = datautils.getFileListContainingString(segmented_ch_folder, '_TOM20seg')

    savepath = '../Results/2021/Aug6/GFPproperties/'

    dnafiles, actinfiles, GFPfiles = datautils.orderfilesbybasenames(dnafnames, actinfnames, GFPfnames)

    assert len(dnafiles) == len(actinfiles) == len(GFPfiles)

    cellcentroidhs, cellvolumes, cellzspans, cellxspans, cellyspans, cellmaxferets, cellminferets = datautils.createlistof3dlists(n=7)
    GFPcentroidhs, GFPvolumes, GFPzspans, GFPxspans, GFPyspans, GFPmaxferets, GFPminferets = datautils.createlistof3dlists(n=7)
    indGFPcentroidhs, indGFPvolumes, indGFPzspans, indGFPxspans, indGFPyspans, indGFPmaxferets, indGFPminferets, indGFPorients = datautils.createlistof3dlists(n=8)
    percvolumes = datautils.create3dlist(usedtreatments, usedweeks)

    # NJS
    num_processes = 8
    executor = ProcessPoolExecutor(num_processes)
    processes = []

    for i, (actinfile, dnafile, GFPfile) in enumerate(zip(actinfiles, dnafiles, GFPfiles)):
        try:
            if i > 20:
                raise Exception
            week, rep, w, r, basename = experimentalparams.getwr(dnafile, actinfile, GFPfile)
            t = experimentalparams.findtreatment(r)
            if w < usedweeks:
                start_ts = datetime.datetime.now()

                IMGGFP_0, IMGactin = stackio.opensegmentedstack(join(segmented_ch_folder, GFPfile)), stackio.opensegmentedstack(
                    join(segmented_ch_folder, actinfile), binary=False)
                actin_label, icellcounts = label(IMGactin > 0)
                obj_df = pd.DataFrame(np.arange(1, icellcounts + 1, 1), columns=['object_index'])
                print('entering loop1')
                ############################################################################################################
                for index, row in obj_df.iterrows():
                    cellinputdict = {}
                    obj_index = int(row['object_index'])
                    objs = actin_label == obj_index  # object with object label 'obj_index'
                    bboxcrop = find_objects(objs)
                    slices = bboxcrop[0]
                    IMGGFP_obj = (IMGGFP_0 & objs)
                    # print(np.unique(IMGGFP_obj), np.count_nonzero(IMGGFP_obj))
                    edgetags = ShapeMetrics.getedgeconnectivity(slices, objs.shape[0])
                    centroid, volume, xspan, yspan, zspan, maxferet, minferet = ShapeMetrics.calcs_(objs[slices])
                    cellvals = [centroid, volume, xspan, yspan, zspan, maxferet, minferet]
                    # print(experimentalparams.checkcellconditions(cellvals))
                    print(datautils.checkfinite(cellvals))

                    # print('done calc1', flush=True)
                    if experimentalparams.checkcellconditions(cellvals) and datautils.checkfinite(cellvals):
                        # GFPcentroid, GFPvolume, GFPxspan, GFPyspan, GFPzspan, GFPmaxferet, GFPminferet = ShapeMetrics.calcs_(IMGGFP_obj[slices])
                        GFPcentroid, GFPvolume, GFPxspan, GFPyspan, GFPzspan, GFPmaxferet, GFPminferet = 0, 0, 0, 0, 0, 0, 0
                        GFPvals = [GFPcentroid, GFPvolume, GFPxspan, GFPyspan, GFPzspan, GFPmaxferet, GFPminferet]
                        # print('done calc2', flush=True)
                        if datautils.checkfinite(GFPvals):
                            cellcentroidhs[t][w].append(centroid), cellvolumes[t][w].append(volume)
                            cellxspans[t][w].append(xspan), cellyspans[t][w].append(yspan), cellzspans[t][w].append(zspan)
                            cellmaxferets[t][w].append(maxferet), cellminferets[t][w].append(minferet)

                            GFPcentroidhs[t][w].append(GFPcentroid), GFPvolumes[t][w].append(GFPvolume)
                            GFPxspans[t][w].append(GFPxspan), GFPyspans[t][w].append(GFPyspan), GFPzspans[t][w].append(GFPzspan)
                            GFPmaxferets[t][w].append(GFPmaxferet), GFPminferets[t][w].append(GFPminferet)

                            percvolumes[t][w].append(GFPvolume / volume)
                            print('entering processes')
                            # NJS
                            if doindividualcalcs:
                                processes.append((t, w, executor.submit(ShapeMetrics.individualcalcs, IMGGFP_obj[slices])))
                                #
                                # t, w, process = processes[-1]
                                # features = process.result()
                                # print(t, w, features)

                # NJS
                for t, w, process in processes:
                    print('in processes')
                    features = process.result()
                    indcentroid, indvolume, indxspan, indyspan, indzspan, indmaxferet, indminferet, indorient = features
                    indGFPcentroidhs[t][w].extend(indcentroid), indGFPvolumes[t][w].extend(indvolume)
                    indGFPxspans[t][w].extend(indxspan), indGFPyspans[t][w].extend(indyspan), indGFPzspans[t][w].extend(
                        indzspan)
                    indGFPmaxferets[t][w].extend(indmaxferet), indGFPminferets[t][w].extend(indminferet)
                    indGFPorients[t][w].extend(indorient)
                end_ts = datetime.datetime.now()

                print(f"{basename} done in {str(end_ts - start_ts)}")
        except Exception as e:
            print(e, traceback.format_exc())
    allcellvals = cellcentroidhs, cellvolumes, cellzspans, cellxspans, cellyspans, cellmaxferets, cellminferets
    allGFPvals = GFPcentroidhs, GFPvolumes, GFPzspans, GFPxspans, GFPyspans, GFPmaxferets, GFPminferets
    indGFPvals = indGFPcentroidhs, indGFPvolumes, indGFPzspans, indGFPxspans, indGFPyspans, indGFPmaxferets, indGFPminferets  # , indGFPorients
    propnames = ["centroid", "volume", "zspan", "xspan", "yspan", "max feret", "min feret"]
    unittype = ["microns", "cu. microns", "microns", "microns", "microns", "microns", "microns"]
    withstrpplt = True
    sigma = 2
    strsigma = "95.45"

    orgenelletype = ["Cell", channel, channel]
    propertycategory = [allcellvals, allGFPvals, indGFPvals]
    for otype in orgenelletype:
        for propertytype in propertycategory:
            for i, prop in enumerate(propertytype):
                plotter.violinstripplot(data=prop, channel=otype, propname=propnames[i], unit=unittype[i], sigma=sigma, savesigma=strsigma)

    # tempcellvol = cellvolumes.copy()
    # weeklycellvols_conform = statcalcs.removeoutliers3dlist(tempcellvol, m=sigma)
    # plotter.generate_plot(weeklycellvols_conform, propname="Cell volume", units="(in cu. micron)", savesigma=strsigma)
    #
    # tempGFPvol = GFPvolumes.copy()
    # weeklyGFPvols_conform = statcalcs.removeoutliers3dlist(tempGFPvol, m=sigma)
    # plotter.generate_plot(weeklyGFPvols_conform, propname="TOM20 volume", units="(in cu. micron)", savesigma=strsigma)
    #
    # temppercvolumes = percvolumes.copy()
    # weeklypercvolumes_conform = statcalcs.removeoutliers3dlist(temppercvolumes, m=sigma)
    # weeklypercvolumes_conform = [[[i * 100 for i in innermost] for innermost in inner] for inner in
    #                              weeklypercvolumes_conform]
    # plotter.generate_plot(weeklypercvolumes_conform, propname="percentage volume TOM20", units="(in percent)",
    #                       savesigma=strsigma)
    #
    # tempindGFPvolumes = indGFPvolumes.copy()
    # weeklyindGFPvolumes_conform = statcalcs.removeoutliers3dlist(tempindGFPvolumes, m=sigma)
    # plotter.generate_plot(weeklyindGFPvolumes_conform, propname="Individual TOM20 Volume", units="(in cu. micron)",
    #                       savesigma=strsigma)
    #
    # tempindGFPzspans = indGFPzspans.copy()
    # weeklyindGFPzspans_conform = statcalcs.removeoutliers3dlist(tempindGFPzspans, m=sigma)
    # plotter.generate_plot(weeklyindGFPzspans_conform, propname="Individual TOM20 zspan", units="(in microns)",
    #                       savesigma=strsigma)
    #
    # tempindGFPxspans = indGFPxspans.copy()
    # weeklyindGFPxspans_conform = statcalcs.removeoutliers3dlist(tempindGFPxspans, m=sigma)
    # plotter.generate_plot(weeklyindGFPxspans_conform, propname="Individual TOM20 xspan", units="(in microns)",
    #                       savesigma=strsigma)
    #
    # tempindGFPyspans = indGFPyspans.copy()
    # weeklyindGFPyspans_conform = statcalcs.removeoutliers3dlist(tempindGFPyspans, m=sigma)
    # plotter.generate_plot(weeklyindGFPyspans_conform, propname="Individual TOM20 yspan", units="(in microns)",
    #                       savesigma=strsigma)
    #
    # tempindGFPmaxferets = indGFPmaxferets.copy()
    # weeklyindGFPmaxferets_conform = statcalcs.removeoutliers3dlist(tempindGFPmaxferets, m=sigma)
    # plotter.generate_plot(weeklyindGFPmaxferets_conform, propname="Individual TOM20 max feret", units="(in microns)",
    #                       savesigma=strsigma)
    #
    # tempindGFPminferets = indGFPminferets.copy()
    # weeklyindGFPminferets_conform = statcalcs.removeoutliers3dlist(tempindGFPminferets, m=sigma)
    # plotter.generate_plot(weeklyindGFPminferets_conform, propname="Individual TOM20 min feret", units="(in microns)",
    #                       savesigma=strsigma)
    #
    # tempindGFPcentroidhs = indGFPcentroidhs.copy()
    # weeklyindGFPcentroidhs_conform = [[[i1[0] for i1 in i2] for i2 in i3] for i3 in tempindGFPcentroidhs]
    # plotter.generate_plot(weeklyindGFPcentroidhs_conform, propname="Individual TOM20 centroid", units="(in microns)",
    #                       savesigma=strsigma)
    #
    # tempcellzspans = cellzspans.copy()
    # weeklycellzspans_conform = statcalcs.removeoutliers3dlist(tempcellzspans, m=sigma)
    # plotter.generate_plot(weeklycellzspans_conform, propname="cell zspan", units="(in microns)", savesigma=strsigma)
    #
    # tempGFPzspans = GFPzspans.copy()
    # weeklyGFPzspans_conform = statcalcs.removeoutliers3dlist(tempGFPzspans, m=sigma)
    # plotter.generate_plot(weeklyGFPzspans_conform, propname="TOM20 zspan", units="(in microns)", savesigma=strsigma)
    #
    # tempcellxspans = cellxspans.copy()
    # weeklycellxspans_conform = statcalcs.removeoutliers3dlist(tempcellxspans, m=sigma)
    # plotter.generate_plot(weeklycellxspans_conform, propname="cell xspan", units="(in microns)", savesigma=strsigma)
    #
    # tempGFPxspans = GFPxspans.copy()
    # weeklyGFPxspans_conform = statcalcs.removeoutliers3dlist(tempGFPxspans, m=sigma)
    # plotter.generate_plot(weeklyGFPxspans_conform, propname="TOM20 xspan", units="(in microns)", savesigma=strsigma)
    #
    # tempcellyspans = cellyspans.copy()
    # weeklycellyspans_conform = statcalcs.removeoutliers3dlist(tempcellyspans, m=sigma)
    # plotter.generate_plot(weeklycellyspans_conform, propname="cell yspan", units="(in microns)", savesigma=strsigma)
    #
    # tempGFPyspans = GFPyspans.copy()
    # weeklyGFPyspans_conform = statcalcs.removeoutliers3dlist(tempGFPyspans, m=sigma)
    # plotter.generate_plot(weeklyGFPyspans_conform, propname="TOM20 yspan", units="(in microns)", savesigma=strsigma)
    #
    # # GFP#centroid heights
    # tempcellcentroidhs = cellcentroidhs.copy()
    # tempcellcentroidhs = [[[i1[0] for i1 in i2] for i2 in i3] for i3 in tempcellcentroidhs]
    # # weeklycellcentroidhs_conform = rmoutliers(tempcellcentroidhs,m=sigma)
    # plotter.generate_plot(tempcellcentroidhs, propname="cell centroid height", units="(in microns)", savesigma=strsigma)
    #
    # tempGFPcentroidhs = GFPcentroidhs.copy()
    # tempGFPcentroidhs = [[[i1[0] for i1 in i2] for i2 in i3] for i3 in tempGFPcentroidhs]
    # plotter.generate_plot(tempGFPcentroidhs, propname="TOM20 centroid height", units="(in microns)", savesigma=strsigma)
    #
    # tempcellmaxferets = cellmaxferets.copy()
    # weeklycellmaxferets_conform = statcalcs.removeoutliers3dlist(tempcellmaxferets, m=sigma)
    # plotter.generate_plot(weeklycellmaxferets_conform, propname="cell max feret", units="(in microns)", savesigma=strsigma)
    #
    # tempcellminferets = cellminferets.copy()
    # weeklycellminferets_conform = statcalcs.removeoutliers3dlist(tempcellminferets, m=sigma)
    # plotter.generate_plot(weeklycellminferets_conform, propname="cell min feret", units="(in microns)", savesigma=strsigma)
    #
    # tempGFPmaxferets = GFPmaxferets.copy()
    # weeklyGFPmaxferets_conform = statcalcs.removeoutliers3dlist(tempGFPmaxferets, m=sigma)
    # plotter.generate_plot(weeklyGFPmaxferets_conform, propname="TOM20 max feret", units="(in microns)", savesigma=strsigma)
    #
    # tempGFPminferets = GFPminferets.copy()
    # weeklyGFPminferets_conform = statcalcs.removeoutliers3dlist(tempGFPminferets, m=sigma)
    # plotter.generate_plot(weeklyGFPminferets_conform, propname="TOM20 min feret", units="(in microns)", savesigma=strsigma)
