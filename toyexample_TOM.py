import datetime
import traceback
from concurrent.futures import ProcessPoolExecutor
from os.path import join

import numpy as np
import pandas as pd
from scipy.ndimage.measurements import find_objects  # ,label
# from scipy.ndimage.measurements import label, find_objects
from skimage.measure import label as skilbl

from src.AnalysisTools import experimentalparams, datautils, ShapeMetrics
from src.Visualization import plotter
from src.stackio import stackio


def getusedchannels(filelist):
    channels = []
    for file in filelist:
        channel = file.split("_")[0].split("-")[-1]
        if channel not in channels:
            channels.append(channel)
    return channels


if __name__ == "__main__":
    doindividualcalcs = True
    channel = "TOM20"
    ###############################################
    usedweeks = experimentalparams.USEDWEEKS
    usedtreatments = experimentalparams.USEDTREATMENTS
    treatment_type = experimentalparams.TREATMENT_TYPES
    ###############################################

    segmented_ch_folder = '../Results/2022/Jan21/TOM_stack_18img/segmented/TOM/'
    savepath = '../Results/2022/Jan21/TOM_stack_18img/segmented/calcs/'

    dnafnames = datautils.getFileListContainingString(segmented_ch_folder, 'DNA_RPE.tif')
    actinfnames = datautils.getFileListContainingString(segmented_ch_folder, 'Actin_RPE.tif')
    GFPfnames = datautils.getFileListContainingString(segmented_ch_folder, '_GFP')

    dnafiles, actinfiles, GFPfiles = datautils.orderfilesbybasenames(dnafnames, actinfnames, GFPfnames, debug=False)
    channels = getusedchannels(actinfiles)
    usedchannels = len(channels)
    totalFs = experimentalparams.TOTALFIELDSOFVIEW
    usedwells = experimentalparams.USEDWELLS  # well numbers divided by treatment
    maxnocells = experimentalparams.MAX_CELLS_PER_STACK
    maxdnapercell = experimentalparams.MAX_DNA_PER_CELL
    maxorganellepercell = experimentalparams.MAX_ORGANELLE_PER_CELL
    # print("USEDCHANNELS: ", usedchannels)
    badfiles = []
    cellstackvols = np.nan * np.ones((usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells))
    cellstackxspan = np.nan * np.ones((usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells))
    cellstackyspan = np.nan * np.ones((usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells))
    cellstackzspan = np.nan * np.ones((usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells))
    cellstackmiparea = np.nan * np.ones((usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells))
    cellstackmaxferet = np.nan * np.ones((usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells))
    cellstackminferet = np.nan * np.ones((usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells))

    dnastackvols = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxdnapercell))
    dnastackxspan = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxdnapercell))
    dnastackyspan = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxdnapercell))
    dnastackzspan = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxdnapercell))
    dnastackmiparea = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxdnapercell))
    dnastackmaxferet = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxdnapercell))
    dnastackminferet = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxdnapercell))

    gfpstackvols = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxorganellepercell))
    gfpstackxspan = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxorganellepercell))
    gfpstackyspan = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxorganellepercell))
    gfpstackzspan = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxorganellepercell))
    gfpstackmiparea = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxorganellepercell))
    gfpstackmaxferet = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxorganellepercell))
    gfpstackminferet = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxorganellepercell))

    num_processes = 8
    executor = ProcessPoolExecutor(num_processes)
    processes = []

    minlength = 5
    binvals = np.arange(minlength)
    freq = binvals.copy()

    for i, (actinfile, dnafile, GFPfile) in enumerate(zip(actinfiles, dnafiles, GFPfiles)):
        # if i != 0:  # Test only 1 dataset
        #     continue
        try:
            week, rep, w, r, fov, fovno, basename = datautils.getwr_3channel(dnafile, actinfile, GFPfile)
            t = experimentalparams.findtreatment(r)
            print(week, rep, w, r, fov, fovno, basename, t)
            if w < usedweeks:

                start_ts = datetime.datetime.now()
                ##GET IMAGES
                img_GFP = stackio.opensegmentedstack(join(segmented_ch_folder, GFPfile))
                img_ACTIN = stackio.opensegmentedstack(join(segmented_ch_folder, actinfile))  # binary=False
                img_DNA = stackio.opensegmentedstack(join(segmented_ch_folder, dnafile))  # binary=False
                print("TEST: all should be equal", np.unique(img_GFP), np.unique(img_ACTIN), np.unique(img_DNA))
                print("TEST: all should be equal", img_GFP.shape, img_ACTIN.shape, img_DNA.shape)
                # imwrite(f'{savepath}{basename}_GFP.tiff', IMGGFP_0, compress=6)
                # imwrite(f'{savepath}{basename}_Actin.tiff', IMGactin, compress=6)
                # imwrite(f'{savepath}{basename}_DNA.tiff', IMGDNA, compress=6)

                # actin_label, icellcounts = label(img_ACTIN)
                # NOTE: default structuring element is full for skimage = ndim and for scipy = 1
                # obj_df = pd.DataFrame(np.arange(1, icellcounts + 1, 1), columns=['object_index'])
                # imwrite(f'{savepath}{basename}_labelactin_skimage.tiff', labelactin, compress=6)
                # imwrite(f'{savepath}{basename}_labelactin_scipy.tiff', actin_label, compress=6)

                labelactin, icellcounts = skilbl(img_ACTIN, return_num=True)
                labeldna = skilbl(img_DNA)
                labelGFP = skilbl(img_GFP)
                # create variables
                is_ = list(np.unique(labelactin))
                js_ = list(np.unique(labeldna))
                is_.remove(0)
                js_.remove(0)
                maxid = None

                savename = "_".join(actinfile.split("_")[:-2]) + "_connectmat"
                # savenameoverlaps = "_".join(actinfile.split("_")[:-2]) + "_overlapmat"
                print(f"starting overlap for {savename} \t actinlabels: ", len(is_), "dnalabels:", len(js_), flush=True)
                ######################################################################################################
                percent_actin_matrix = np.zeros((len(is_), len(js_)))
                ####percentage based on total actin overlap###########################################################
                dna_actin_membership = np.zeros((len(is_), len(js_)), dtype=int)
                ###membership array###################################################################################
                over75list_actin = np.zeros(len(js_))
                ######################################################################################################
                for j in js_:  # in dna objects
                    selected_dna = (labeldna == j) > 0  # select volume
                    # print("SELECTED DNA", selected_dna.shape, np.unique(selected_dna), np.sum(selected_dna) )
                    lactin = labelactin[selected_dna]
                    connected_actins = list(np.unique(lactin))
                    combined_actins_volume_overlap = np.count_nonzero(lactin)
                    # print("SELECTED DNA", selected_dna.shape, np.unique(selected_dna), np.sum(selected_dna), actins_volume_overlap_total, connected_actins)
                    for i in is_:  # actin ids - this also removes 0 since its background
                        if i in connected_actins:
                            selected_actin_overlap_volume = np.count_nonzero(lactin == i)
                            # bboxcrop = find_objects(objs)
                            # slices = bboxcrop[0]
                            percent_coverage_actin = selected_actin_overlap_volume / combined_actins_volume_overlap
                            percent_actin_matrix[i - 1, j - 1] = percent_coverage_actin
                            if percent_coverage_actin >= 0.75:  ##0.75% or more of dna lies in one actin
                                over75list_actin[j - 1] = i
                                dna_actin_membership[i - 1, j - 1] = 1
                bincount = np.bincount(np.sum(dna_actin_membership, axis=1), minlength=minlength)
                freq = np.vstack((freq, bincount[binvals]))  # number of dna per actin
                print(f"Overlap calculations for {basename} done")
                # print(freq)
                # print(dna_actin_membership.shape)
                # print(over75list_actin)
                # print(over75list_actin.shape, len(np.unique(over75list_actin)))
                # print(dna_actin_membership, len(np.unique(dna_actin_membership)))
                # pd.DataFrame(dna_actin_membership).to_csv(savepath+"dnamem.csv")
                ##Now the calculation with membership data
                obj_df = pd.DataFrame(np.arange(1, icellcounts + 1, 1), columns=['object_index'])
                for index, row in obj_df.iterrows():
                    cellinputdict = {}
                    obj_index = int(row['object_index'])
                    objs = labelactin == obj_index  # object with object label 'obj_index'
                    # get bounding box of actin
                    bbox_actin = find_objects(objs)
                    slices = bbox_actin[0]
                    # print("calculating actin object")
                    edgetags, top, bot = ShapeMetrics.getedgeconnectivity(slices, objs.shape[0])

                    centroid, volume, xspan, yspan, zspan, maxferet, minferet, miparea = ShapeMetrics.calcs_(
                        objs[slices])
                    # print("calculated actin object")
                    cellvals = [centroid, volume, xspan, yspan, zspan, maxferet, minferet, miparea, top, bot]
                    if experimentalparams.checkcellconditions(cellvals) and datautils.checkfinitetemp(cellvals[1:8]):
                        cellstackvols[t, w, 0, r%5, fovno, obj_index] = volume
                        cellstackxspan[t, w, 0, r%5, fovno, obj_index] = xspan
                        cellstackyspan[t, w, 0, r%5, fovno, obj_index] = yspan
                        cellstackzspan[t, w, 0, r%5, fovno, obj_index] = zspan
                        cellstackmiparea[t, w, 0, r%5, fovno, obj_index] = miparea
                        cellstackmaxferet[t, w, 0, r%5, fovno, obj_index] = maxferet
                        cellstackminferet[t, w, 0, r%5, fovno, obj_index] = minferet
                        ##DNA members
                        memberdnas = np.where(dna_actin_membership[i - 1, :])[
                            0]  # -1 since this matrix uses indices from 0
                        no_members = memberdnas.shape[0]
                        # actin_channel = 255 * ((labelactin == i) > 0)
                        selected_dna_members = np.zeros_like(labelactin[slices])
                        # print(memberdnas, no_members, selected_dna_members.shape)
                        # print("calculated dna object")

                        if no_members > 0:
                            for memberdna_no in range(no_members):
                                if memberdna_no > 2:
                                    continue
                                dnaobj = (labeldna[slices] == (memberdnas[memberdna_no] + 1))
                                assert (dnaobj.shape == labeldna[slices].shape == labelactin[slices].shape)
                                centroid, dnavol, dnax, dnay, dnaz, dnamaxf, dnaminf, dnamip = ShapeMetrics.calcs_(
                                    dnaobj)
                                dnastackvols[t, w, 0, r%5, fovno, obj_index, memberdna_no] = dnavol
                                dnastackxspan[t, w, 0, r%5, fovno, obj_index, memberdna_no] = dnax
                                dnastackyspan[t, w, 0, r%5, fovno, obj_index, memberdna_no] = dnay
                                dnastackzspan[t, w, 0, r%5, fovno, obj_index, memberdna_no] = dnaz
                                dnastackmiparea[t, w, 0, r%5, fovno, obj_index, memberdna_no] = dnamip
                                dnastackmaxferet[t, w, 0, r%5, fovno, obj_index, memberdna_no] = dnamaxf
                                dnastackminferet[t, w, 0, r%5, fovno, obj_index, memberdna_no] = dnaminf
                        # GFP members
                        IMGGFP_obj = (img_GFP & objs)[slices]
                        # print("IMGGFP_obj:: ", IMGGFP_obj.shape)

                        GFPcentroid, GFPvolume, GFPxspan, GFPyspan, GFPzspan, GFPmaxferet, GFPminferet, GFPmip = ShapeMetrics.calcs_(
                            IMGGFP_obj[slices])
                        GFPvals = [GFPvolume, GFPxspan, GFPyspan, GFPzspan, GFPmaxferet, GFPminferet, GFPmip]
                        # print("GFPvals:: ", GFPvals)
                        if datautils.checkfinitetemp(GFPvals):
                            if doindividualcalcs:
                                print("doing individual calcs")
                                indcentroid, indvolume, indxspan, indyspan, indzspan, indmaxferet, indminferet, indmiparea = ShapeMetrics.individualcalcs(
                                    IMGGFP_obj)
                                gfpstackvols[t, w, 0, r%5, fovno, obj_index, :] = indvolume
                                gfpstackxspan[t, w, 0, r%5, fovno, obj_index, :] = indxspan
                                gfpstackyspan[t, w, 0, r%5, fovno, obj_index, :] = indyspan
                                gfpstackzspan[t, w, 0, r%5, fovno, obj_index, :] = indzspan
                                gfpstackmiparea[t, w, 0, r%5, fovno, obj_index, :] = indmiparea
                                gfpstackmaxferet[t, w, 0, r%5, fovno, obj_index, :] = indmaxferet
                                gfpstackminferet[t, w, 0, r%5, fovno, obj_index, :] = indminferet
                                # processes.append((t, w, r, fovno, obj_index,
                                #                   executor.submit(ShapeMetrics.individualcalcs, IMGGFP_obj[slices])))

                # for t, w, r, fovno,obj_id, process in processes:
                #     print('in processes')
                #     features = process.result()
                #     indcentroid, indvolume, indxspan, indyspan, indzspan, indmaxferet, indminferet, indmiparea = features

            end_ts = datetime.datetime.now()
            #############
            print(f"{basename} done in {str(end_ts - start_ts)}")

        except Exception as e:
            print(e, traceback.format_exc())

    allCELLvals = [cellstackvols, cellstackxspan, cellstackyspan, cellstackzspan, cellstackmiparea, cellstackmaxferet,
                   cellstackminferet]
    allDNAvals = [dnastackvols, dnastackxspan, dnastackyspan, dnastackzspan, dnastackmiparea, dnastackmaxferet,
                  dnastackminferet]
    allGFPvals = [gfpstackvols, gfpstackxspan, gfpstackyspan, gfpstackzspan, gfpstackmiparea, gfpstackmaxferet,
                  gfpstackminferet]

    # allCELLvalnames = ["cellstackvols", "cellstackxspan", "cellstackyspan", "cellstackzspan", "cellstackmiparea","cellstackmaxferet","cellstackminferet"]
    # allDNAvalnames = ["dnastackvols", "dnastackxspan", "dnastackyspan", "dnastackzspan", "dnastackmiparea", "dnastackmaxferet", "dnastackminferet"]
    # allGFPvalnames = ["gfpstackvols", "gfpstackxspan", "gfpstackyspan", "gfpstackzspan", "gfpstackmiparea", "gfpstackmaxferet", "gfpstackminferet"]

    # indGFPvals = indGFPcentroidhs, indGFPvolumes, indGFPzspans, indGFPxspans, indGFPyspans, indGFPmaxferets, indGFPminferets  # , indGFPorients
    propnames = ["volume", "xspan", "yspan", "zspan", "miparea", "max feret", "min feret"]
    unittype = ["cu. microns", "microns", "microns", "microns", "sq. microns", "microns", "microns"]
    withstrpplt = True
    sigma = 2
    strsigma = "95.45"

    orgenelletype = ["Cell", "DNA", channel]
    propertycategory = [allCELLvals, allDNAvals, allGFPvals]
    # propertycategory_names = [allCELLvalnames,allDNAvalnames,allGFPvalnames]
    generateplots, savedata = True, True
    # for otype in orgenelletype:

    for o, (propertytype, otype) in enumerate(zip(propertycategory, orgenelletype)):
        for i, prop in enumerate(propertytype):
            if savedata:
                filename = f"{otype}_{propnames[i]}_{strsigma}.npz"
                fpath = join(savepath, filename)
                stackio.saveproperty(prop, filepath=fpath, type="npz")
                loaded = stackio.loadproperty(fpath)

                success = datautils.array_nan_equal(loaded[loaded.files[0]], prop)
                if success:
                    print(f"SAVE SUCCESSFUL FOR {filename}")
                else:  # (2, 4, 1, 5, 6, 1000, 50)
                    print(loaded.files, loaded[loaded.files[0]].shape, prop.shape)
            try:
                if generateplots:
                    plotter.violinstripplot(stackdata=prop, channel=otype, propname=propnames[i], units=unittype[i],
                                            savesigma=True, selected_method_type=["Stackwise"])
            except Exception as e:
                print(e)
    # tempcellvol = cellvolumes.copy()
