import datetime
import traceback
from concurrent.futures import ProcessPoolExecutor
from os.path import join, exists
from aicsimageio import AICSImage

import numpy as np
import pandas as pd
from scipy.ndimage.measurements import find_objects  # ,label
# from scipy.ndimage.measurements import label, find_objects
from skimage.measure import label as skilbl

from src.AnalysisTools import experimentalparams, datautils, ShapeMetrics
from src.Visualization import plotter
from src.stackio import stackio
# backup 2/2/22
if __name__ == "__main__":
    doindividualcalcs = True
    channel = "TOM20"
    ###############################################
    usedweeks = experimentalparams.USEDWEEKS
    usedtreatments = experimentalparams.USEDTREATMENTS
    treatment_type = experimentalparams.TREATMENT_TYPES
    ###############################################

    # segmented_ch_folder = '../Results/2022/Jan21/TOM_stack_18img/segmented/TOM/'

    # segmented_ch_folder = '../Results/2022/Jan28/TOM/segmented/'
    # savepath = '../Results/2022/Feb4/TOM/all/'

    segmented_ch_folder = '../Results/2022/Jan21/TOM_stack_18img/segmented/TOM/'
    # savepath = '../Results/2022/Jan21/TOM_stack_18img/segmented/calcs/'
    # savepath = '../Results/2022/Jan28/TOM/TOM_calcs_test/'
    savepath = '../Results/2022/Feb4/TOM/newpropstest/'
    assert exists(segmented_ch_folder)
    assert exists(savepath)

    dnafnames = datautils.getFileListContainingString(segmented_ch_folder, 'DNA_RPE.tif')
    actinfnames = datautils.getFileListContainingString(segmented_ch_folder, 'Actin_RPE.tif')
    GFPfnames = datautils.getFileListContainingString(segmented_ch_folder, '_GFP')

    dnafiles, actinfiles, GFPfiles, no_stacks= datautils.orderfilesbybasenames(dnafnames, actinfnames, GFPfnames, debug=False)
    channels = experimentalparams.getusedchannels(actinfiles)
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
    gfpstackindorientations = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxorganellepercell,
         3))  # for 3 dimensions/axes
    gfpstackzdistr = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxorganellepercell))
    gfpstackraddist2d = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxorganellepercell))
    gfpstackraddist3d = np.nan * np.ones(
        (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells, maxorganellepercell))

    num_processes = 8
    executor = ProcessPoolExecutor(max_workers=num_processes)
    processes = []

    minlength = 5
    binvals = np.arange(minlength)
    freq = binvals.copy()

    for stackid, (actinfile, dnafile, GFPfile) in enumerate(zip(actinfiles, dnafiles, GFPfiles)):
        processes = []
        try:
            start_ts = datetime.datetime.now()

            week, rep, w, r, fov, fovno, basename = datautils.getwr_3channel(dnafile, actinfile, GFPfile)
            t = experimentalparams.findtreatment(r)
            print(f"\nWeek:{week}, {w}\t|| Replicate: {rep}, {r}\t|| Treatment {t}\t|| Field of view: {fov}, {fovno}\t||Basename: {basename}")
            if w < usedweeks:

                ##GET IMAGES
                img_GFP = stackio.opensegmentedstack(join(segmented_ch_folder, GFPfile))
                img_ACTIN = stackio.opensegmentedstack(join(segmented_ch_folder, actinfile))  # binary=False
                img_DNA = stackio.opensegmentedstack(join(segmented_ch_folder, dnafile))  # binary=False
                # assert np.unique(img_GFP)==np.unique(img_ACTIN)== np.unique(img_DNA)
                assert img_GFP.shape == img_ACTIN.shape == img_DNA.shape
                print("TEST: all unique values should be equal", np.unique(img_GFP), np.unique(img_ACTIN),
                      np.unique(img_DNA))
                print("TEST: all dimensions  should be equal", img_GFP.shape, img_ACTIN.shape, img_DNA.shape)
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
                percent_actin_matrix = np.zeros((len(is_), len(js_)))  # percentage based on total actin overlap
                dna_actin_membership = np.zeros((len(is_), len(js_)), dtype=int)  # membership array
                over75list_actin = np.zeros(len(js_))  # over 75% overlap array
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
                            if percent_coverage_actin >= 0.75:  ##75% or more of dna lies in one actin
                                over75list_actin[j - 1] = i
                                dna_actin_membership[i - 1, j - 1] = 1
                bincount = np.bincount(np.sum(dna_actin_membership, axis=1), minlength=minlength)
                freq = np.vstack((freq, bincount[binvals]))  # number of dna per actin - can be used to plot frequency of membership
                print(f"Overlap calculations for {basename} done")
                # print( over75list_actin.shape, len(np.unique(over75list_actin)))
                # print(dna_actin_membership.shape, len(np.unique(dna_actin_membership)))
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

                    centroid, volume, xspan, yspan, zspan, maxferet, minferet, miparea = ShapeMetrics.calculate_object_properties(
                        objs[slices])
                    # print("calculated actin object")
                    cellvals = [centroid, volume, xspan, yspan, zspan, maxferet, minferet, miparea, top, bot]
                    if experimentalparams.checkcellconditions(cellvals) and datautils.checkfinitetemp(cellvals[1:8]):
                        ##DNA members
                        savethisimage = False
                        memberdnas = np.where(dna_actin_membership[obj_index - 1, :])[0]  # -1 since this matrix uses indices from 0
                        no_members = memberdnas.shape[0]
                        selected_dna_members = np.zeros_like(labelactin[slices])
                        actin_channel = 255 * ((labelactin == index) > 0)

                        if no_members > 0:
                            if False: # decide condition
                                savethisimage = True
                            cellstackvols[t, w, 0, r % 5, fovno, obj_index] = volume
                            cellstackxspan[t, w, 0, r % 5, fovno, obj_index] = xspan
                            cellstackyspan[t, w, 0, r % 5, fovno, obj_index] = yspan
                            cellstackzspan[t, w, 0, r % 5, fovno, obj_index] = zspan
                            cellstackmiparea[t, w, 0, r % 5, fovno, obj_index] = miparea
                            cellstackmaxferet[t, w, 0, r % 5, fovno, obj_index] = maxferet
                            cellstackminferet[t, w, 0, r % 5, fovno, obj_index] = minferet
                            for memberdna_no in range(no_members):
                                # print("DNAS",memberdna_no , no_members)
                                if memberdna_no >= 2:
                                    print(f"found member dna: {memberdna_no+1} of {no_members}")
                                    continue
                                dnaobj = (labeldna[slices] == (memberdnas[memberdna_no] + 1))
                                if savethisimage:
                                    selected_dna_members = selected_dna_members | 255 * dnaobj
                                    dna_channel = selected_dna_members & actin_channel
                                    together = (actin_channel + dna_channel) // 2
                                    temp_together = together[bbox_actin]
                                    out = np.expand_dims(temp_together, axis=(0, 1))
                                    out = out.astype(np.uint8)
                                    writer = omeTifWriter.OmeTifWriter(savepath + f'{savename}_Actin{stackid}_.tiff',
                                                                       overwrite_file=True)
                                    writer.save(out)
                                assert (dnaobj.shape == labeldna[slices].shape == labelactin[slices].shape)
                                centroid, dnavol, dnax, dnay, dnaz, dnamaxf, dnaminf, dnamip = ShapeMetrics.calculate_object_properties(
                                    dnaobj)
                                # centroid, volume, xspan, yspan, zspan, maxferet, minferet, miparea
                                dnastackvols[t, w, 0, r % 5, fovno, obj_index, memberdna_no] = dnavol
                                dnastackxspan[t, w, 0, r % 5, fovno, obj_index, memberdna_no] = dnax
                                dnastackyspan[t, w, 0, r % 5, fovno, obj_index, memberdna_no] = dnay
                                dnastackzspan[t, w, 0, r % 5, fovno, obj_index, memberdna_no] = dnaz
                                dnastackmiparea[t, w, 0, r % 5, fovno, obj_index, memberdna_no] = dnamip
                                dnastackmaxferet[t, w, 0, r % 5, fovno, obj_index, memberdna_no] = dnamaxf
                                dnastackminferet[t, w, 0, r % 5, fovno, obj_index, memberdna_no] = dnaminf
                        # GFP members
                        IMGGFP_obj = (img_GFP & objs)[slices]
                        # print("IMGGFP_obj:: ", IMGGFP_obj.shape)

                        # GFPcentroid, GFPvolume, GFPxspan, GFPyspan, GFPzspan, GFPmaxferet, GFPminferet, GFPmip = ShapeMetrics.calculate_object_properties(IMGGFP_obj[slices])
                        # GFPvals = [GFPvolume, GFPxspan, GFPyspan, GFPzspan, GFPmaxferet, GFPminferet, GFPmip]
                        # print("GFPvals:: ", GFPvals)
                        # if datautils.checkfinitetemp(GFPvals):
                        if doindividualcalcs:
                            # print("doing individual calcs")
                            processes.append((t, w, r, fovno, obj_index, executor.submit(ShapeMetrics.individualcalcs, IMGGFP_obj)))
                print("Processes = ", len(processes))

                # print(indorient3D.shape)
                # gfpstackindorientations[t, w, 0, r % 5, fovno, obj_id, :, 3] = indorient3D.T
                # gfpstackzdistr[t, w, 0, r % 5, fovno, obj_id, :] = z_dist
                # gfpstackraddist2d[t, w, 0, r % 5, fovno, obj_id, :] = radial_dist2d
                # gfpstackraddist3d[t, w, 0, r % 5, fovno, obj_id, :] = radial_dist3d
                #############
            for it, iw, ir, ifovno, obj_id, process in processes:
                # print('in processes')
                features = process.result()
                # indcentroid, indvolume, indxspan, indyspan, indzspan, indmaxferet, indminferet, indmiparea, indorient3D, z_dist, radial_dist2d, radial_dist3d = features
                indcentroid, indvolume, indxspan, indyspan, indzspan, indmaxferet, indminferet, indmiparea = features
                gfpstackvols[it,iw, 0, ir % 5, ifovno, obj_id, :] = indvolume
                gfpstackxspan[it, iw, 0, ir % 5, fovno, obj_id, :] = indxspan
                gfpstackyspan[it, iw, 0, ir % 5, fovno, obj_id, :] = indyspan
                gfpstackzspan[it, iw, 0, ir % 5, fovno, obj_id, :] = indzspan
                gfpstackmiparea[it, iw, 0, ir % 5, fovno, obj_id, :] = indmiparea
                gfpstackmaxferet[it, iw, 0, ir % 5, fovno, obj_id, :] = indmaxferet
                gfpstackminferet[it, iw, 0, ir % 5, fovno, obj_id, :] = indminferet
            end_ts = datetime.datetime.now()
            print("TOMvolvalues : ", np.count_nonzero(~np.isnan(gfpstackvols)))

        except Exception as e:
            print("Exception: ", e, traceback.format_exc(),"TOMvolvalues", np.count_nonzero(~np.isnan(gfpstackvols)))


        print(f"{basename} done in {str(end_ts - start_ts)}")

    allCELLvals = [cellstackvols, cellstackxspan, cellstackyspan, cellstackzspan, cellstackmiparea, cellstackmaxferet,
                   cellstackminferet]
    allDNAvals = [dnastackvols, dnastackxspan, dnastackyspan, dnastackzspan, dnastackmiparea, dnastackmaxferet,
                  dnastackminferet]
    allGFPvals = [gfpstackvols, gfpstackxspan, gfpstackyspan, gfpstackzspan, gfpstackmiparea, gfpstackmaxferet,
                  gfpstackminferet]#, gfpstackindorientations, gfpstackzdistr, gfpstackraddist2d, gfpstackraddist3d]

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
    uselog = [False, False, True]
    for o, (propertytype, otype) in enumerate(zip(propertycategory, orgenelletype)):
        for i, prop in enumerate(propertytype):
            if savedata:
                filename = f"{channel}_{otype}_{propnames[i]}_{strsigma}.npz"
                fpath = join(savepath, filename)
                stackio.saveproperty(prop, filepath=fpath, type="npz")
                loaded = stackio.loadproperty(fpath)

                success = datautils.array_nan_equal(loaded[loaded.files[0]], prop)
                if success:
                    print(f"SAVE SUCCESSFUL FOR {filename}\t\tNo. of Datapoints: {np.count_nonzero(~np.isnan(prop))}")
                else:  # (2, 4, 1, 5, 6, 1000, 50)
                    print(loaded.files, loaded[loaded.files[0]].shape, prop.shape)
            try:
                if generateplots:
                    plotter.violinstripplot(stackdata=prop, channel=otype, propname=propnames[i], units=unittype[i],
                                            savesigma=True, selected_method_type=["Stackwise"], uselog=uselog[o])
            except Exception as e:
                print(e)
    # tempcellvol = cellvolumes.copy()
