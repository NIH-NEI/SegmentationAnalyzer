import datetime
import traceback
from concurrent.futures import ProcessPoolExecutor
from os import mkdir
from os.path import join, exists

import numpy as np
import pandas as pd
from scipy.ndimage.measurements import find_objects  # ,label
# from scipy.ndimage.measurements import label, find_objects
from skimage.measure import label as skilbl

# from src.stackio import metadataHandler as meta
from src.AnalysisTools import experimentalparams, datautils, ShapeMetrics
from src.Visualization import plotter, cellstack
from src.stackio import stackio

if __name__ == "__main__":
    # Important: make sure this indent is maintained throughout
    doindividualcalcs = True
    channel = "TOM20"
    ###############################################
    usedWs = experimentalparams.USEDWEEKS
    usedTs = experimentalparams.USEDTREATMENTS
    treatment = experimentalparams.TREATMENT_TYPES

    totalFs = experimentalparams.TOTALFIELDSOFVIEW
    usedwells = experimentalparams.USEDWELLS  # well numbers divided by treatment
    maxcells = experimentalparams.MAX_CELLS_PER_STACK
    maxdnapercell = experimentalparams.MAX_DNA_PER_CELL
    maxgfp_cell = experimentalparams.MAX_ORGANELLE_PER_CELL
    ###############################################

    # segmented_ch_folder = '../Results/2022/Jan21/TOM_stack_18img/segmented/TOM/'
    # segmented_ch_folder = '../Results/2022/Jan28/TOM/segmented/'
    # savepath = '../Results/2022/Feb4/TOM/all/'

    segmented_ch_folder = '../Results/2022/Jan21/TOM_stack_18img/segmented/TOM/'
    # savepath = '../Results/2022/Jan21/TOM_stack_18img/segmented/calcs/'
    # savepath = '../Results/2022/Jan28/TOM/TOM_calcs_test/'
    savepath = '../Results/2022/Feb25/TOM/results_cellstack/'
    savepathmeta = join(savepath, "meta")
    assert exists(segmented_ch_folder)
    assert exists(savepath)
    if not exists(savepathmeta):
        mkdir(savepathmeta)

    dnafnames = datautils.getFileListContainingString(segmented_ch_folder, 'DNA_RPE.tif')
    actinfnames = datautils.getFileListContainingString(segmented_ch_folder, 'Actin_RPE.tif')
    GFPfnames = datautils.getFileListContainingString(segmented_ch_folder, '_GFP')

    dnafiles, actinfiles, GFPfiles, no_stacks = datautils.orderfilesbybasenames(dnafnames, actinfnames, GFPfnames,
                                                                                debug=False)
    chnls = experimentalparams.getusedchannels(actinfiles)
    no_chnls = len(chnls)
    # print("USEDCHANNELS: ", usedchannels)
    badfiles = []
    # Cell = {}
    # DNA = {}
    # GFP = {}
    # Organelles = [Cell,DNA, GFP]
    # Properties = ["Volume", "Centroids","Xspan"]
    # dims = (usedtreatments, usedweeks, usedchannels, usedwells, totalFs, maxnocells) # depends on organelle
    # for Organelle in Organelles:
    #     for propertyname in properties:
    #         Organelle[propertyname] = np.nan *

    cellstackvols = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells))
    cellstackcentroids = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, 3))
    cellstackxspan = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells))
    cellstackyspan = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells))
    cellstackzspan = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells))
    cellstackmiparea = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells))
    cellstackmaxferet = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells))
    cellstackaspectratio2d = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells))
    cellstackminferet = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells))
    cellstacksphericity = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells))

    dnastackvols = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell))
    dnastackcentroids = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell, 3))
    dnastackxspan = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell))
    dnastackyspan = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell))
    dnastackzspan = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell))
    dnastackmiparea = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell))
    dnastackmaxferet = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell))
    dnastackminferet = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell))
    dnastacksphericity = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell))
    dnastackaspectratio2d = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell))
    dnastackvolfraction = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell))
    dnastackinvaginationvfrac = np.nan * np.ones(
        (usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell))

    gfpstackvols = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell))
    gfpstackmeanvols = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells))
    gfpstackcentroids = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell, 3))
    gfpstackxspan = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell))
    gfpstackyspan = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell))
    gfpstackzspan = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell))
    gfpstackmiparea = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell))
    gfpstackmaxferet = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell))
    gfpstackminferet = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell))
    gfpstackaspectratio2d = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell))
    gfpstackindorientations = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell, 3))
    gfpstackcpc = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells))
    gfpstackvolfrac = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell))
    gfpstackzdistr = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell))
    gfpstackraddist2d = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell))
    gfpstackraddist2dmean = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell))
    gfpstackraddist3d = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell))

    num_processes = 4
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
            print(
                f"\nWeek:{week}, {w}\t|| Replicate: {rep}, {r}\t|| Treatment {t}\t|| Field of view: {fov}, {fovno}\t||Basename: {basename}")
            if w < usedWs:

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
                # number of dna per actin - can be used to plot frequency of membership
                freq = np.vstack((freq, bincount[binvals]))
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
                    CellObject = objs[slices]
                    cellproperties = ShapeMetrics.calculate_object_properties(CellObject)
                    Ccentroid, Cvolume, Cxspan, Cyspan, Czspan, Cmaxferet, Cmeanferet, Cminferet, Cmiparea, Csphericity = cellproperties
                    # print("calculated actin object")
                    cellattribute_testvals = [Ccentroid, Cvolume, Cxspan, Cyspan, Czspan, Cmaxferet, Cminferet, Cmiparea, top, bot]
                    biological_conditions_satisfied, cellcut = experimentalparams.checkcellconditions(cellattribute_testvals)
                    # print("satisfied biological conditions?", biological_conditions_satisfied)
                    if biological_conditions_satisfied:  # and datautils.checkfinitetemp(cellattribute_testvals[1:8]):
                        ##DNA members
                        savethisimage = False
                        # subtract 1 since this matrix uses indices from 0
                        memberdnas = np.where(dna_actin_membership[obj_index - 1, :])[0]
                        no_members = memberdnas.shape[0]
                        DNAObjects = np.zeros_like(labelactin[slices])
                        cell_cstack = 255 * ((labelactin == index) > 0)
                        dna_cstack = None
                        gfp_cstack = None
                        if no_members > 0:
                            if False:  # decide condition
                                savethisimage = True
                            cellstackvols[t, w, 0, r % 5, fovno, obj_index] = Cvolume
                            cellstackxspan[t, w, 0, r % 5, fovno, obj_index] = Cxspan
                            cellstackyspan[t, w, 0, r % 5, fovno, obj_index] = Cyspan
                            cellstackzspan[t, w, 0, r % 5, fovno, obj_index] = Czspan
                            cellstackmiparea[t, w, 0, r % 5, fovno, obj_index] = Cmiparea
                            cellstacksphericity[t, w, 0, r % 5, fovno, obj_index] = Csphericity
                            cellstackmaxferet[t, w, 0, r % 5, fovno, obj_index] = Cmaxferet
                            cellstackminferet[t, w, 0, r % 5, fovno, obj_index] = Cminferet
                            cellstackcentroids[t, w, 0, r % 5, fovno, obj_index] = Ccentroid
                            cellstackaspectratio2d[t, w, 0, r % 5, fovno, obj_index] = Cminferet / Cmaxferet
                            # meta.createcelldict()
                            for memberdna_no in range(no_members):
                                # print("DNAS",memberdna_no , no_members)
                                if memberdna_no >= 2:
                                    print(f"found member dna: {memberdna_no + 1} of {no_members}")
                                    continue
                                # select only DNA that were selected using membership rules
                                dnaobj = (labeldna[slices] == (memberdnas[memberdna_no] + 1))
                                DNAObjects = DNAObjects | 255 * dnaobj

                                assert (dnaobj.shape == labeldna[slices].shape == labelactin[slices].shape)
                                Dcentroid, Dvolume, Dxspan, Dyspan, Dzspan, Dmaxferet, Dmeanferet, Dminferet, Dmiparea, Dsphericity = ShapeMetrics.calculate_object_properties(
                                    dnaobj)
                                dnastackcentroids[t, w, 0, r % 5, fovno, obj_index, memberdna_no] = Dcentroid
                                dnastackvols[t, w, 0, r % 5, fovno, obj_index, memberdna_no] = Dvolume
                                dnastackxspan[t, w, 0, r % 5, fovno, obj_index, memberdna_no] = Dxspan
                                dnastackyspan[t, w, 0, r % 5, fovno, obj_index, memberdna_no] = Dyspan
                                dnastackzspan[t, w, 0, r % 5, fovno, obj_index, memberdna_no] = Dzspan
                                dnastackmiparea[t, w, 0, r % 5, fovno, obj_index, memberdna_no] = Dmiparea
                                dnastacksphericity[t, w, 0, r % 5, fovno, obj_index, memberdna_no] = Dsphericity
                                dnastackmaxferet[t, w, 0, r % 5, fovno, obj_index, memberdna_no] = Dmaxferet
                                dnastackminferet[t, w, 0, r % 5, fovno, obj_index, memberdna_no] = Dminferet
                                dnastackaspectratio2d[
                                    t, w, 0, r % 5, fovno, obj_index, memberdna_no] = Dminferet / Dmaxferet
                                dnastackvolfraction[
                                    t, w, 0, r % 5, fovno, obj_index, memberdna_no] = Dvolume / Cvolume * 100
                                # if savethisimage:
                                #     together = (cell_cstack + dna_cstack) // 2
                                #     temp_together = together[bbox_actin]
                                #     out = np.expand_dims(temp_together, axis=(0, 1))
                                #     out = out.astype(np.uint8)
                                #     writer = OmeTifWriter.save(savepath + f'{savename}_Actin{stackid}_.tiff',
                                #                                        overwrite_file=True)
                        # GFP members
                        GFPObjects = (img_GFP[slices] & CellObject)

                        # saveindividualcellstack = (np.random.random(1)[0] < 0.1) #10% sample ~~is_//10
                        saveindividualcellstack = True #10% sample ~~is_//10
                        if saveindividualcellstack:
                            stackfilename = f"{channel}_{basename}_{obj_index}"
                            cellstack.mergestack(CellObject, DNAObjects, GFPObjects, savename = join(savepath, stackfilename), save = True)
                        # print("shapes: ", CellObject.shape, DNAObjects.shape, GFPObjects.shape)
                        if doindividualcalcs:
                            processes.append((t, w, r, fovno, obj_index, Cvolume,executor.submit(ShapeMetrics.calculate_multiorganelle_properties, GFPObjects)))
                            # print("doing individual calcs")
                            # Gcount, Gcentroid, Gvolume, Gspan, Gyspan, Gzspan, Gmaxferet, Gminferet, Gmiparea, Gorient3D, Gz_dist, Gradial_dist2d, Gradial_dist3d = ShapeMetrics.calculate_multiorganelle_properties((GFPObjects))
                            # it = t
                            # iw=w
                            # ir = r
                            # ifovno = fovno
                            # obj_id = obj_index
                            # gfpstackcentroids[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gcentroid
                            # gfpstackvols[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gvolume
                            # gfpstackxspan[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gspan
                            # gfpstackyspan[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gyspan
                            # gfpstackzspan[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gzspan
                            # gfpstackmiparea[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gmiparea
                            # gfpstackmaxferet[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gmaxferet
                            # gfpstackminferet[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gminferet
                            # gfpstackaspectratio2d[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gminferet / Gmaxferet
                            # gfpstackvolfrac[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gvolume / Cvolume * 100
                            # gfpstackcpc[it, iw, 0, ir % 5, ifovno, :] = Gcount
                            # # print("indorient", indorient3D.shape, indorient3D.T.shape)
                            # gfpstackindorientations[it, iw, 0, ir % 5, fovno, obj_id, :] = Gorient3D
                            # gfpstackzdistr[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gz_dist
                            # gfpstackraddist2d[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gradial_dist2d
                            # gfpstackraddist3d[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gradial_dist3d
                print("Processes = ", len(processes))

            for it, iw, ir, ifovno, obj_id, cvol, process in processes:
                features = process.result()
                Gcount, Gcentroid, Gvolume, Gspan, Gyspan, Gzspan, Gmaxferet, Gmeanferet, Gminferet, Gmiparea, Gorient3D, Gz_dist, Gradial_dist2d, Gradial_dist3d, Gmeanvol = features
                # print("gcount:", Gcount)
                # print("indorient", indorient3D.shape, indorient3D.T.shape)
                gfpstackcpc[it, iw, 0, ir % 5, ifovno, obj_id] = Gcount
                gfpstackmeanvols[it, iw, 0, ir % 5, ifovno, obj_id] = Gmeanvol
                gfpstackcentroids[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gcentroid
                gfpstackvols[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gvolume
                gfpstackxspan[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gspan
                gfpstackyspan[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gyspan
                gfpstackzspan[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gzspan
                gfpstackmiparea[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gmiparea
                gfpstackmaxferet[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gmaxferet
                gfpstackminferet[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gminferet
                gfpstackaspectratio2d[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gminferet / Gmaxferet
                gfpstackvolfrac[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gvolume / cvol * 100
                gfpstackindorientations[it, iw, 0, ir % 5, fovno, obj_id, :] = Gorient3D
                gfpstackzdistr[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gz_dist
                gfpstackraddist2d[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gradial_dist2d
                gfpstackraddist2dmean[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gradial_dist2d/Cmeanferet
                gfpstackraddist3d[it, iw, 0, ir % 5, ifovno, obj_id, :] = Gradial_dist3d
            end_ts = datetime.datetime.now()
            print(f"{basename} done in {str(end_ts - start_ts)}")

            print(f"{channel}volvalues : {np.count_nonzero(~np.isnan(gfpstackvols))}")
        except Exception as e:
            print("Exception: ", e, traceback.format_exc())

    allCellvals = [cellstackcentroids, cellstackvols, cellstackxspan, cellstackyspan, cellstackzspan, cellstackmiparea,
                   cellstackmaxferet, cellstackminferet, cellstackaspectratio2d, cellstacksphericity]  ##
    cellpropnames = ["Centroid", "Volume", "X span", "Y span", "Z span", "MIP area", "Max feret", "Min feret",
                     "2D Aspect ratio", "Sphericity"]

    allDNAvals = [dnastackcentroids, dnastackvols, dnastackxspan, dnastackyspan, dnastackzspan, dnastackmiparea,
                  dnastackmaxferet, dnastackminferet, dnastackaspectratio2d, dnastackvolfraction, dnastacksphericity]
    DNApropnames = ["Centroid", "Volume", "X span", "Y span", "Z span", "MIP area", "Max feret", "Min feret",
                    "2D Aspect ratio", "Volume fraction", "Sphericity"]
    # dnastackinvaginationvfrac
    allGFPvals = [gfpstackcentroids, gfpstackvols, gfpstackxspan, gfpstackyspan, gfpstackzspan, gfpstackmiparea,
                  gfpstackmaxferet, gfpstackminferet, gfpstackaspectratio2d, gfpstackvolfrac, gfpstackcpc,
                  gfpstackindorientations, gfpstackzdistr, gfpstackraddist2d, gfpstackraddist2dmean, gfpstackraddist3d, gfpstackmeanvols]
    GFPpropnames = ["Centroid", "Volume", "X span", "Y span", "Z span", "MIP area", "Max feret", "Min feret",
                    "2D Aspect ratio", "Volume fraction", "Count per cell", "Orientation", "z-distribution",
                    "radial distribution 2D","normalized radial distribution 2D", "radial distribution 3D", "Mean Volume"]
    propnames = [cellpropnames, DNApropnames, GFPpropnames]
    # indGFPvals = indGFPcentroidhs, indGFPvolumes, indGFPzspans, indGFPxspans, indGFPyspans, indGFPmaxferets, indGFPminferets  # , indGFPorients
    withstrpplt = True
    sigma = 2
    strsigma = "95.45"

    orgenelletype = ["Cell", "DNA", channel]
    propertycategory = [allCellvals, allDNAvals, allGFPvals]
    # propertycategory_names = [allCELLvalnames,allDNAvalnames,allGFPvalnames]
    generateplots, savedata = False, True
    # for otype in orgenelletype:
    uselog = [False, False, True]
    for o, (propertytype, otype) in enumerate(zip(propertycategory, orgenelletype)):
        for i, prop in enumerate(propertytype):
            if savedata:
                propertyname = propnames[o][i]
                filename = f"{channel}_{otype}_{propertyname}_{strsigma}.npz"
                fpath = join(savepath, filename)
                stackio.saveproperty(prop, filepath=fpath, type="npz")
                loaded = stackio.loadproperty(fpath)
                success = checksavedfileintegrity(loaded, prop)
                # success = datautils.array_nan_equal(loaded[loaded.files[0]], prop)
                if success:
                    print(f"SAVE SUCCESSFUL FOR {filename}\t\tNo. of Datapoints: {np.count_nonzero(~np.isnan(prop))}")
                else:  # (2, 4, 1, 5, 6, 1000, 50)
                    print(loaded.files, loaded[loaded.files[0]].shape, prop.shape)
            try:
                if generateplots:
                    plotter.violinstripplot(stackdata=prop, channel=otype, propname=propnames[i],
                                            units=experimentalparams.getunits(propertyname),
                                            savesigma=True, selected_method_type=None, uselog=uselog[o])
            except Exception as e:
                print(e)
