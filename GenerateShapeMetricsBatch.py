import datetime
import traceback
from concurrent.futures import ProcessPoolExecutor
from os import mkdir
from os.path import join, exists, isdir
import time
import click
import numpy as np
import pandas as pd
from scipy.ndimage import find_objects  # ,label

# from src.stackio import metadataHandler as meta
from src.AnalysisTools import experimentalparams, datautils, ShapeMetrics
from src.AnalysisTools.dtypes import PathLike
from src.Visualization import plotter, cellstack
from src.stackio import stackio


@click.command(options_metavar="<options>")
@click.option("--GFPFolder", default="../SegmentationAnalyzer/temp/",
              help="Folder containing segmented files for GFP channel",
              metavar="<PathLike>")
@click.option("--CellFolder", default="../SegmentationAnalyzer/temp/", metavar="<PathLike>",
              help="Folder containing co-registered segmented files for corresponding Actin and DNA channel")
@click.option("--savepath", default="../SegmentationAnalyzer/temp/", metavar="<PathLike>",
              help="Path to save measurements/results")
@click.option("--channel", default="None", metavar="<String>", help="Name of channel")
@click.option("--usesampledataonly", default=False, metavar="<Boolean>", help="Use only a sample of the data")
@click.option("--test", default=False, metavar="<Boolean>",
              help="test the code withouth actually doing any calculations")
@click.option("--dontsave", default=False, metavar="<Boolean>",
              help="Option to savedata. Be careful when setting this as data will not be saved if set to true")
@click.option("--generateplots", default=False, metavar="<Boolean>",
              help="Option to generate plots immediately after saving data")
@click.option("--debug", default=False, metavar="<Boolean>", help="Show extra information for debugging")
@click.option("--usednareference", default=False, metavar="<Boolean>", help="Use DNA as a reference instead of Cell")
@click.option("--num_processes", default=4, metavar="<int>", help="Use DNA as a reference instead of Cell")
# @click.option("--help", help="Show details for function ")
def calculateCellMetrics(gfpfolder: PathLike, cellfolder: PathLike, savepath: PathLike, channel: str,
                         usesampledataonly: bool, test: bool, dontsave: bool, generateplots: bool, debug: bool,
                         usednareference=False,
                         num_processes: int = 4):
    """
    Read all segmented image files. Measure shape metrics based on corresponding co-registered channels and save data for each metric.

    """
    print(f"Paths: gfp folder = {gfpfolder}\t cell folder = {cellfolder}\t savefolder = {savepath}")
    print((gfpfolder == "../SegmentationAnalyzer/temp/"), (cellfolder == "../SegmentationAnalyzer/temp/"),
          (savepath == "../SegmentationAnalyzer/temp/"))
    if (gfpfolder == "../SegmentationAnalyzer/temp/") or (cellfolder == "../SegmentationAnalyzer/temp/") or (
            savepath == "../SegmentationAnalyzer/temp/"):
        if not test:
            print("default path detected. Conducting test run")
            print(f"Paths: gfp folder = {gfpfolder}\t cell folder = {cellfolder}\t savefolder = {savepath}")
            test = True

    # Important: make sure this indent is maintained throughout
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

    savepathmeta = join(savepath, "meta")
    assert exists(gfpfolder), "GFP folder doesn't exist"
    assert exists(cellfolder), "cellfolder folder doesn't exist"
    assert exists(savepath), "Savepath folder doesn't exist"
    if not exists(savepathmeta):
        mkdir(savepathmeta)

    dnafnames = datautils.getFileListContainingString(cellfolder, '_DNA_RPE')
    actinfnames = datautils.getFileListContainingString(cellfolder, '_Actin_RPE')
    GFPfnames = datautils.getFileListContainingString(gfpfolder, '_GFP')

    dnafiles, actinfiles, GFPfiles, no_stacks = datautils.orderfilesbybasenames(dnafnames, actinfnames, GFPfnames,
                                                                                debug=False)
    chnls = experimentalparams.getusedchannels(actinfiles)
    no_chnls = len(chnls)
    badfiles = []

    cell = {}
    cell["shape"] = (usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells)
    cell["shape3d"] = (usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, 3)
    cell["Volumes"] = np.nan * np.ones(cell["shape"])
    cell["Centroids"] = np.nan * np.ones(cell["shape3d"])
    cell["xspan"] = np.nan * np.ones(cell["shape"])
    cell["yspan"] = np.nan * np.ones(cell["shape"])
    cell["zspan"] = np.nan * np.ones(cell["shape"])
    cell["miparea"] = np.nan * np.ones(cell["shape"])
    cell["maxferet"] = np.nan * np.ones(cell["shape"])
    cell["aspectratio2d"] = np.nan * np.ones(cell["shape"])
    cell["minferet"] = np.nan * np.ones(cell["shape"])
    cell["sphericity"] = np.nan * np.ones(cell["shape"])

    dna = {}
    dna["shape"] = (usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell)
    dna["shape3d"] = (usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell, 3)
    dna["Volumes"] = np.nan * np.ones(dna["shape"])
    dna["Centroids"] = np.nan * np.ones(dna["shape3d"])
    dna["xspan"] = np.nan * np.ones(dna["shape"])
    dna["yspan"] = np.nan * np.ones(dna["shape"])
    dna["zspan"] = np.nan * np.ones(dna["shape"])
    dna["miparea"] = np.nan * np.ones(dna["shape"])
    dna["maxferet"] = np.nan * np.ones(dna["shape"])
    dna["minferet"] = np.nan * np.ones(dna["shape"])
    dna["sphericity"] = np.nan * np.ones(dna["shape"])
    dna["aspectratio2d"] = np.nan * np.ones(dna["shape"])
    dna["volume_fraction"] = np.nan * np.ones(dna["shape"])
    dna["zdistr"] = np.nan * np.ones(dna["shape"])
    # dnastackinvaginationvfrac = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell))

    gfp = {}
    gfp["shape"] = (usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell)
    gfp["shape3d"] = (usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell, 3)
    gfp["Volumes"] = np.nan * np.ones(gfp["shape"])
    gfp["meanvols"] = np.nan * np.ones(gfp["shape"])
    gfp["Centroids"] = np.nan * np.ones(gfp["shape3d"])
    gfp["xspan"] = np.nan * np.ones(gfp["shape"])
    gfp["yspan"] = np.nan * np.ones(gfp["shape"])
    gfp["zspan"] = np.nan * np.ones(gfp["shape"])
    gfp["miparea"] = np.nan * np.ones(gfp["shape"])
    gfp["maxferet"] = np.nan * np.ones(gfp["shape"])
    gfp["minferet"] = np.nan * np.ones(gfp["shape"])
    gfp["aspectratio2d"] = np.nan * np.ones(gfp["shape"])
    gfp["orientations"] = np.nan * np.ones(gfp["shape3d"])
    gfp["cpc"] = np.nan * np.ones(cell["shape"])  # Note: Uses shape from cell
    gfp["volfrac"] = np.nan * np.ones(gfp["shape"])
    gfp["zdistr"] = np.nan * np.ones(gfp["shape"])
    gfp["raddist2d"] = np.nan * np.ones(gfp["shape"])
    gfp["raddist2dmean"] = np.nan * np.ones(gfp["shape"])
    gfp["raddist3d"] = np.nan * np.ones(gfp["shape"])
    max_pad_length = 6  # use value 1 more than final dilation
    for pl in range(max_pad_length):
        gfp[f"wallDist2dms{pl}"] = np.nan * np.ones(cell["shape"])
        gfp[f"wallDist2dSS{pl}"] = np.nan * np.ones(cell["shape"])
        gfp[f"wallDist3dms{pl}"] = np.nan * np.ones(cell["shape"])
        gfp[f"wallDist3dSS{pl}"] = np.nan * np.ones(cell["shape"])

    executor = ProcessPoolExecutor(max_workers=num_processes)
    processes = []

    minlength = 5
    binvals = np.arange(minlength)
    freq = binvals.copy()

    for stackid, (actinfile, dnafile, GFPfile) in enumerate(zip(actinfiles, dnafiles, GFPfiles)):
        processes = []
        if test:
            click.echo("Test run")
            break
        try:
            start_ts = datetime.datetime.now()

            week, rep, w, r, fov, fovno, basename = datautils.getwr_3channel(dnafile, actinfile, GFPfile)
            t = experimentalparams.find_treatment(r)
            if usesampledataonly:
                uss_fovs = [0, 1]
                uss_fovs = [1]
                uss_reps = [0, 1]  # will also do 5 and 6 respectively
                if not (fovno in uss_fovs and r % 5 in uss_reps):
                    print(f"Skipping {basename} since using a smaller set.")
                    continue
            print(
                f"\nWeek:{week}, {w}\t|| Replicate: {rep}, {r}\t|| Treatment {t}\t|| Field of view: {fov}, {fovno}\t||Basename: {basename}")
            if w < usedWs:

                ##GET LABELLED IMAGES
                GFPfilepath = join(gfpfolder, GFPfile)
                Actinfilepath = join(cellfolder, actinfile)
                DNAfilepath = join(cellfolder, dnafile)
                labelactin, labeldna = stackio.read_get_labelledstacks(Actinfilepath, DNAfilepath)
                img_GFP = stackio.opensegmentedstack(GFPfilepath)
                stackshape = img_GFP.shape
                print(f"img_GFP.shape: {stackshape} == labelactin.shape:"
                      f" {labelactin.shape} == labeldna.shape: {labeldna.shape}")
                is_ = list(np.unique(labelactin))
                js_ = list(np.unique(labeldna))
                is_.remove(0)
                js_.remove(0)

                savename = "_".join(actinfile.split("_")[:-2]) + "_connectmat"
                # savenameoverlaps = "_".join(actinfile.split("_")[:-2]) + "_overlapmat"
                print(f"starting overlap for {savename} \t actinlabels: ", len(is_), "dnalabels:", len(js_), flush=True)
                ######################################################################################################
                percent_actin_matrix = np.zeros((len(is_), len(js_)))  # percentage based on total actin overlap
                dna_actin_membership = np.zeros((len(is_), len(js_)), dtype=int)  # membership array
                over75list_actin = np.zeros(len(js_))  # over 75% overlap array
                ######################################################################################################
                for j in js_:  # in dna objects
                    selected_dna = (labeldna == j) > 0  # select DNA volume - only truth values
                    # print("SELECTED DNA", selected_dna.shape, np.unique(selected_dna), np.sum(selected_dna) )
                    labelactin_currentdna = labelactin[selected_dna]
                    # get all connected actin IDs
                    connected_actins = list(np.unique(labelactin_currentdna))
                    if 0 in connected_actins:  # not necessary since is_ doesnt have 0. Just an additional precaution
                        connected_actins.remove(0)
                    # calculate total overlap area for all actins with current DNA
                    combined_actins_volume_overlap = np.count_nonzero(labelactin_currentdna)
                    # print("SELECTED DNA", selected_dna.shape, np.unique(selected_dna), np.sum(selected_dna), actins_volume_overlap_total, connected_actins)
                    for i in is_:  # actin ids - this also removes 0 since its background
                        if i in connected_actins:
                            selected_actin_overlap_volume = np.count_nonzero(labelactin_currentdna == i)
                            # bboxcrop = find_objects(objs)
                            # slices = bboxcrop[0]
                            percent_dna_in_actin_i_rel = selected_actin_overlap_volume / combined_actins_volume_overlap
                            percent_actin_matrix[i - 1, j - 1] = percent_dna_in_actin_i_rel
                            if percent_dna_in_actin_i_rel >= 0.75:  ##75% or more of dna lies in one actin
                                over75list_actin[j - 1] = i
                                dna_actin_membership[i - 1, j - 1] = 1
                bincount = np.bincount(np.sum(dna_actin_membership, axis=1), minlength=minlength)
                # number of dna per actin - can be used to plot frequency of membership
                freq = np.vstack((freq, bincount[binvals]))
                print(f"Overlap calculations for {basename} done")
                if debug:
                    print(over75list_actin.shape, len(np.unique(over75list_actin)))
                    print(dna_actin_membership.shape, len(np.unique(dna_actin_membership)))
                    pd.DataFrame(dna_actin_membership).to_csv(savepath + "dnamem.csv")
                ## Now the calculation with membership data

                # len(is_) should be the same as unique cells +1
                cell_obj_df = pd.DataFrame(np.arange(1, len(is_) + 1, 1), columns=['cell_index'])
                for index, row in cell_obj_df.iterrows():
                    cellinputdict = {}
                    cell_index = int(row['cell_index'])
                    objs = labelactin == cell_index  # object with object label 'obj_index'
                    # get bounding box of actin
                    bbox_actin = find_objects(objs)
                    slices = bbox_actin[0]
                    edgetags, top, bot = ShapeMetrics.get_edge_connectivity(slices, objs.shape[0])
                    CellObject = objs[slices]
                    cellproperties = ShapeMetrics.calculate_object_properties(CellObject)
                    Ccentroid, Cvolume, Cxspan, Cyspan, Czspan, Cmaxferet, Cmeanferet, Cminferet, Cmiparea, Csphericity = cellproperties
                    cellattribute_testvals = [Ccentroid, Cvolume, Cxspan, Cyspan, Czspan, Cmaxferet, Cminferet,
                                              Cmiparea, top, bot]
                    biological_conditions_satisfied, cellcut = experimentalparams.cell_biologically_valid(
                        cellattribute_testvals)
                    # print("satisfied biological conditions?", biological_conditions_satisfied)
                    if biological_conditions_satisfied:  # and datautils.checkfinitetemp(cellattribute_testvals[1:8]):
                        ##DNA members
                        # savethisimage = False
                        # subtract 1 since this matrix uses indices from 0
                        memberdnas = np.where(dna_actin_membership[cell_index - 1, :])[0]
                        no_members = memberdnas.shape[0]
                        DNAObjects = np.zeros_like(labelactin[slices])
                        cell_cstack = 255 * ((labelactin == index) > 0)
                        # dna_cstack = None
                        # gfp_cstack = None
                        if no_members >= 0:  # include cells with no nuclei
                            # if False:  # decide condition
                            # savethisimage = True
                            cell["Volumes"][t, w, 0, r % 5, fovno, cell_index] = Cvolume
                            cell["xspan"][t, w, 0, r % 5, fovno, cell_index] = Cxspan
                            cell["yspan"][t, w, 0, r % 5, fovno, cell_index] = Cyspan
                            cell["zspan"][t, w, 0, r % 5, fovno, cell_index] = Czspan
                            cell["miparea"][t, w, 0, r % 5, fovno, cell_index] = Cmiparea
                            cell["sphericity"][t, w, 0, r % 5, fovno, cell_index] = Csphericity
                            cell["maxferet"][t, w, 0, r % 5, fovno, cell_index] = Cmaxferet
                            cell["minferet"][t, w, 0, r % 5, fovno, cell_index] = Cminferet
                            cell["Centroids"][t, w, 0, r % 5, fovno, cell_index] = Ccentroid
                            cell["aspectratio2d"][t, w, 0, r % 5, fovno, cell_index] = Cmaxferet / Cminferet
                            # meta.createcelldict()
                            extramembersnotify = True
                            for memberdna_no in range(no_members):
                                # print("DNAS",memberdna_no , no_members)

                                # select only DNA that were selected using membership rules
                                # - and only overlapping with cell object
                                dnaobj = (labeldna[slices] == (memberdnas[memberdna_no] + 1)) & CellObject
                                DNAObjects = DNAObjects | 255 * dnaobj

                                assert (dnaobj.shape == labeldna[slices].shape == labelactin[slices].shape)
                                Dcentroid, Dvolume, Dxspan, Dyspan, Dzspan, Dmaxferet, Dmeanferet, Dminferet, Dmiparea, Dsphericity = ShapeMetrics.calculate_object_properties(
                                    dnaobj)

                                if memberdna_no >= 2:
                                    if extramembersnotify:
                                        print(f"found more than 2 dna: {no_members}")
                                        extramembersnotify = False
                                    dnavol1 = dna["Volumes"][t, w, 0, r % 5, fovno, cell_index, 0]
                                    dnavol2 = dna["Volumes"][t, w, 0, r % 5, fovno, cell_index, 1]

                                    if Dvolume == dnavol1 or Dvolume == dnavol2:
                                        print(f"Current volume is the same size as one of the volumes. Skipping")
                                        continue
                                    originaldnas = (dnavol1, dnavol2, Dvolume)
                                    listdnas = list(originaldnas)
                                    listdnas.sort(reverse=True)
                                    # print(f"debug: {listdnas.index(Dvolume)}, {listdnas}, {originaldnas}")

                                    if listdnas.index(Dvolume) < 2:  # means current DNA is top 2 in volume
                                        dna_index = originaldnas.index(listdnas[-1])
                                        print(f"Found current volume {Dvolume} > old volume {listdnas[-1]}. "
                                              f"Replacing DNA at {dna_index} with current DNA. ")
                                    else:
                                        continue
                                else:
                                    dna_index = memberdna_no
                                dna_c_rel = Dcentroid - Ccentroid

                                dna["Centroids"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dcentroid
                                dna["Volumes"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dvolume
                                dna["xspan"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dxspan
                                dna["yspan"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dyspan
                                dna["zspan"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dzspan
                                dna["miparea"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dmiparea
                                dna["sphericity"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dsphericity
                                dna["maxferet"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dmaxferet
                                dna["minferet"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dminferet
                                dna["aspectratio2d"][
                                    t, w, 0, r % 5, fovno, cell_index, dna_index] = Dmaxferet / Dminferet
                                dna["volume_fraction"][
                                    t, w, 0, r % 5, fovno, cell_index, dna_index] = Dvolume / Cvolume * 100
                                dna["zdistr"][t, w, 0, r % 5, fovno, cell_index, dna_index] = dna_c_rel[0]
                        # GFP members
                        GFPObjects = (img_GFP[slices] & CellObject)  # Uses default slices

                        # NOTE: This is currently necessary for calculation of distance to wall metrics
                        for pad_length in range(max_pad_length):
                            d2w_slices, slicediffpad, shifted_slice_idx = ShapeMetrics.pad_3d_slice(slices, pad_length=pad_length, stackshape=stackshape)
                            # add phantom pad to mimic a equal dilation
                            gfp_bbox = ShapeMetrics.phantom_pad(img_GFP[d2w_slices], slicediffpad)
                            # print(f"padl:{pad_length}, d2w{d2w_slices}, "
                            #       f"sldiffpad{slicediffpad}, shift{shifted_slice_idx}")
                            cell_bbox = CellObject.copy()
                            # print(f"pad_length: {pad_length},  gfp_bbox: {gfp_bbox.shape}, cell_bbox shape: {cell_bbox.shape}, stackshape:{stackshape}")
                            if pad_length:
                                # Dilate bounding boxes for  cell to match organelle
                                cell_bbox = ShapeMetrics.dilate_bbox_uniform(CellObject, m=pad_length)
                                # Dilate boundary only for Cell - unnecessary with new method
                                dilated_cell_bbox = ShapeMetrics.dilate_boundary(cell_bbox, m=pad_length)
                                mask_gfp_bbox = gfp_bbox & dilated_cell_bbox
                                # cellstack.OmeTiffWriter.save(data=mask_gfp_bbox*1, uri=f"{savepath}/debug/mask_gfp_{basename}_{cell_index}_{t}_{w}_{r % 5}_{fovno}_{pad_length}.tiff", overwrite_file=True)
                                # cellstack.OmeTiffWriter.save(data=dilated_cell_bbox*1, uri=f"{savepath}/debug/dilatedcell_{basename}_{cell_index}_{t}_{w}_{r % 5}_{fovno}_{pad_length}.tiff", overwrite_file=True)
                                # cellstack.OmeTiffWriter.save(data=dilated_cell_bbox*1 -cell_bbox*1, uri=f"{savepath}/debug/dilation_{basename}_{cell_index}_{t}_{w}_{r % 5}_{fovno}_{pad_length}.tiff", overwrite_file=True)
                                # print(f"slicediffpad: {slicediffpad},  slices: {slices}, d2wslices: {d2w_slices}, shifted_slice_idx {shifted_slice_idx } ")
                                # print(f"mask_gfp_bbox: {mask_gfp_bbox.shape},  orig_gfp_bbox: {img_GFP[d2w_slices].shape }, dilated_cell_bbox: {dilated_cell_bbox.shape} ")
                            else:
                                mask_gfp_bbox = gfp_bbox & cell_bbox
                            # cellstack.OmeTiffWriter.save(data=cell_bbox*1, uri=f"{savepath}/debug/cellbbox{basename}_{cell_index}_{t}_{w}_{r % 5}_{fovno}_{pad_length}.tiff", overwrite_file=True)
                            # cellstack.OmeTiffWriter.save(data=gfp_bbox*1, uri=f"{savepath}/debug/gfp_bbox{basename}_{cell_index}_{t}_{w}_{r % 5}_{fovno}_{pad_length}.tiff", overwrite_file=True)

                            wall_dist_2d_m, wall_dist_2d_s = ShapeMetrics.distance_from_wall_2d(org_bbox=mask_gfp_bbox,
                                                                                                cell_bbox=cell_bbox)
                            wall_dist_3d_m, wall_dist_3d_s = ShapeMetrics.distance_from_wall_3d(org_bbox=mask_gfp_bbox,
                                                                                                cell_bbox=cell_bbox)
                            gfp[f"wallDist2dms{pad_length}"][t, w, 0, r % 5, fovno, cell_index] = wall_dist_2d_m
                            gfp[f"wallDist2dSS{pad_length}"][t, w, 0, r % 5, fovno, cell_index] = wall_dist_2d_s
                            gfp[f"wallDist3dms{pad_length}"][t, w, 0, r % 5, fovno, cell_index] = wall_dist_3d_m
                            gfp[f"wallDist3dSS{pad_length}"][t, w, 0, r % 5, fovno, cell_index] = wall_dist_3d_s
                        saveindividualcellstack = (np.random.random(1)[0] < 0.05)  # 5% sample ~~is_//10

                        if saveindividualcellstack:
                            cellstackfolder = join(savepath, 'cellstacks').replace("\\", "/")
                            if not isdir(cellstackfolder):
                                mkdir(cellstackfolder)
                            stackfilename = f"{channel}_{basename}_{cell_index}"
                            cellstack.mergestack(CellObject, DNAObjects, GFPObjects,
                                                 savename=join(cellstackfolder, stackfilename), save=True,
                                                 add_3d_cell_outline=False)
                        # print("shapes: ", CellObject.shape, DNAObjects.shape, GFPObjects.shape)
                        usednareference = False  # TODO
                        refcentroid = None

                        if usednareference:
                            refcentroid = Dcentroid
                        else:
                            refcentroid = Ccentroid

                        processes.append((t, w, r, fovno, cell_index, Cvolume, Cmeanferet,
                                          executor.submit(ShapeMetrics.calculate_multiorganelle_properties,
                                                          GFPObjects, refcentroid)))
                print("Processes = ", len(processes))

            for it, iw, ir, ifovno, cell_id, cvol, cmferet, process in processes:
                features = process.result()
                Gcount, Gcentroid, Gvolume, Gspan, Gyspan, Gzspan, Gmaxferet, Gmeanferet, Gminferet, Gmiparea, Gorient3D, Gz_dist, Gradial_dist2d, Gradial_dist3d, Gmeanvol = features
                # print("gcount:", Gcount)
                # print("indorient", indorient3D.shape, indorient3D.T.shape)
                gfp["cpc"][it, iw, 0, ir % 5, ifovno, cell_id] = Gcount
                gfp["meanvols"][it, iw, 0, ir % 5, ifovno, cell_id] = Gmeanvol
                gfp["Centroids"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gcentroid
                gfp["Volumes"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gvolume
                gfp["xspan"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gspan
                gfp["yspan"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gyspan
                gfp["zspan"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gzspan
                gfp["miparea"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gmiparea
                gfp["maxferet"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gmaxferet
                gfp["minferet"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gminferet
                gfp["aspectratio2d"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gmaxferet / Gminferet
                gfp["volfrac"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gvolume / cvol * 100
                gfp["orientations"][it, iw, 0, ir % 5, fovno, cell_id, :] = Gorient3D
                gfp["zdistr"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gz_dist
                gfp["raddist2d"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gradial_dist2d
                gfp["raddist2dmean"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gradial_dist2d / cmferet
                gfp["raddist3d"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gradial_dist3d

            end_ts = datetime.datetime.now()
            print(f"{basename} done in {str(end_ts - start_ts)}")

            # print(f"{channel}volvalues : {np.count_nonzero(~np.isnan(gfp["Volumes"]))}")
        except Exception as e:
            print("Exception: ", e, traceback.format_exc())

    if not test:
        time.sleep(120)

    allCellvals = [cell["Centroids"], cell["Volumes"], cell["xspan"], cell["yspan"], cell["zspan"], cell["miparea"],
                   cell["maxferet"], cell["minferet"], cell["aspectratio2d"], cell["sphericity"]]  ##
    cellpropnames = ["Centroid", "Volume", "X span", "Y span", "Z span", "MIP area", "Max feret", "Min feret",
                     "2D Aspect ratio", "Sphericity"]

    allDNAvals = [dna["Centroids"], dna["Volumes"], dna["xspan"], dna["yspan"], dna["zspan"], dna["miparea"],
                  dna["maxferet"], dna["minferet"], dna["aspectratio2d"], dna["volume_fraction"], dna["sphericity"],
                  dna["zdistr"]]
    DNApropnames = ["Centroid", "Volume", "X span", "Y span", "Z span", "MIP area", "Max feret", "Min feret",
                    "2D Aspect ratio", "Volume fraction", "Sphericity", "z-distribution"]
    # dnastackinvaginationvfrac
    allGFPvals = [gfp["Centroids"], gfp["Volumes"], gfp["xspan"], gfp["yspan"], gfp["zspan"], gfp["miparea"],
                  gfp["maxferet"], gfp["minferet"], gfp["aspectratio2d"], gfp["volfrac"], gfp["cpc"],
                  gfp["orientations"], gfp["zdistr"], gfp["raddist2d"], gfp["raddist2dmean"], gfp["raddist3d"],
                  gfp["meanvols"], gfp["wallDist2dms0"], gfp["wallDist2dSS0"], gfp["wallDist3dms0"],
                  gfp["wallDist3dSS0"], gfp["wallDist2dms1"], gfp["wallDist2dSS1"], gfp["wallDist3dms1"],
                  gfp["wallDist3dSS1"], gfp["wallDist2dms2"], gfp["wallDist2dSS2"], gfp["wallDist3dms2"],
                  gfp["wallDist3dSS2"], gfp["wallDist2dms3"], gfp["wallDist2dSS3"], gfp["wallDist3dms3"],
                  gfp["wallDist3dSS3"], gfp["wallDist2dms4"], gfp["wallDist2dSS4"], gfp["wallDist3dms4"],
                  gfp["wallDist3dSS4"], gfp["wallDist2dms5"], gfp["wallDist2dSS5"], gfp["wallDist3dms5"],
                  gfp["wallDist3dSS5"]]
    GFPpropnames = ["Centroid", "Volume", "X span", "Y span", "Z span", "MIP area", "Max feret", "Min feret",
                    "2D Aspect ratio", "Volume fraction", "Count per cell", "Orientation", "z-distribution",
                    "radial distribution 2D", "normalized radial distribution 2D", "radial distribution 3D",
                    "Mean Volume", "Mean 2D distance to wall d0", "Stdev 2D distance to wall d0",
                    "Mean 3D distance to wall d0",
                    "Stdev 3D distance to wall d0", "Mean 2D distance to wall d1", "Stdev 2D distance to wall d1",
                    "Mean 3D distance to wall d1",
                    "Stdev 3D distance to wall d1", "Mean 2D distance to wall d2", "Stdev 2D distance to wall d2",
                    "Mean 3D distance to wall d2",
                    "Stdev 3D distance to wall d2", "Mean 2D distance to wall d3", "Stdev 2D distance to wall d3",
                    "Mean 3D distance to wall d3",
                    "Stdev 3D distance to wall d3", "Mean 2D distance to wall d4", "Stdev 2D distance to wall d4",
                    "Mean 3D distance to wall d4",
                    "Stdev 3D distance to wall d4", "Mean 2D distance to wall d5", "Stdev 2D distance to wall d5",
                    "Mean 3D distance to wall d5",
                    "Stdev 3D distance to wall d5"]
    propnames = [cellpropnames, DNApropnames, GFPpropnames]
    # indGFPvals = indGFPcentroidhs, indGFPvolumes, indGFPzspans, indGFPxspans, indGFPyspans, indGFPmaxferets, indGFPminferets  # , indGFPorients
    withstrpplt = True
    sigma = 2
    strsigma = "95.45"

    orgenelletype = ["Cell", "DNA", channel]
    propertycategory = [allCellvals, allDNAvals, allGFPvals]
    # propertycategory_names = [allCELLvalnames,allDNAvalnames,allGFPvalnames]
    # for otype in orgenelletype:
    uselog = [False, False, True]
    for o, (propertytype, otype) in enumerate(zip(propertycategory, orgenelletype)):
        for i, prop in enumerate(propertytype):
            if not dontsave:
                propertyname = propnames[o][i]
                # filename = f"{channel}_{otype}_{propertyname}_{strsigma}.npz"
                filename = f"{channel}_{otype}_{propertyname}.npz"
                fpath = join(savepath, filename)
                stackio.saveproperty(prop, filepath=fpath, type="npz")
                loaded = stackio.loadproperty(fpath)
                success = stackio.checksavedfileintegrity(loaded, prop)

                # success = datautils.array_nan_equal(loaded[loaded.files[0]], prop)
                if success:
                    print(f"SAVE SUCCESSFUL FOR {filename}\t\tNo. of Datapoints: {np.count_nonzero(~np.isnan(prop))}")
                else:  # (2, 4, 1, 5, 6, 1000, 50)
                    print(loaded.files, loaded[loaded.files[0]].shape, prop.shape)
            try:
                if generateplots:
                    plotter.violinstripplot(stackdata=prop, channel=otype, propname=propnames[pl],
                                            units=experimentalparams.getunits(propertyname),
                                            percentile_include=True, selected_method_type=None, uselog=uselog[o])
            except Exception as e:
                print(e)
    for organelle in orgenelletype:
        print(f"converting organelle: {organelle}")
        stackio.convertfromnpz_allproperties(npzfolderpath=savepath, targetdir=join(savepath, "csv/"),
                                             organelle=organelle)


if __name__ == "__main__":
    import sys

    args = sys.argv
    print(f"args:{args})")
    # exit()
    # Example usage python <path-to-repo>/GenerateShapeMetricsBatch.py --
    calculateCellMetrics()
