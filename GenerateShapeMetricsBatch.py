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

    dna = {}
    dna["shape"] = (usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell)
    dna["shape3d"] = (usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell, 3)

    gfp = {}
    gfp["shape"] = (usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell)
    gfp["shape3d"] = (usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxgfp_cell, 3)
    cellkeys = ["Volume", "Centroid", "X span", "Y span", "Z span", "MIP area", "2D Aspect ratio", "Min feret",
                "Mean feret", "Max feret", "Sphericity"]
    dnakeys = ["Volume", "Centroid", "X span", "Y span", "Z span", "MIP area", "2D Aspect ratio", "Min feret",
               "Mean feret", "Max feret", "Sphericity", "Volume fraction", "z-distribution"]
    gfpkeys = ["Volume", "Mean Volume", "Centroid", "X span", "Y span", "Z span", "MIP area", "2D Aspect ratio",
               "Min feret", "Mean feret", "Max feret", "Sphericity", "Volume fraction", "z-distribution", "Orientation",
               "Count per cell", "radial distribution 2D", "radial distribution 3D", "radial distribution 2Dmean"]
    # dnastackinvaginationvfrac = np.nan * np.ones((usedTs, usedWs, no_chnls, usedwells, totalFs, maxcells, maxdnapercell))

    max_pad_length = 6  # use value 1 more than final dilation
    for pl in range(max_pad_length):
        # TODO: Wait for Davide 
        cell["Volume"] = np.nan * np.ones(cell["shape"])
        cell["Centroid"] = np.nan * np.ones(cell["shape3d"])
        cell["X span"] = np.nan * np.ones(cell["shape"])
        cell["Y span"] = np.nan * np.ones(cell["shape"])
        cell["Z span"] = np.nan * np.ones(cell["shape"])
        cell["MIP area"] = np.nan * np.ones(cell["shape"])
        cell["2D Aspect ratio"] = np.nan * np.ones(cell["shape"])
        cell["Min feret"] = np.nan * np.ones(cell["shape"])
        cell["Mean feret"] = np.nan * np.ones(cell["shape"])
        cell["Max feret"] = np.nan * np.ones(cell["shape"])
        cell["Sphericity"] = np.nan * np.ones(cell["shape"])

        dna["Volume"] = np.nan * np.ones(dna["shape"])
        dna["Centroid"] = np.nan * np.ones(dna["shape3d"])
        dna["X span"] = np.nan * np.ones(dna["shape"])
        dna["Y span"] = np.nan * np.ones(dna["shape"])
        dna["Z span"] = np.nan * np.ones(dna["shape"])
        dna["MIP area"] = np.nan * np.ones(dna["shape"])
        dna["Min feret"] = np.nan * np.ones(dna["shape"])
        dna["Mean feret"] = np.nan * np.ones(dna["shape"])
        dna["Max feret"] = np.nan * np.ones(dna["shape"])
        dna["Sphericity"] = np.nan * np.ones(dna["shape"])
        dna["2D Aspect ratio"] = np.nan * np.ones(dna["shape"])
        dna["Volume fraction"] = np.nan * np.ones(dna["shape"])
        dna["z-distribution"] = np.nan * np.ones(dna["shape"])

        gfp["Volume"] = np.nan * np.ones(gfp["shape"])
        gfp["Mean Volume"] = np.nan * np.ones(gfp["shape"])
        gfp["Centroid"] = np.nan * np.ones(gfp["shape3d"])
        gfp["X span"] = np.nan * np.ones(gfp["shape"])
        gfp["Y span"] = np.nan * np.ones(gfp["shape"])
        gfp["Z span"] = np.nan * np.ones(gfp["shape"])
        gfp["MIP area"] = np.nan * np.ones(gfp["shape"])
        gfp["Min feret"] = np.nan * np.ones(gfp["shape"])
        gfp["Mean feret"] = np.nan * np.ones(gfp["shape"])
        gfp["Max feret"] = np.nan * np.ones(gfp["shape"])
        gfp["2D Aspect ratio"] = np.nan * np.ones(gfp["shape"])
        gfp["Orientation"] = np.nan * np.ones(gfp["shape3d"])
        gfp["Count per cell"] = np.nan * np.ones(cell["shape"])  # Note: Uses shape from cell
        gfp["Volume fraction"] = np.nan * np.ones(gfp["shape"])
        gfp["z-distribution"] = np.nan * np.ones(gfp["shape"])
        gfp["radial distribution 2D"] = np.nan * np.ones(gfp["shape"])
        gfp["radial distribution 2Dmean"] = np.nan * np.ones(gfp["shape"])
        gfp["radial distribution 3D"] = np.nan * np.ones(gfp["shape"])

        gfp[f"Mean 2D distance to wall d{pl}"] = np.nan * np.ones(cell["shape"])
        gfp[f"Mean3D distance to wall d{pl}"] = np.nan * np.ones(cell["shape"])
        gfp[f"Stdev 3D distance to wall d{pl}"] = np.nan * np.ones(cell["shape"])
        gfp[f"Stdev 2D distance to wall d{pl}"] = np.nan * np.ones(cell["shape"])
        gfp[f"Mean Bottom z-distance d{pl}"] = np.nan * np.ones(cell["shape"])
        gfp[f"Stdev Bottom z-distance d{pl}"] = np.nan * np.ones(cell["shape"])
        gfp[f"Mean Top z-distance d{pl}"] = np.nan * np.ones(cell["shape"])
        gfp[f"Stdev Top z-distance d{pl}"] = np.nan * np.ones(cell["shape"])
        gfp[f"Mean Bottom surface z-distance d{pl}"] = np.nan * np.ones(cell["shape"])
        gfp[f"Stdev Bottom surface z-distance d{pl}"] = np.nan * np.ones(cell["shape"])
        gfp[f"Mean Top surface z-distance d{pl}"] = np.nan * np.ones(cell["shape"])
        gfp[f"Stdev Top surface z-distance d{pl}"] = np.nan * np.ones(cell["shape"])

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
                    if 0 in connected_actins:  # not necessary since is_ doesn't have 0. Just an additional precaution
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
                            cell["Volume"][t, w, 0, r % 5, fovno, cell_index] = Cvolume
                            cell["X span"][t, w, 0, r % 5, fovno, cell_index] = Cxspan
                            cell["Y span"][t, w, 0, r % 5, fovno, cell_index] = Cyspan
                            cell["Z span"][t, w, 0, r % 5, fovno, cell_index] = Czspan
                            cell["MIP area"][t, w, 0, r % 5, fovno, cell_index] = Cmiparea
                            cell["Sphericity"][t, w, 0, r % 5, fovno, cell_index] = Csphericity
                            cell["Min feret"][t, w, 0, r % 5, fovno, cell_index] = Cminferet
                            cell["Mean feret"][t, w, 0, r % 5, fovno, cell_index] = Cmeanferet
                            cell["Max feret"][t, w, 0, r % 5, fovno, cell_index] = Cmaxferet
                            cell["Centroid"][t, w, 0, r % 5, fovno, cell_index] = Ccentroid
                            cell["2D Aspect ratio"][t, w, 0, r % 5, fovno, cell_index] = Cmaxferet / Cminferet
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
                                    dnavol1 = dna["Volume"][t, w, 0, r % 5, fovno, cell_index, 0]
                                    dnavol2 = dna["Volume"][t, w, 0, r % 5, fovno, cell_index, 1]

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

                                dna["Centroid"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dcentroid
                                dna["Volume"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dvolume
                                dna["X span"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dxspan
                                dna["Y span"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dyspan
                                dna["Z span"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dzspan
                                dna["MIP area"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dmiparea
                                dna["Sphericity"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dsphericity
                                dna["Min feret"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dminferet
                                dna["Mean feret"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dmeanferet
                                dna["Max feret"][t, w, 0, r % 5, fovno, cell_index, dna_index] = Dmaxferet
                                dna["2D Aspect ratio"][
                                    t, w, 0, r % 5, fovno, cell_index, dna_index] = Dmaxferet / Dminferet
                                dna["Volume fraction"][
                                    t, w, 0, r % 5, fovno, cell_index, dna_index] = Dvolume / Cvolume * 100
                                dna["z-distribution"][t, w, 0, r % 5, fovno, cell_index, dna_index] = dna_c_rel[0]
                        # GFP members
                        GFPObjects = (img_GFP[slices] & CellObject)  # Uses default slices

                        # NOTE: This is currently necessary for calculation of distance to wall metrics
                        usednareference = False  # TODO
                        refcentroid = None

                        if usednareference:
                            refcentroid = Dcentroid
                        else:
                            refcentroid = Ccentroid
                        for pad_length in range(max_pad_length):
                            # condition needed as dilations less than one are considered equivalent to infinity
                            d2w_slices, slicediffpad, shifted_slice_idx = ShapeMetrics.pad_3d_slice(slices,
                                                                                                    pad_length=pad_length,
                                                                                                    stackshape=stackshape)
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
                                dilated_cell_bbox = ShapeMetrics.dilate_boundary_zxy(cell_bbox, dilatexyonly=True,
                                                                                     m=pad_length)
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

                            z_dists_bot_mean, z_dists_bot_std, z_dists_top_mean, z_dists_top_std = ShapeMetrics.z_dist_top_bottom_extrema(
                                org_bbox=mask_gfp_bbox.copy(), cell_bbox=cell_bbox.copy())

                            gfp[f"Mean Bottom z-distance d{pad_length}"][
                                t, w, 0, r % 5, fovno, cell_index] = z_dists_bot_mean
                            gfp[f"Stdev Bottom z-distance d{pad_length}"][
                                t, w, 0, r % 5, fovno, cell_index] = z_dists_bot_std
                            gfp[f"Mean Top z-distance d{pad_length}"][
                                t, w, 0, r % 5, fovno, cell_index] = z_dists_top_mean
                            gfp[f"Stdev Top z-distance d{pad_length}"][
                                t, w, 0, r % 5, fovno, cell_index] = z_dists_top_std

                            z_dists_bot_surface_mean, z_dists_bot_surface_std, z_dists_top_surface_mean, z_dists_top_surface_std = ShapeMetrics.z_dist_top_bottom_surface(
                                org_bbox=mask_gfp_bbox.copy(), cell_bbox=cell_bbox.copy())

                            gfp[f"Mean Bottom surface z-distance d{pad_length}"][
                                t, w, 0, r % 5, fovno, cell_index] = z_dists_bot_surface_mean
                            gfp[f"Stdev Bottom surface z-distance d{pad_length}"][
                                t, w, 0, r % 5, fovno, cell_index] = z_dists_bot_surface_std
                            gfp[f"Mean Top surface z-distance d{pad_length}"][
                                t, w, 0, r % 5, fovno, cell_index] = z_dists_top_surface_mean
                            gfp[f"Stdev Top surface z-distance d{pad_length}"][
                                t, w, 0, r % 5, fovno, cell_index] = z_dists_top_surface_std

                            wall_dist_2d_m, wall_dist_2d_s = ShapeMetrics.distance_from_wall_2d(
                                org_bbox=mask_gfp_bbox.copy(),
                                cell_bbox=cell_bbox.copy())
                            wall_dist_3d_m, wall_dist_3d_s = ShapeMetrics.distance_from_wall_3d(
                                org_bbox=mask_gfp_bbox.copy(),
                                cell_bbox=cell_bbox.copy())
                            gfp[f"Mean 2D distance to wall d{pad_length}"][
                                t, w, 0, r % 5, fovno, cell_index] = wall_dist_2d_m
                            gfp[f"Stdev 2D distance to wall d{pad_length}"][
                                t, w, 0, r % 5, fovno, cell_index] = wall_dist_2d_s
                            gfp[f"Mean3D distance to wall d{pad_length}"][
                                t, w, 0, r % 5, fovno, cell_index] = wall_dist_3d_m
                            gfp[f"Stdev 3D distance to wall d{pad_length}"][
                                t, w, 0, r % 5, fovno, cell_index] = wall_dist_3d_s
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

                            processes.append((t, w, r, fovno, cell_index, Cvolume, Cmeanferet,
                                              executor.submit(ShapeMetrics.calculate_multiorganelle_properties,
                                                              GFPObjects, refcentroid)))
                        # print(gfp[f"Mean Bottom surface z-distance d{pad_length}"][t, w, 0, r % 5, fovno, cell_index], end="\t")
                        # print(gfp[f"Stdev Bottom surface z-distance d{pad_length}"][ t, w, 0, r % 5, fovno, cell_index], end="\t")
                        # print(gfp[f"Mean Top surface z-distance d{pad_length}"][ t, w, 0, r % 5, fovno, cell_index], end="\t")
                        # print(gfp[f"Stdev Top surface z-distance d{pad_length}"][ t, w, 0, r % 5, fovno, cell_index], end="\t")
                print("Processes = ", len(processes))

            for it, iw, ir, ifovno, cell_id, cvol, cmferet, process in processes:
                features = process.result()
                Gcount, Gcentroid, Gvolume, Gspan, Gyspan, Gzspan, Gmaxferet, Gmeanferet, Gminferet, Gmiparea, Gorient3D, Gz_dist, Gradial_dist2d, Gradial_dist3d, Gmeanvol = features
                # print("gcount:", Gcount)
                # print("indorient", indorient3D.shape, indorient3D.T.shape)
                gfp["Count per cell"][it, iw, 0, ir % 5, ifovno, cell_id] = Gcount
                gfp["Mean Volume"][it, iw, 0, ir % 5, ifovno, cell_id] = Gmeanvol
                gfp["Centroid"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gcentroid
                gfp["Volume"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gvolume
                gfp["X span"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gspan
                gfp["Y span"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gyspan
                gfp["Z span"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gzspan
                gfp["MIP area"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gmiparea
                gfp["Min feret"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gminferet
                gfp["Mean feret"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gmeanferet
                gfp["Max feret"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gmaxferet
                gfp["2D Aspect ratio"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gmaxferet / Gminferet
                gfp["Volume fraction"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gvolume / cvol * 100
                gfp["Orientation"][it, iw, 0, ir % 5, fovno, cell_id, :] = Gorient3D
                gfp["z-distribution"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gz_dist
                gfp["radial distribution 2D"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gradial_dist2d
                gfp["radial distribution 2Dmean"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gradial_dist2d / cmferet
                gfp["radial distribution 3D"][it, iw, 0, ir % 5, ifovno, cell_id, :] = Gradial_dist3d

            end_ts = datetime.datetime.now()
            print(f"{basename} done in {str(end_ts - start_ts)}")

            # print(f"{channel}volvalues : {np.count_nonzero(~np.isnan(gfp["Volume"]))}")
        except Exception as e:
            print("Exception: ", e, traceback.format_exc())

    if not test:
        time.sleep(120)

    allCellvals = [cell["Centroid"], cell["Volume"], cell["X span"], cell["Y span"], cell["Z span"], cell["MIP area"],
                   cell["Max feret"], cell["Min feret"], cell["Mean feret"], cell["2D Aspect ratio"],
                   cell["Sphericity"]]  ##

    allDNAvals = [dna["Centroid"], dna["Volume"], dna["X span"], dna["Y span"], dna["Z span"], dna["MIP area"],
                  dna["Max feret"], dna["Min feret"], dna["Mean feret"], dna["2D Aspect ratio"], dna["Volume fraction"],
                  dna["Sphericity"], dna["z-distribution"]]

    # dnastackinvaginationvfrac
    allGFPvals = [gfp["Centroid"], gfp["Volume"], gfp["X span"], gfp["Y span"], gfp["Z span"], gfp["MIP area"],
                  gfp["Max feret"], gfp["Min feret"], gfp["Mean feret"], gfp["2D Aspect ratio"], gfp["Volume fraction"],
                  gfp["Count per cell"],
                  gfp["Orientation"], gfp["z-distribution"], gfp["radial distribution 2D"],
                  gfp["radial distribution 2Dmean"], gfp["radial distribution 3D"],
                  gfp["Mean Volume"], gfp["Mean 2D distance to wall d0"], gfp["Stdev 2D distance to wall d0"],
                  gfp["Mean3D distance to wall d0"],
                  gfp["Stdev 3D distance to wall d0"], gfp["Mean 2D distance to wall d1"],
                  gfp["Stdev 2D distance to wall d1"], gfp["Mean3D distance to wall d1"],
                  gfp["Stdev 3D distance to wall d2"], gfp["Mean 2D distance to wall d3"],
                  gfp["Stdev 2D distance to wall d3"], gfp["Mean3D distance to wall d3"],
                  gfp["Stdev 3D distance to wall d3"], gfp["Mean 2D distance to wall d4"],
                  gfp["Stdev 2D distance to wall d4"], gfp["Mean3D distance to wall d4"],
                  gfp["Stdev 3D distance to wall d4"], gfp["Mean 2D distance to wall d5"],
                  gfp["Stdev 2D distance to wall d5"], gfp["Mean3D distance to wall d5"],
                  gfp["Stdev 3D distance to wall d5"], gfp[f"Mean Bottom z-distance d0"],
                  gfp[f"Stdev Bottom z-distance d0"], gfp[f"Mean Bottom z-distance d1"],
                  gfp[f"Stdev Bottom z-distance d1"], gfp[f"Mean Bottom z-distance d2"],
                  gfp[f"Stdev Bottom z-distance d2"], gfp[f"Mean Bottom z-distance d3"],
                  gfp[f"Stdev Bottom z-distance d3"], gfp[f"Mean Bottom z-distance d4"],
                  gfp[f"Stdev Bottom z-distance d4"], gfp[f"Mean Bottom z-distance d5"],
                  gfp[f"Stdev Bottom z-distance d5"], gfp[f"Mean Top z-distance d0"], gfp[f"Stdev Top z-distance d0"],
                  gfp[f"Mean Top z-distance d1"],
                  gfp[f"Stdev Top z-distance d1"], gfp[f"Mean Top z-distance d2"], gfp[f"Stdev Top z-distance d2"],
                  gfp[f"Mean Top z-distance d3"],
                  gfp[f"Stdev Top z-distance d3"], gfp[f"Mean Top z-distance d4"], gfp[f"Stdev Top z-distance d4"],
                  gfp[f"Mean Top z-distance d5"],
                  gfp[f"Stdev Top z-distance d5"], gfp[f"Mean Bottom surface z-distance d0"],
                  gfp[f"Stdev Bottom surface z-distance d0"],
                  gfp[f"Mean Bottom surface z-distance d1"],
                  gfp[f"Stdev Bottom surface z-distance d1"], gfp[f"Mean Bottom surface z-distance d2"],
                  gfp[f"Stdev Bottom surface z-distance d2"],
                  gfp[f"Mean Bottom surface z-distance d3"],
                  gfp[f"Stdev Bottom surface z-distance d3"], gfp[f"Mean Bottom surface z-distance d4"],
                  gfp[f"Stdev Bottom surface z-distance d4"],
                  gfp[f"Mean Bottom surface z-distance d5"],
                  gfp[f"Stdev Bottom surface z-distance d5"], gfp[f"Mean Top surface z-distance d0"],
                  gfp[f"Stdev Top surface z-distance d0"],
                  gfp[f"Mean Top surface z-distance d1"],
                  gfp[f"Stdev Top surface z-distance d1"], gfp[f"Mean Top surface z-distance d2"],
                  gfp[f"Stdev Top surface z-distance d2"],
                  gfp[f"Mean Top surface z-distance d3"],
                  gfp[f"Stdev Top surface z-distance d3"], gfp[f"Mean Top surface z-distance d4"],
                  gfp[f"Stdev Top surface z-distance d4"],
                  gfp[f"Mean Top surface z-distance d5"],
                  gfp[f"Stdev Top surface z-distance d5"]]

    print(f"{len(cellpropnames)}:{len(allCellvals)}")
    print(f"{len(DNApropnames)}:{len(allDNAvals)}")
    print(f"{len(GFPpropnames)}:{len(allGFPvals)}")
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
