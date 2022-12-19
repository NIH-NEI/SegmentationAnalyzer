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
@click.option("--debug", default=False, metavar="<Boolean>", help="Show extra information for debugging")
@click.option("--usednareference", default=False, metavar="<Boolean>", help="Use DNA as a reference instead of Cell")
@click.option("--num_processes", default=4, metavar="<int>", help="Use DNA as a reference instead of Cell")
# @click.option("--help", help="Show details for function ")
def calculateCellMetrics(gfpfolder: PathLike, cellfolder: PathLike, savepath: PathLike, channel: str,
                         usesampledataonly: bool,
                         test: bool, debug: bool, usednareference=False, num_processes: int = 4):
    """
    Read all segmented image files. Measure shape metrics based on corresponding co-registered channels and save data for each metric.

    """
    print(f"Paths: gfp folder = {gfpfolder}\t cell folder = {cellfolder}\t savefolder = {savepath}")
    # print(test)
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
    assert exists(gfpfolder)
    assert exists(cellfolder)
    assert exists(savepath)
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
    gfp["wall_dist_2d_ms"] = np.nan * np.ones(gfp["shape"])
    gfp["wall_dist_2d_ss"] = np.nan * np.ones(gfp["shape"])
    gfp["wall_dist_3d_ms"] = np.nan * np.ones(gfp["shape"])
    gfp["wall_dist_3d_ss"] = np.nan * np.ones(cell["shape"])

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

                print(
                    f"img_GFP.shape: {img_GFP.shape} == labelactin.shape: {labelactin.shape} == labeldna.shape: {labeldna.shape}")

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
                obj_df = pd.DataFrame(np.arange(1, len(is_) + 1, 1), columns=['object_index'])
                for index, row in obj_df.iterrows():
                    cellinputdict = {}
                    obj_index = int(row['object_index'])
                    objs = labelactin == obj_index  # object with object label 'obj_index'
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
                        memberdnas = np.where(dna_actin_membership[obj_index - 1, :])[0]
                        no_members = memberdnas.shape[0]
                        DNAObjects = np.zeros_like(labelactin[slices])
                        cell_cstack = 255 * ((labelactin == index) > 0)
                        # dna_cstack = None
                        # gfp_cstack = None
                        if no_members >= 0:  # include cells with no nuclei
                            # if False:  # decide condition
                            # savethisimage = True
                            cell["Volumes"][t, w, 0, r % 5, fovno, obj_index] = Cvolume
                            cell["xspan"][t, w, 0, r % 5, fovno, obj_index] = Cxspan
                            cell["yspan"][t, w, 0, r % 5, fovno, obj_index] = Cyspan
                            cell["zspan"][t, w, 0, r % 5, fovno, obj_index] = Czspan
                            cell["miparea"][t, w, 0, r % 5, fovno, obj_index] = Cmiparea
                            cell["sphericity"][t, w, 0, r % 5, fovno, obj_index] = Csphericity
                            cell["maxferet"][t, w, 0, r % 5, fovno, obj_index] = Cmaxferet
                            cell["minferet"][t, w, 0, r % 5, fovno, obj_index] = Cminferet
                            cell["Centroids"][t, w, 0, r % 5, fovno, obj_index] = Ccentroid
                            cell["aspectratio2d"][t, w, 0, r % 5, fovno, obj_index] = Cmaxferet / Cminferet
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
                                    dnavol1 = dna["Volumes"][t, w, 0, r % 5, fovno, obj_index, 0]
                                    dnavol2 = dna["Volumes"][t, w, 0, r % 5, fovno, obj_index, 1]

                                    if Dvolume == dnavol1 or Dvolume == dnavol2:
                                        print(f"Current volume is the same size as one of the volumes. Skipping")
                                        continue
                                    originaldnas = (dnavol1, dnavol2, Dvolume)
                                    listdnas = list(originaldnas)
                                    listdnas.sort(reverse=True)
                                    # print(f"debug: {listdnas.index(Dvolume)}, {listdnas}, {originaldnas}")

                                    if listdnas.index(Dvolume) < 2:  # means current DNA is top 2 in volume
                                        usememberdnaid = originaldnas.index(listdnas[-1])
                                        print(f"Found current volume {Dvolume} > old volume {listdnas[-1]}. "
                                              f"Replacing DNA at {usememberdnaid} with current DNA. ")
                                    else:
                                        continue
                                else:
                                    usememberdnaid = memberdna_no
                                dna_c_rel = Dcentroid - Ccentroid

                                dna["Centroids"][t, w, 0, r % 5, fovno, obj_index, usememberdnaid] = Dcentroid
                                dna["Volumes"][t, w, 0, r % 5, fovno, obj_index, usememberdnaid] = Dvolume
                                dna["xspan"][t, w, 0, r % 5, fovno, obj_index, usememberdnaid] = Dxspan
                                dna["yspan"][t, w, 0, r % 5, fovno, obj_index, usememberdnaid] = Dyspan
                                dna["zspan"][t, w, 0, r % 5, fovno, obj_index, usememberdnaid] = Dzspan
                                dna["miparea"][t, w, 0, r % 5, fovno, obj_index, usememberdnaid] = Dmiparea
                                dna["sphericity"][t, w, 0, r % 5, fovno, obj_index, usememberdnaid] = Dsphericity
                                dna["maxferet"][t, w, 0, r % 5, fovno, obj_index, usememberdnaid] = Dmaxferet
                                dna["minferet"][t, w, 0, r % 5, fovno, obj_index, usememberdnaid] = Dminferet
                                dna["aspectratio2d"][
                                    t, w, 0, r % 5, fovno, obj_index, usememberdnaid] = Dmaxferet / Dminferet
                                dna["volume_fraction"][
                                    t, w, 0, r % 5, fovno, obj_index, usememberdnaid] = Dvolume / Cvolume * 100
                                dna["zdistr"][t, w, 0, r % 5, fovno, obj_index, usememberdnaid] = dna_c_rel[0]
                        # GFP members
                        GFPObjects = (img_GFP[slices] & CellObject)

                        saveindividualcellstack = (np.random.random(1)[0] < 0.05)  # 10% sample ~~is_//10
                        # saveindividualcellstack = True  # 10% sample ~~is_//10
                        if saveindividualcellstack:
                            cellstackfolder = join(savepath, 'cellstacks').replace("\\", "/")
                            if not isdir(cellstackfolder):
                                mkdir(cellstackfolder)
                            stackfilename = f"{channel}_{basename}_{obj_index}"
                            cellstack.mergestack(CellObject, DNAObjects, GFPObjects,
                                                 savename=join(cellstackfolder, stackfilename), save=True,
                                                 add_3d_cell_outline=False)
                        # print("shapes: ", CellObject.shape, DNAObjects.shape, GFPObjects.shape)
                        usednareference = False
                        refcentroid = None

                        if usednareference:
                            refcentroid = Dcentroid
                        else:
                            refcentroid = Ccentroid

                        processes.append((t, w, r, fovno, obj_index, Cvolume, Cmeanferet,
                                          executor.submit(ShapeMetrics.calculate_multiorganelle_properties,
                                                          GFPObjects, refcentroid, CellObject)))
                print("Processes = ", len(processes))

            for it, iw, ir, ifovno, obj_id, cvol, cmferet, process in processes:
                features = process.result()
                Gcount, Gcentroid, Gvolume, Gspan, Gyspan, Gzspan, Gmaxferet, Gmeanferet, Gminferet, Gmiparea, Gorient3D, Gz_dist, Gradial_dist2d, Gradial_dist3d, Gmeanvol, Gwall_dist_2d_ms, Gwall_dist_2d_ss, Gwall_dist_3d_ms, Gwall_dist_3d_ss = features
                # print("gcount:", Gcount)
                # print("indorient", indorient3D.shape, indorient3D.T.shape)
                gfp["cpc"][it, iw, 0, ir % 5, ifovno, obj_id] = Gcount
                gfp["meanvols"][it, iw, 0, ir % 5, ifovno, obj_id] = Gmeanvol
                gfp["Centroids"][it, iw, 0, ir % 5, ifovno, obj_id, :] = Gcentroid
                gfp["Volumes"][it, iw, 0, ir % 5, ifovno, obj_id, :] = Gvolume
                gfp["xspan"][it, iw, 0, ir % 5, ifovno, obj_id, :] = Gspan
                gfp["yspan"][it, iw, 0, ir % 5, ifovno, obj_id, :] = Gyspan
                gfp["zspan"][it, iw, 0, ir % 5, ifovno, obj_id, :] = Gzspan
                gfp["miparea"][it, iw, 0, ir % 5, ifovno, obj_id, :] = Gmiparea
                gfp["maxferet"][it, iw, 0, ir % 5, ifovno, obj_id, :] = Gmaxferet
                gfp["minferet"][it, iw, 0, ir % 5, ifovno, obj_id, :] = Gminferet
                gfp["aspectratio2d"][it, iw, 0, ir % 5, ifovno, obj_id, :] = Gmaxferet / Gminferet
                gfp["volfrac"][it, iw, 0, ir % 5, ifovno, obj_id, :] = Gvolume / cvol * 100
                gfp["orientations"][it, iw, 0, ir % 5, fovno, obj_id, :] = Gorient3D
                gfp["zdistr"][it, iw, 0, ir % 5, ifovno, obj_id, :] = Gz_dist
                gfp["raddist2d"][it, iw, 0, ir % 5, ifovno, obj_id, :] = Gradial_dist2d
                gfp["raddist2dmean"][it, iw, 0, ir % 5, ifovno, obj_id, :] = Gradial_dist2d / cmferet
                gfp["raddist3d"][it, iw, 0, ir % 5, ifovno, obj_id, :] = Gradial_dist3d
                gfp["wall_dist_2d_ms"][it, iw, 0, ir % 5, ifovno, obj_id] = Gwall_dist_2d_ms
                gfp["wall_dist_2d_ss"][it, iw, 0, ir % 5, ifovno, obj_id] = Gwall_dist_2d_ss
                gfp["wall_dist_3d_ms"][it, iw, 0, ir % 5, ifovno, obj_id] = Gwall_dist_3d_ms
                gfp["wall_dist_3d_ss"][it, iw, 0, ir % 5, ifovno, obj_id] = Gwall_dist_3d_ss

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
                  gfp["meanvols"], gfp["wall_dist_2d_ms"], gfp["wall_dist_2d_ss"], gfp["wall_dist_3d_ms"],
                  gfp["wall_dist_3d_ss"]]
    GFPpropnames = ["Centroid", "Volume", "X span", "Y span", "Z span", "MIP area", "Max feret", "Min feret",
                    "2D Aspect ratio", "Volume fraction", "Count per cell", "Orientation", "z-distribution",
                    "radial distribution 2D", "normalized radial distribution 2D", "radial distribution 3D",
                    "Mean Volume"]
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
                success = stackio.checksavedfileintegrity(loaded, prop)

                # success = datautils.array_nan_equal(loaded[loaded.files[0]], prop)
                if success:
                    print(f"SAVE SUCCESSFUL FOR {filename}\t\tNo. of Datapoints: {np.count_nonzero(~np.isnan(prop))}")
                else:  # (2, 4, 1, 5, 6, 1000, 50)
                    print(loaded.files, loaded[loaded.files[0]].shape, prop.shape)
            try:
                if generateplots:
                    plotter.violinstripplot(stackdata=prop, channel=otype, propname=propnames[i],
                                            units=experimentalparams.getunits(propertyname),
                                            percentile_include=True, selected_method_type=None, uselog=uselog[o])
            except Exception as e:
                print(e)
    for organelle in orgenelletype:
        stackio.convertfromnpz_allproperties(npzfolderpath=savepath, targetdir=join(savepath, "csv/"),
                                         organelle=organelle)


if __name__ == "__main__":
    import sys

    args = sys.argv
    print(f"args:{args})")
    # exit()
    # segmented_ch_folder_GFP = 'C:/Users/satheps/PycharmProjects/Results/2022/final_segmentations/CETN2/'
    # segmented_ch_folder_Cell = 'C:/Users/satheps/PycharmProjects/Results/2022/final_segmentations/CETN2/Cell/csv/'
    # savepath = '../Results/2022/May6/cetn2/calcs/'
    calculateCellMetrics()
