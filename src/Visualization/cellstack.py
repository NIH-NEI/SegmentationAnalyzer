import sys

sys.path.extend('../../../SegmentationAnalyzer')

import click
import numpy as np
from aicsimageio.writers import OmeTiffWriter
from skimage.morphology import binary_dilation
from skimage.morphology import octahedron
from src.AnalysisTools.dtypes import PathLike


def mergestack(CellObject, DNAObjects, GFPObjects, savename, save=True, add_3d_cell_outline=False, debug=False):
    """

    :param CellObject: 3 dimensional stack of
    :param DNAObjects:
    :param GFPObjects:
    :param savename:
    :param save: whether to save the file
    :return: True if operation succeeded
    """
    success = False
    try:

        assert CellObject.ndim == 3, "Stacks must have 3 dimensions"
        assert CellObject.shape == DNAObjects.shape == GFPObjects.shape, "all channels must be the same size to generate stack"
        if add_3d_cell_outline:
            CellObject = np.pad(CellObject, ((1, 1), (1, 1), (1, 1)))
            DNAObjects = np.pad(DNAObjects, ((1, 1), (1, 1), (1, 1)))
            GFPObjects = np.pad(GFPObjects, ((1, 1), (1, 1), (1, 1)))
            CellObject = 1 * binary_dilation(CellObject) - 1 * CellObject
        cellbw = CellObject > 0
        dnabw = DNAObjects > 0
        gfpbw = GFPObjects > 0
        # CZXY
        mergedchannel = np.stack([cellbw, dnabw, gfpbw], axis=0)
        # TCZXY
        mergedchannel = np.expand_dims(mergedchannel, axis=0)
        if debug:
            print(f"final dimensions of merged stack: {mergedchannel.shape}")
        mergedchannel = mergedchannel.astype(np.uint8)
        mergedchannel = mergedchannel * 255

        if save:
            OmeTiffWriter.save(data=mergedchannel, uri=f"{savename}.tiff", overwrite_file=True)
        success = True
    except Exception as e:
        print(e)
    return success


def merge_entire_stack(Cellstackpath, DNAstackpath, GFPstackpath, savename="", dilation=0, dilatexyonly=True):
    from src.stackio import stackio
    success = False
    try:
        img_GFP = stackio.opensegmentedstack(GFPstackpath)
        img_ACTIN = stackio.opensegmentedstack(Cellstackpath)  # binary=False
        img_DNA = stackio.opensegmentedstack(DNAstackpath)  # binary=False
        # CZXY
        structuring_element = np.zeros((3, 3, 3), dtype=int)
        if dilatexyonly:
            structuring_element[1, 1, :] = 1
            structuring_element[1, :, 1] = 1
        else:
            structuring_element = octahedron(1)
        val = max(np.unique(img_ACTIN))
        dilated_img_ACTIN = binary_dilation(img_ACTIN > 0, structure=structuring_element, iterations=dilation) * val
        img_DNA = img_DNA & dilated_img_ACTIN
        img_GFP = img_GFP & dilated_img_ACTIN
        mergedchannel = np.stack([img_ACTIN, img_DNA, img_GFP], axis=0)
        # TCZXY
        mergedchannel = np.expand_dims(mergedchannel, axis=0)
        mergedchannel = mergedchannel * 255
        OmeTiffWriter.save(data=mergedchannel, uri=f"{savename}.tiff", overwrite_file=True)
        success = True
    except Exception as e:
        print(e)
    return success


@click.command(options_metavar="<options>")
@click.option("--segmentpath", default="C:/Users/satheps/PycharmProjects/Results/2022/final_segmentations/",
              help="Path to folder containing segmentations. Folder must contain GFP segmentations. <segmentpath>/Cell "
                   "folder must contain corresponding Actin and DNA segmentations ", metavar="<PathLike>")
@click.option("--savepathdir", default="C:/Users/satheps/PycharmProjects/Results/2022/Imaris visualizations/",
              metavar="<PathLike>", help="Path to folder where imaris visualization-ready stacks should be saved")
@click.option("--ndilations", default=1, metavar="<int>", help="Number of cell dilations to be used for GFP channel")
@click.option("--treatments", "ts", default=[0, 1], metavar="List<int>", multiple=True,
              help="Treatment types. Must be 0 and/or 1")
@click.option("--replicates", "rs", default=[0, 5], metavar="List<int>", multiple=True,
              help="replicates. Must be 0 and/or 1")
@click.option("--fovs", "fovnos", default=[5], metavar="List<int>", multiple=True,
              help="Fields of View types. Must be number combinatiosn from [0,...,5]")
# @click.option("--help", help="Show details for function ")
def mergeallstacks(segmentpath: PathLike, savepathdir: PathLike, ndilations: int, ts: list = [0, 1], rs: list = [0, 5],
                   fovnos: list = [5]):
    """

    :param segmentpath:
    :param savepathdir:
    :param ndilations:
    :param ts:
    :param rs:
    :param fovnos:
    :return:
    """

    import os
    print(os.getcwd())
    # [cell, dna, gfp] = [np.random.random((20, 500, 500)) >= 0.4 for _ in range(3)]
    # mergestack(cell, dna, gfp, savename="test")

    from src.AnalysisTools import datautils, experimentalparams as ep
    # from src.stackio import stackio
    from os.path import join
    import os

    chlist = os.listdir(segmentpath)

    print(f"Channel List (must contain all channels):\t{chlist}")

    # exit()
    for ch in chlist:
        segmented_ch_folder_GFP = f'{segmentpath}{ch}/'
        segmented_ch_folder_Cell = f'{segmentpath}{ch}/Cell/'
        dnafnames = datautils.getFileListContainingString(segmented_ch_folder_Cell, 'DNA_RPE.tif')
        actinfnames = datautils.getFileListContainingString(segmented_ch_folder_Cell, 'Actin_RPE.tif')
        GFPfnames = datautils.getFileListContainingString(segmented_ch_folder_GFP, '_GFP')
        # savename = 'C:/Users/satheps/PycharmProjects/Results/2022/May6/cetn2/illustrations_CETN2/imgs/'
        foldername = f'{savepathdir}{ch}/p'
        if not os.path.exists(foldername):
            os.mkdir(foldername)
        savename = f'{savepathdir}{ch}/imgs/'
        if not os.path.exists(savename):
            os.mkdir(savename)
        print(len(dnafnames), len(actinfnames), len(GFPfnames))
        dnafiles, actinfiles, GFPfiles, no_stacks = datautils.orderfilesbybasenames(dnafnames, actinfnames, GFPfnames,
                                                                                    debug=False)
        # standard stacks
        for stackid, (actinfile, dnafile, GFPfile) in enumerate(zip(actinfiles, dnafiles, GFPfiles)):
            week, rep, w, r, fov, fovno, basename = datautils.getwr_3channel(dnafile, actinfile, GFPfile)
            t = ep.find_treatment(r)
            if t in ts and r in rs and fovno in fovnos:
                print(f"\nWeek:{week}, {w}\t|| Replicate: {rep}, {r}\t|| Treatment {t}\t"
                      f"|| Field of view: {fov}, {fovno}\t|| Basename: {basename}")
                Cellstackpath = join(segmented_ch_folder_Cell, actinfile)
                DNAstackpath = join(segmented_ch_folder_Cell, dnafile)
                GFPstackpath = join(segmented_ch_folder_GFP, GFPfile)
                savepath = join(savename, basename)
                merge_entire_stack(Cellstackpath, DNAstackpath, GFPstackpath, savename=savepath, dilation=ndilations,
                                   dilatexyonly=True)


if __name__ == "__main__":
    mergeallstacks()
