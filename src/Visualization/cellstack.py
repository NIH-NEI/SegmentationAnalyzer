import numpy as np
from aicsimageio.writers import OmeTiffWriter
from aicsshparam import shparam, shtools
from skimage.morphology import binary_dilation


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


def saveSHEsurfaces(stack3d, lmax=50):
    stack3d = stack3d.data.astype(np.uint8)
    stack3d = stack3d > 0
    stack3d = stack3d.squeeze()
    (coeffs, grid_rec), (image_, mesh, grid, centroid) = shparam.get_shcoeffs(image=stack3d, lmax=lmax)
    mse = shtools.get_reconstruction_error(grid, grid_rec)

    # Print results
    print('Coefficients:', len(coeffs), coeffs)
    print(coeffs.values())
    print('Error:', mse)
    pass


def merge_entire_stack(Cellstackpath, DNAstackpath, GFPstackpath, savename=""):
    from src.stackio import stackio
    success = False
    try:
        img_GFP = stackio.opensegmentedstack(GFPstackpath)
        img_ACTIN = stackio.opensegmentedstack(Cellstackpath)  # binary=False
        img_DNA = stackio.opensegmentedstack(DNAstackpath)  # binary=False
        # CZXY
        img_DNA = img_DNA & img_ACTIN
        img_GFP = img_GFP & img_ACTIN
        mergedchannel = np.stack([img_ACTIN, img_DNA, img_GFP], axis=0)
        # TCZXY
        mergedchannel = np.expand_dims(mergedchannel, axis=0)
        mergedchannel = mergedchannel * 255
        OmeTiffWriter.save(data=mergedchannel, uri=f"{savename}.tiff", overwrite_file=True)
        success = True
    except Exception as e:
        print(e)
    return success


if __name__ == "__main__":
    # [cell, dna, gfp] = [np.random.random((20, 500, 500)) >= 0.4 for _ in range(3)]
    # mergestack(cell, dna, gfp, savename="test")
    # Cellstackpath= "C:/Users/satheps/PycharmProjects/Results/2022/Mar25/illustrations_TOM/P1-W2-SEC_G02_F004_Actin_RPE.tif"
    # DNAstackpath= "C:/Users/satheps/PycharmProjects/Results/2022/Mar25/illustrations_TOM/P1-W2-SEC_G02_F004_DNA_RPE.tif"
    # GFPstackpath= "C:/Users/satheps/PycharmProjects/Results/2022/Mar25/illustrations_TOM/P1-W2-SEC_G02_F004_0.625_0.075_GFP.tiff"
    from src.AnalysisTools import datautils, experimentalparams as ep
    from src.stackio import stackio
    from os.path import join

    segmented_ch_folder_GFP = 'C:/Users/satheps/PycharmProjects/Results/2022/final_segmentations/LAMP1/'
    segmented_ch_folder_Cell = 'C:/Users/satheps/PycharmProjects/Results/2022/final_segmentations/LAMP1/Cell/'
    dnafnames = datautils.getFileListContainingString(segmented_ch_folder_Cell, 'DNA_RPE.tif')
    actinfnames = datautils.getFileListContainingString(segmented_ch_folder_Cell, 'Actin_RPE.tif')
    GFPfnames = datautils.getFileListContainingString(segmented_ch_folder_GFP, '_GFP')
    savename = "C:/Users/satheps/PycharmProjects/Results/2022/Mar25/illustrations_LAMP/"
    print(len(dnafnames), len(actinfnames),len(GFPfnames))
    dnafiles, actinfiles, GFPfiles, no_stacks = datautils.orderfilesbybasenames(dnafnames, actinfnames, GFPfnames,
                                                                                debug=True)
    # standard stacks
    ts = [0, 1]
    rs = [0, 5]
    fovnos = [0]
    for stackid, (actinfile, dnafile, GFPfile) in enumerate(zip(actinfiles, dnafiles, GFPfiles)):
        week, rep, w, r, fov, fovno, basename = datautils.getwr_3channel(dnafile, actinfile, GFPfile)
        t = ep.findtreatment(r)
        # print(
        #     f"\nWeek:{week}, {w}\t|| Replicate: {rep}, {r}\t|| Treatment {t}\t|| Field of view: {fov}, {fovno}\t||Basename: {basename}")
        if t in ts and r in rs and fovno in fovnos:
            print(
                f"\nWeek:{week}, {w}\t|| Replicate: {rep}, {r}\t|| Treatment {t}\t|| Field of view: {fov}, {fovno}\t||Basename: {basename}")

            Cellstackpath = join(segmented_ch_folder_Cell, actinfile)
            DNAstackpath = join(segmented_ch_folder_Cell, dnafile)
            GFPstackpath = join(segmented_ch_folder_GFP, GFPfile)
            savepath = join(savename, basename)
            merge_entire_stack(Cellstackpath, DNAstackpath, GFPstackpath, savename=savepath)
