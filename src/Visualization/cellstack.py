import numpy as np
from aicsimageio.writers import OmeTiffWriter
from aicsshparam import shparam, shtools


def mergestack(CellObject, DNAObjects, GFPObjects, savename, save=True, debug = False):
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
        mergedchannel = mergedchannel*255

        if save:
            OmeTiffWriter.save(data=mergedchannel, uri=f"{savename}.tiff", overwrite_file=True)
        success = True
    except Exception as e:
        print(e)
    return success

def saveSHEsurfaces(stack3d, lmax = 50):
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



if __name__ == "__main__":
    [cell, dna, gfp] = [np.random.random((20, 500, 500)) >= 0.4 for _ in range(3)]
    mergestack(cell, dna, gfp, savename="test")
