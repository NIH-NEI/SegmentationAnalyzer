import warnings

import numpy as np
import pandas as pd
from imea import measure_2d
from scipy.ndimage.measurements import label, find_objects, center_of_mass
from sklearn.decomposition import PCA

from src.AnalysisTools import conv_hull
from src.AnalysisTools import experimentalparams as ep
from src.AnalysisTools.datautils import checkfinitetemp

XSCALE, YSCALE, ZSCALE = ep.XSCALE, ep.YSCALE, ep.ZSCALE
VOLUMESCALE = ep.VOLUMESCALE
AREASCALE = ep.AREASCALE


def getedgeconnectivity(slices, maxz):
    """
    Returns tags indicating connectivity of 3d object to top or bottom or both. Such data will be
    excluded from calculations due to the possibility of it being cut off.

    :param slices: slice object
    :param maxz: max z based on z dimension of original stack
    :return: tags indicating connectivity of 3d object to top or bottom or both.
    """
    connectedbot = False
    connectedtop = False
    minz = 0
    if maxz == 0:
        raise Exception
    top = slices[0].start
    bottom = slices[0].stop
    tags = ""
    if top == 0:
        tags = tags + "t"  # change to t?
        connectedtop = True
    if bottom == maxz:
        tags = tags + "b"
        connectedbot = True
    if tags == "":
        tags = tags + "n"
    #     print(top, bottom, maxz,"tags",tags,slices,bottom==(maxz-1), maxz-1, flush=True)

    return tags, connectedtop, connectedbot


def orientation_3D(bboximage):
    """
    Calculates 3D orientation based on PCA with 3 components. Note that this orientation does not
    measure actual feret length, but is based on the distribution of data.

    :param bboximage:
    :return:
    """
    # find all filled points
    Xpoints = np.array(np.where(bboximage > 0)).T
    anglex, angley, anglez = np.nan, np.nan, np.nan
    if Xpoints.shape < (3,3):
        pass
    else:
        # X = np.nan*np.ones_like(Xpoints)
        # X = np.nan * np.ones().T
        pca = PCA(n_components=3).fit(Xpoints)
        a, b, g = pca.components_[0]
        anglex = np.arctan2(a, b)
        angley = np.arctan2(b, g)
        anglez = np.arctan2(g, a)
    return [anglex, angley, anglez]


def calcs_(bboxdata, usephull=False, debug=False):
    """
    Does calculations for True voxels within a bounding box provided in input. Using the selected
    area reduces calculation time required. Calculations are done for spans along X, Y and Z axes.
    Maximum and minimum feret lengths, centroid coordinates and volume.

    :param bboxdata: 3D data in region of interest (bounding box)
    :return:centroid, volume, xspan, yspan, zspan, maxferet, minferet measurements
    """
    testdata = (bboxdata > 0)
    miparea = np.nan
    maxferet, minferet = np.nan, np.nan
    centroid = center_of_mass(testdata)
    volume = np.count_nonzero(testdata) * VOLUMESCALE
    xspan, yspan, zspan = np.nan, np.nan, np.nan
    if volume > 0:
        try:
            ns = np.transpose(np.nonzero(testdata))
            zspan = (ns.max(axis=0)[0] - ns.min(axis=0)[0]) * ZSCALE
            xspan = (ns.max(axis=0)[1] - ns.min(axis=0)[1]) * XSCALE
            yspan = (ns.max(axis=0)[2] - ns.min(axis=0)[2]) * YSCALE
            # proj2dshadow = np.max(testdata, axis=0) > 0 # same as np.any TODO: test speed
            proj2dshadow = np.any(testdata, axis=0)
            miparea = np.count_nonzero(proj2dshadow) * AREASCALE
            # print("SHADOW TEST:", proj2dshadow==image)
            # print("SHADOW TEST:", False in (proj2dshadow==image))
            # print(proj2dshadow)
            # print(image)
            if usephull:
                phull = conv_hull.pseudo_hull(proj2dshadow)
                chull = phull[conv_hull.ConvexHull(phull).vertices]
                rhull = conv_hull.remove_noisy_points(chull, 0.9)
                statlengths = conv_hull.feret_diam(chull)
            else:
                statlengths, _, _, _, _, _ = measure_2d.statistical_length.compute_statistical_lengths(proj2dshadow,
                                                                                                       daplha=1)
            maxferet, minferet = np.amax(statlengths) * XSCALE, np.amin(statlengths) * XSCALE
            if not (minferet < XSCALE, YSCALE < maxferet):
                warnings.warn("possible discrepancy in feret")
        except Exception as e:
            print(e)
    else:
        # print("VOLUME", volume, volume > 0, volume > 0. )
        volume = np.nan
    return centroid, volume, xspan, yspan, zspan, maxferet, minferet, miparea


def individualcalcs(bboxdata):
    """

    :param bboxdata: 3D data in region of interest
    :return: centroids, volumes, xspans, yspans, zspans, maxferets, minferets, orientation3D
    measurements for individual organelles
    """
    # centroids, volumes, xspans, yspans, zspans, maxferets, minferets, orientations3D = [], [], [], [], [], [], [], []
    mno = ep.MAX_ORGANELLE_PER_CELL
    centerz = ep.Z_FRAMES_PER_STACK//2+1

    volumes, xspans, yspans, zspans, maxferets, minferets, mipareas = np.nan * np.ones(mno), np.nan * np.ones(mno), np.nan * np.ones(mno), np.nan * np.ones(mno), np.nan * np.ones(mno), np.nan * np.ones(mno), np.nan * np.ones(mno)
    z_distributions = np.zeros(centerz*2)
    organellelabel, organellecounts = label(bboxdata > 0)
    org_df = pd.DataFrame(np.arange(1, organellecounts + 1, 1), columns=['organelle_index'])

    # distributioncalcs
    radial_distribution2ds, radial_distribution3ds = [], []
    centroids, orientations3D = np.nan * np.ones((mno,3)), np.nan * np.ones((mno, 3))
    cellcentroid = center_of_mass(bboxdata)

    for index, row in org_df.iterrows():
        if index < mno:
            org_index = int(row['organelle_index'])
            orgs = organellelabel == org_index
            bboxcrop = find_objects(orgs)
            slices = bboxcrop[0]
            centroid, volume, xspan, yspan, zspan, maxferet, minferet, miparea = calcs_(bboxdata[slices])
            # distributioncalcs TODO
            # z_dist = cellcentroid[0] - centroid[0]
            # radial_distribution2d = np.sqrt((cellcentroid[1] - centroid[1]) ** 2 + (cellcentroid[2] - centroid[2]) ** 2)
            # radial_distribution3d = np.cbrt(
            #     (cellcentroid[0] - centroid[0]) ** 2 + (cellcentroid[1] - centroid[1]) ** 2 + (
            #                 cellcentroid[2] - centroid[2]) ** 2)
            # orientation3D = orientation_3D(bboxdata)
            #         orientations - PCA?
            # orgvals = [volume, xspan, yspan, zspan, maxferet, minferet, miparea]
            # if checkfinitetemp(orgvals):
            centroids[index, :] = np.array(centroid)
            volumes[index] = volume
            xspans[index] = xspan
            yspans[index] = yspan
            zspans[index] = zspan
            maxferets[index] = maxferet
            minferets[index] = minferet
            mipareas[index] = miparea
                # z_distributions[round(centerz+z_dist)]+=1
                # radial_distribution2ds.append(radial_distribution2d)
                # radial_distribution3ds.append(radial_distribution3d)
                # orientations3D[index,:] = np.array(orientation3D)
            # z_distributions = np.asarray(z_distributions)
            # radial_distribution2ds = np.asarray(radial_distribution2ds)
            # radial_distribution3ds = np.asarray(radial_distribution3ds)
            # print(orientations3D.shape)
            # print("EXITING TEST")
            # exit()

        else:
            print(f"more than {mno} organelles found: {organellecounts}")

    return centroids, volumes, xspans, yspans, zspans, maxferets, minferets, mipareas#, orientations3D, z_distributions, radial_distribution2ds, radial_distribution3ds
