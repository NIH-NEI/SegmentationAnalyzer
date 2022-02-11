import warnings

import numpy as np
import pandas as pd
from imea import measure_2d
from scipy.ndimage.measurements import label, find_objects, center_of_mass
from sklearn.decomposition import PCA
import traceback
from src.AnalysisTools import conv_hull
from src.AnalysisTools import experimentalparams as ep

XSCALE, YSCALE, ZSCALE = ep.XSCALE, ep.YSCALE, ep.ZSCALE
VOLUMESCALE = ep.VOLUMESCALE
AREASCALE = ep.AREASCALE


def getedgeconnectivity(slices, maxz):
    """
    Returns tags indicating connectivity of 3d object to top or bottom or both. Such data should be
    excluded from calculations due to the possibility of it being cut off.

    :param slices: slice object
    :param maxz: max z based on z dimension of original stack
    :return: tags indicating connectivity of 3d object to top or bottom or both.
    """
    connectedbot = False
    connectedtop = False
    minz = 0
    assert maxz >0, "maximum z must be greater than 0 to detect edge connectivity"
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

    :param bboximage: stack dimensions are in order z,x,y
    :return:
    """
    # find all filled points
    Xpoints = np.array(np.where(bboximage > 0)).T
    r, theta, phi = np.nan, np.nan, np.nan
    if Xpoints.shape < (3, 3):
        pass
    else:
        # X = np.nan*np.ones_like(Xpoints)
        # X = np.nan * np.ones().T
        pca = PCA(n_components=3).fit(Xpoints)
        zc, xc, yc = pca.components_[0] # selecting z,y,x values of the first component
        # Calculations of angles for a spherical coordinate system

        r = np.sqrt(zc**2 + xc**2 + yc**2)
        xy = np.sqrt(xc**2 + yc**2)
        theta = np.arctan2(zc, xy) * 180/np.pi
        phi = np.arctan2(yc, zc) * 180/np.pi
    return [r, theta, phi]


def calculate_object_properties(bboxdata, usephull=False, debug=False):
    """
    Does calculations for True voxels within a bounding box provided in input. Using the selected
    area reduces calculation time required. Calculations are done for spans along X, Y and Z axes.
    Maximum and minimum feret lengths, centroid coordinates and volume.

    :param bboxdata: 3D data in region of interest (bounding box)
    :return:centroid, volume, xspan, yspan, zspan, maxferet, minferet measurements
    """
    bboxdatabw = (bboxdata > 0)
    miparea = np.nan
    maxferet, minferet = np.nan, np.nan
    # print(np.asarray(center_of_mass(bboxdatabw)),np.array([ZSCALE, XSCALE, YSCALE]) )
    centroid = np.multiply(np.asarray(center_of_mass(bboxdatabw)), np.array([ZSCALE, XSCALE, YSCALE]))
    volume = np.count_nonzero(bboxdatabw) * VOLUMESCALE
    xspan, yspan, zspan = np.nan, np.nan, np.nan
    if volume > 0:
        try:
            ns = np.transpose(np.nonzero(bboxdatabw))
            #NOTE: new calculations
            zspan = (ns.max(axis=0)[0] - ns.min(axis=0)[0] + 1) * ZSCALE
            xspan = (ns.max(axis=0)[1] - ns.min(axis=0)[1] + 1) * XSCALE
            yspan = (ns.max(axis=0)[2] - ns.min(axis=0)[2] + 1) * YSCALE
            # proj2dshadow = np.max(bboxdatabw, axis=0) > 0 # same as np.any
            proj2dshadow = np.any(bboxdatabw, axis=0)
            miparea = np.count_nonzero(proj2dshadow) * AREASCALE
            # print("Object shadow test:", False in (proj2dshadow==image))
            if usephull:
                phull = conv_hull.pseudo_hull(proj2dshadow)
                chull = phull[conv_hull.ConvexHull(phull).vertices]
                rhull = conv_hull.remove_noisy_points(chull, 0.9)
                statlengths = conv_hull.feret_diam(chull)
            else:
                statlengths, _, _, _, _, _ = measure_2d.statistical_length.compute_statistical_lengths(proj2dshadow,
                                                                                                       daplha=1)
            maxferet, minferet = np.amax(statlengths) * XSCALE, np.amin(statlengths) * XSCALE
            meanferet = np.mean(statlengths) * XSCALE
            if not (minferet < XSCALE, YSCALE < maxferet):
                warnings.warn("possible discrepancy in feret measurements")
        except Exception as e:
            print(e)
    else:
        volume = np.nan
    return centroid, volume, xspan, yspan, zspan, maxferet, meanferet, minferet, miparea


def individualcalcs(bboxdata):
    """
    Note: Dimensiond must be in the order: z,x,y

    :param bboxdata: 3D data in region of interest
    :return:
    :param centroids: center of mass (with all voxels weighted equally) giving the geometric centroid.
    :param volumes: volume in pixels of each organelle
    :param xspans:
    :param yspans:
    :param zspans:
    :param maxferets:
    :param minferets:
    :param mipareas:
    :param orientation3D:
    :param z_distributions:
    :param radial_distribution2ds:
    :param radial_distribution3ds:

    measurements for individual organelles
    """
    # print("GFP initiated", flush=True)
    # centroids, volumes, xspans, yspans, zspans, maxferets, minferets, orientations3D = [], [], [], [], [], [], [], []
    mno = ep.MAX_ORGANELLE_PER_CELL
    # centerz = ep.Z_FRAMES_PER_STACK//2+1

    volumes, xspans, yspans, zspans, maxferets, meanferets, minferets, mipareas = np.nan * np.ones(mno), np.nan * np.ones(
        mno), np.nan * np.ones(mno), np.nan * np.ones(mno), np.nan * np.ones(mno), np.nan * np.ones(
        mno), np.nan * np.ones(mno), np.nan * np.ones(mno)
    z_distributions, radial_distribution3ds, radial_distribution2ds = np.nan * np.ones(mno), np.nan * np.ones(
        mno), np.nan * np.ones(mno)
    organellelabel, organellecounts = label(bboxdata > 0)
    org_df = pd.DataFrame(np.arange(1, organellecounts + 1, 1), columns=['organelle_index'])

    centroids, orientations3D = np.nan * np.ones((mno, 3)), np.nan * np.ones((mno, 3))
    cellcentroid = np.multiply(np.asarray(center_of_mass(bboxdata)), np.array([ZSCALE, XSCALE, YSCALE]))
    # print("GFP initiated2", flush=True)
    # try:
    for index, row in org_df.iterrows():
        if index < mno:
            org_index = int(row['organelle_index'])
            orgs = organellelabel == org_index
            bboxcrop = find_objects(orgs)
            gfpslices = bboxcrop[0] # slices for gfp channel
            # All properties obtained from calcs are already scaled
            centroid, volume, xspan, yspan, zspan, maxferet, meanferet, minferet, miparea = calculate_object_properties(bboxdata[gfpslices])
            # distributioncalcs
            z_dist = centroid[0] - cellcentroid[0] # distance from centroid of the cell
            radial_distribution2d = np.sqrt((centroid[1] - cellcentroid[1]) ** 2 + (centroid[2] - cellcentroid[2]) ** 2)

            radial_distribution3d = np.sqrt((cellcentroid[0] - centroid[0]) ** 2 + (cellcentroid[1] - centroid[1]) ** 2
                                            + (cellcentroid[2] - centroid[2]) ** 2)
            orientation3D = orientation_3D(bboxdata)
            # orientations - PCA
            centroids[index, :] = np.array(centroid)
            volumes[index] = volume
            xspans[index] = xspan
            yspans[index] = yspan
            zspans[index] = zspan
            maxferets[index] = maxferet
            meanferets[index] = maxferet
            minferets[index] = minferet
            mipareas[index] = miparea
            z_distributions[index] = z_dist
            radial_distribution2ds[index] = radial_distribution2d
            radial_distribution3ds[index] = radial_distribution3d
            orientations3D[index, :] = np.array(orientation3D)

            # print(z_dist, radial_distribution2d, radial_distribution3d, orientation3D, volume)
            # print(np.count_nonzero(~np.isnan(z_distributions)), np.count_nonzero(~np.isnan(radial_distribution2ds)),
            #       np.count_nonzero(~np.isnan(radial_distribution3ds)), np.count_nonzero(~np.isnan(orientations3D)),
            #       np.count_nonzero(~np.isnan(volumes)), np.count_nonzero(~np.isnan(xspan)), np.count_nonzero(~np.isnan(miparea)))
            # exit()
            # z_distributions = np.asarray(z_distributions)
            # radial_distribution2ds = np.asarray(radial_distribution2ds)
            # radial_distribution3ds = np.asarray(radial_distribution3ds)
            # print(orientations3D.shape)
            # print("EXITING TEST")
            # exit()

        else:
            print(f"more than {mno} organelles found: {organellecounts}")
            # add "organellecounts"
    # except Exception as e:
    #     print(e, traceback.format_exc())
    return organellecounts, centroids, volumes, xspans, yspans, zspans, maxferets, meanferets, minferets, mipareas, orientations3D, z_distributions, radial_distribution2ds, radial_distribution3ds
