import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from scipy.ndimage.measurements import label, find_objects, center_of_mass
from src.AnalysisTools import experimentalparams as ep
from src.AnalysisTools.datautils import checkfinite
import warnings
from imea import shape_measurements_2d, measure_2d
XSCALE, YSCALE, ZSCALE = ep.XSCALE, ep.YSCALE, ep.ZSCALE
VOLUMESCALE = ep.VOLUMESCALE
AREASCALE = ep.AREASCALE


def getedgeconnectivity(slices, maxz):

    minz = 0
    if maxz == 0:
        raise Exception
    top = slices[0].start
    bottom = slices[0].stop
    tags = ""
    if top == 0:
        tags = tags + "t"  # change to t?
    if bottom == maxz:
        tags = tags + "b"
    if tags == "":
        tags = tags + "n"
    #     print(top, bottom, maxz,"tags",tags,slices,bottom==(maxz-1), maxz-1, flush=True)

    return tags


def orientation_3D(bboximage):
    # find all filled points
    X = np.array(np.where(bboximage > 0)).T
    pca = PCA(n_components=3).fit(X)
    a, b, g = pca.components_[0]
    anglex = np.arctan2(a, b)
    angley = np.arctan2(b, g)
    anglez = np.arctan2(g, a)
    return [anglex, angley, anglez]


def calcs_(bboxdata):
    """

    :param bboxdata: 3D data in region of interest
    :return:centroid, volume, xspan, yspan, zspan, maxferet, minferet measurements
    """
    testdata = (bboxdata > 0)
    maxferet, minferet = np.nan, np.nan
    centroid = center_of_mass(testdata)
    volume = np.count_nonzero(testdata) * VOLUMESCALE
    xspan, yspan, zspan = np.nan, np.nan, np.nan  # change to nan?
    try:
        ns = np.transpose(np.nonzero(testdata))
        zspan = (ns.max(axis=0)[0] - ns.min(axis=0)[0]) * ZSCALE
        xspan = (ns.max(axis=0)[1] - ns.min(axis=0)[1]) * XSCALE
        yspan = (ns.max(axis=0)[2] - ns.min(axis=0)[2]) * YSCALE
        proj2dshadow = np.max(testdata, axis=0) > 0
        statlengths, _, _, _, _, _ = measure_2d.statistical_length.compute_statistical_lengths(proj2dshadow, daplha=1)
        maxferet, minferet = np.amax(statlengths) * XSCALE, np.amin(statlengths) * XSCALE
        if not (minferet < XSCALE, YSCALE < maxferet):
            warnings.warn("possible discrepancy in feret")
    except Exception as E:
        print(E)
        # pass
    #     m = moments_central
    return centroid, volume, xspan, yspan, zspan, maxferet, minferet


def individualcalcs(bboxdata):
    """

    :param bboxdata: 3D data in region of interest
    :return: centroids, volumes, xspans, yspans, zspans, maxferets, minferets, orientation3D measurements for individual organelles
    """
    centroids, volumes, xspans, yspans, zspans, maxferets, minferets = [], [], [], [], [], [], []
    organellelabel, organellecounts = label(bboxdata > 0)
    org_df = pd.DataFrame(np.arange(1, organellecounts + 1, 1), columns=['organelle_index'])

    # distributioncalcs
    z_distributions, radial_distribution2ds, radial_distribution3ds, orientations = [], [], [], []
    cellcentroid = center_of_mass(bboxdata)

    for index, row in org_df.iterrows():
        org_index = int(row['organelle_index'])
        orgs = organellelabel == org_index
        bboxcrop = find_objects(orgs)
        slices = bboxcrop[0]
        centroid, volume, xspan, yspan, zspan, maxferet, minferet = calcs_(bboxdata[slices])
        orgvals = [centroid, volume, xspan, yspan, zspan, maxferet, minferet]
        # distributioncalcs
        z_distribution = cellcentroid[0] - centroid[0]
        radial_distribution2d = np.sqrt((cellcentroid[1] - centroid[1]) ** 2 + (cellcentroid[2] - centroid[2]) ** 2)
        radial_distribution3d = np.cbrt((cellcentroid[0] - centroid[0]) ** 2 + (cellcentroid[1] - centroid[1]) ** 2 + (
                cellcentroid[2] - centroid[2]) ** 2)
        orientation3D = orientation_3D(bboxdata)
        #         orientations - PCA?
        if checkfinite(orgvals):
            centroids.append(centroid)
            volumes.append(volume)
            xspans.append(xspan)
            yspans.append(yspan)
            zspans.append(zspan)
            maxferets.append(maxferet)
            minferets.append(minferet)
    return centroids, volumes, xspans, yspans, zspans, maxferets, minferets, orientation3D
