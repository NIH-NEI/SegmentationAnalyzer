import warnings

import numpy as np
import pandas as pd
from imea import measure_2d
from scipy.ndimage.measurements import label, find_objects, center_of_mass
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.measure import marching_cubes, mesh_surface_area
from sklearn.decomposition import PCA
from aicsimageio.writers import OmeTiffWriter

from src.AnalysisTools import conv_hull
from src.AnalysisTools import experimentalparams as ep

XSCALE, YSCALE, ZSCALE = ep.XSCALE, ep.YSCALE, ep.ZSCALE
VOLUMESCALE = ep.VOLUMESCALE
AREASCALE = ep.AREASCALE


def get_edge_connectivity(slices: slice, max_z: int):
    """
    Returns tags indicating connectivity of 3d object to top or bottom or both. Such data should be
    excluded from calculations due to the possibility of it being cut off.

    :param slices: slice object
    :param max_z: max z based on z dimension of original stack
    :return: tags indicating connectivity of 3d object to top or bottom or both.
    """
    connectedbot = False
    connectedtop = False
    minz = 0
    assert max_z > 0, "maximum z must be greater than 0 to detect edge connectivity"
    if max_z == 0:
        raise Exception
    top = slices[0].start
    bottom = slices[0].stop
    tags = ""
    if top == 0:
        tags = tags + "t"  # change to t?
        connectedtop = True
    if bottom == max_z:
        tags = tags + "b"
        connectedbot = True
    if tags == "":
        tags = tags + "n"
    #     print(top, bottom, maxz,"tags",tags,slices,bottom==(maxz-1), maxz-1, flush=True)

    return tags, connectedtop, connectedbot


def orientation_3d(bboximage):
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
        zc, xc, yc = pca.components_[0]  # selecting z,y,x values of the first component
        # Calculations of angles for a spherical coordinate system

        r = np.sqrt(zc ** 2 + xc ** 2 + yc ** 2)
        xy = np.sqrt(xc ** 2 + yc ** 2)
        theta = np.arctan2(zc, xy) * 180 / np.pi
        phi = np.arctan2(yc, xc) * 180 / np.pi
    return [r, theta, phi]


def calculate_object_properties(bboxdata, usephull=False, debug=False, small_organelle=False):
    """
    Does calculations for True voxels within a bounding box provided in input. Using the selected
    area reduces calculation time required. Calculations are done for spans along X, Y and Z axes.
    Maximum and minimum feret lengths, centroid coordinates and volume.

    :param bboxdata: 3D data in region of interest (bounding box)
    :param usephull: Use pseudo hull for rotation based calculations instead of the entire object
    :param debug: use when debugging
    :param small_organelle: is channel gfp?
    :return:centroid, volume, xspan, yspan, zspan, maxferet, minferet measurements
    """
    bboxdatabw = (bboxdata > 0)
    miparea = np.nan
    maxferet, minferet, meanferet, sphericity = np.nan, np.nan, np.nan, np.nan
    xspan, yspan, zspan = np.nan, np.nan, np.nan
    # print(np.asarray(center_of_mass(bboxdatabw)),np.array([ZSCALE, XSCALE, YSCALE]) )
    centroid = np.multiply(np.asarray(center_of_mass(bboxdatabw)), np.array([ZSCALE, XSCALE, YSCALE]))
    volume = np.count_nonzero(bboxdatabw) * VOLUMESCALE
    if volume > 0:
        try:
            ns = np.transpose(np.nonzero(bboxdatabw))
            # NOTE: new calculations
            zspan = (ns.max(axis=0)[0] - ns.min(axis=0)[0] + 1) * ZSCALE
            xspan = (ns.max(axis=0)[1] - ns.min(axis=0)[1] + 1) * XSCALE
            yspan = (ns.max(axis=0)[2] - ns.min(axis=0)[2] + 1) * YSCALE
            # proj2dshadow = np.max(bboxdatabw, axis=0) > 0 # same as np.any
            proj2dshadow = np.any(bboxdatabw, axis=0)
            miparea = np.count_nonzero(proj2dshadow) * AREASCALE
            # if (zspan>ZSCALE) and (xspan>XSCALE) and (yspan>YSCALE):
            if not small_organelle:
                try:
                    sphericity = getsphericity(bboxdatabw, volume)
                except:
                    pass
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
    return centroid, volume, xspan, yspan, zspan, maxferet, meanferet, minferet, miparea, sphericity


def dilate_bbox_uniform(ip_bbox, m=0):
    """
    'dilates'/pads given ndimensional matrix with given value

    :param ip_bbox:
    :param m:
    :return:
    """
    assert isinstance(m, int), "number of dilations must be an integer"
    # dilate_boundary
    n_dim_pad = tuple([(m, m)] * ip_bbox.ndim)
    op_bbox = np.pad(ip_bbox, pad_width=n_dim_pad)
    return op_bbox


def dilate_boundary(bbox, m=0):
    """
    performs 'binary' dilation on integer matrix.

    :param bbox:
    :param m:
    :return:
    """
    assert isinstance(m, int), "number of dilations must be an integer"
    # print("before", bbox.shape, np.unique(bbox), np.count_nonzero(bbox), m)
    assert len(np.unique(bbox)) <= 2, f"Input bounding box must contain upto 2 values, currently{len(np.unique(bbox))}"
    val = max(np.unique(bbox))
    bbox = binary_dilation(bbox > 0, iterations=m) * val
    # print("after", bbox.shape, np.unique(bbox), np.count_nonzero(bbox))
    # dilate object
    return bbox


def pad_3d_slice(ip_slice_obj, pad_length, stackshape):
    """
    pads a 3d slice taking into account the limits of the original stack. This roundabout method is required to avoid passing the entire stack as argument which slows calculations down.

    :param ip_slice_obj:
    :param pad_length:
    :param stackshape:
    :return:
    """
    modified_slice_obj = []
    slicediffs = []
    ideal_shifted_slice = []
    for i, ip_slice in enumerate(ip_slice_obj):
        start_0 = ip_slice.start - pad_length
        stop_0 = ip_slice.stop + pad_length
        start = max(start_0, 0)
        stop = min(stop_0, stackshape[i])
        new_slice = slice(start, stop, ip_slice.step)
        # if ignore_first_dim and i == 0:
        #     new_slice = ip_slice
        modified_slice_obj.append(new_slice)
        # diffpad value can only be positive as we can only remove values
        slicediff = (np.abs(start - start_0), np.abs(stop - stop_0))
        slicediffs.append(slicediff)
        #
        shifted_slice = slice(pad_length, ip_slice.stop - ip_slice.start + pad_length, ip_slice.step)
        ideal_shifted_slice.append(shifted_slice)
    return tuple(modified_slice_obj), slicediffs, tuple(ideal_shifted_slice)


def phantom_pad(bbox, slicediff):
    """
    Pads the box with given pad dimensions
    :param bbox:
    :param slicediff:
    :return:
    """
    assert bbox.ndim == len(slicediff), f"dimensions for bbox and slicediff must match, " \
                                        f"currently bbox:{bbox.ndim}, slicediff:{len(slicediff)}"
    op_bbox = np.pad(bbox, pad_width=slicediff)
    return op_bbox


def distance_from_wall_2d(org_bbox, cell_bbox, returnmap=False, axis=0, usescale=True, scales=None, temppath=""):
    """
    calculates the mean and standard deviation of distance of each pixel from the wall for each frame layer-by-layer
    Data must be in the form : Z, X, Y -> axis 0 is assumed to be z.
    :param org_bbox: bounding box with segmented organelle, pre-dilated if m_dilations !=0
    :param cell_bbox: bounding box with corresponding segmented cell, undilated
    :param returnmap : returns euclidean distance transform map
    :param axis : axis along which all frames are considered.
    :param usescale : scales euclidean distance map based on resolution
    :param scales :
    :return: mean and std of distance of each pixel from cell border
    """

    assert org_bbox.shape == cell_bbox.shape, "bounding boxes of organelle and enclosing cell must be equal"
    assert axis < cell_bbox.ndim
    ###################################################################################
    # minscale = None
    if usescale:
        if scales is None:
            # minscale = min([XSCALE, YSCALE])
            scales = np.array([XSCALE, YSCALE])  # / minscale
    else:
        scales = [1, 1]
    ###################################################################################
    org_bbox = org_bbox > 0
    cell_bbox = (cell_bbox > 0) * 1
    cell_bbox_inv = (cell_bbox == 0) * 1
    # print(f"DEBUG: cell:{np.shape(cell_bbox), np.count_nonzero(cell_bbox)}, "
    #       f"cellinv:{np.shape(cell_bbox_inv), np.count_nonzero(cell_bbox_inv)}, "
    #       f"SUM: {np.count_nonzero(cell_bbox) + np.count_nonzero(cell_bbox_inv)}")
    dims = cell_bbox.shape
    org_map_n = np.full(cell_bbox.shape, fill_value=np.nan)
    # distance map for cell
    for z in range(dims[axis]):
        ed_map_out, ed_map_in = None, None
        org2d = org_bbox[z, :, :].squeeze()
        cell2d = cell_bbox[z, :, :].squeeze()
        cell2d_inv = cell_bbox_inv[z, :, :].squeeze()

        if len(np.unique(cell2d)) < 2:
            ed_map_in = np.zeros_like(cell2d)
            # print(f"skipping z = {z}, uniques: {np.unique(cell2d)}, {np.unique(org2d)}")
        else:
            ed_map_in = distance_transform_edt(cell2d, sampling=scales)

        # Calculate edt and inverse edt
        if len(np.unique(cell2d_inv)) < 2:
            ed_map_out = np.zeros_like(cell2d_inv)
            # print(f"skipping z = {z}, uniques: {np.unique(cell2d)}, {np.unique(org2d)}")
        else:
            ed_map_out = distance_transform_edt(cell2d_inv, sampling=scales)
        # Combine edt and inverse edt
        ed_map = ed_map_in - ed_map_out
        # print("MIN edmapin: ", np.min(ed_map_in), "edmap: ", np.min(ed_map), "edmapout: ", np.min(ed_map_out), end="\t")
        # print("MAX edmapin: ", np.max(ed_map_in), "edmap: ", np.max(ed_map), "edmapout: ", np.max(ed_map_out), end="\t")
        # print(f"DEBUG: edin:{np.shape(ed_map_in), np.count_nonzero(ed_map_in)},"
        #       f"\tedout:{np.shape(ed_map_out), np.count_nonzero(ed_map_out)}"
        #       f"\ted_map: {np.shape(ed_map), np.count_nonzero(ed_map)}")
        # print(f"minscale: {minscale}")
        # print(f"DEBUG: edin:{np.shape(ed_map_in), np.count_nonzero(ed_map_in)},"
        #       f"\tedout:{np.shape(ed_map_out), np.count_nonzero(ed_map_out)}"
        #       f"\ted_map: {np.shape(ed_map), np.count_nonzero(ed_map)}")
        mask2d = org2d > 0
        org_map = ed_map * mask2d
        org_map_n[z, :, :] = org_map
    # print(f"MASK2d:MINorgmap: {np.min(org_map_n)}, MAXorgmap: {np.max(org_map_n)}")
        # average and sd

    # OmeTiffWriter.save(data=org_map_n, uri=f"{temppath}_orgmapn2d.tiff", overwrite_file=True)
    org_map_nonzero = org_map_n[np.nonzero(org_bbox)]
    m = np.mean(org_map_nonzero)  # * minscale
    s = np.std(org_map_nonzero)  # * minscale
    if returnmap:
        return m, s, org_map_n, cell_bbox
    return m, s


def distance_from_wall_3d(org_bbox, cell_bbox, returnmap=False, usescale=True, scales=None, temppath=""):
    """
    calculates the mean and standard deviation of distance of each pixel from the wall in 3D
    :param org_bbox : bounding box with segmented organelle
    :param cell_bbox : bounding box with corresponding segmented cell
    :param returnmap : returns euclidean distance transform map
    :param usescale :
    :param scales :

    :return: mean and std of distance of each pixel from cell border
    """
    assert org_bbox.shape == cell_bbox.shape, "bounding boxes of organelle and enclosing cell must be equal"

    ###################################################################################
    # minscale = None
    if usescale:
        if scales is None:
            # minscale = min([ZSCALE, XSCALE, YSCALE])
            scales = np.array([ZSCALE, XSCALE, YSCALE])  # / minscale
    else:
        scales = [1, 1, 1]
    ###################################################################################

    cell_bbox = (cell_bbox > 0) * 1
    cell_bbox_inv = (cell_bbox == 0) * 1
    # print(f"DEBUG: cell:{np.shape(cell_bbox)}, {np.count_nonzero(cell_bbox)}, "
    #       f"cellinv:{np.shape(cell_bbox_inv)},{np.count_nonzero(cell_bbox_inv)}, "
    #       f"SUM: {np.count_nonzero(cell_bbox) + np.count_nonzero(cell_bbox_inv)}")
    # distance map for cell
    ed_map_in = distance_transform_edt(cell_bbox, sampling=scales)
    ed_map_out = distance_transform_edt(cell_bbox_inv, sampling=scales)
    # Combine edt and inverse edt
    ed_map = ed_map_in - ed_map_out
    # print("MIN edmapin: ", np.min(ed_map_in), "edmap: ", np.min(ed_map), "edmapout: ", np.min(ed_map_out))
    # print("MAX edmapin: ", np.max(ed_map_in), "edmap: ", np.max(ed_map), "edmapout: ", np.max(ed_map_out))
    # print(f"DEBUG: edin:{np.shape(ed_map_in), np.count_nonzero(ed_map_in)},"
    #       f"\tedout:{np.shape(ed_map_out), np.count_nonzero(ed_map_out)}"
    #       f"\ted_map: {np.shape(ed_map), np.count_nonzero(ed_map)}")
    # distance map for organelle locations
    mask = org_bbox > 0
    org_map = ed_map * mask
    # print(f"MASK3d, MINorgmap: {np.min(org_map)}, MAXorgmap: {np.max(org_map)}")
    # average and sd
    # OmeTiffWriter.save(data=org_map, uri=f"{temppath}_orgmap3d.tiff", overwrite_file=True)

    org_map_nonzero = org_map[np.nonzero(org_bbox)]
    m = np.mean(org_map_nonzero)  # * minscale
    s = np.std(org_map_nonzero)  # * minscale
    if returnmap:
        return m, s, org_map, cell_bbox
    return m, s


def distance_from_centroid_2d(org_bbox, cell_centroid):
    """
    calculates the mean and standard deviation of distance of each pixel from the wall
    :param org_bbox: bounding box with segmented organelle
    :param cell_bbox: bounding box with corresponding segmented cell
    :return: mean and std of distance of each pixel from cell border
    """
    org_points = np.asarray(np.where(org_bbox > 0)).T  # coordinates of all relevant points
    dists = np.sum(np.square(org_points - cell_centroid))  # euclidean distance for all points
    m, s = dists.mean(), dists.std()
    return m, s


def getsphericity(bboxdata, volume):
    bboxdata = bboxdata.squeeze()
    # assert bboxdata.ndim == 3, f"sphericity inputs must be 3 dimensional, currently: {bboxdata.ndim} dimensional"
    assert bboxdata.ndim == 3
    verts, faces, normals, values = marching_cubes(bboxdata, 0)  # levelset set to 0 for outermost contour
    surface_area = mesh_surface_area(verts, faces)
    sphericity = (36 * np.pi * volume ** 2) ** (1. / 3.) / surface_area
    return sphericity


def organellecentroid_samerefframe(bboxdata):
    bboxdatabw = (bboxdata > 0)
    centroid = np.multiply(np.asarray(center_of_mass(bboxdatabw)), np.array([ZSCALE, XSCALE, YSCALE]))
    return centroid


def calculate_multiorganelle_properties(org_bboxdata, ref_centroid):
    """
    Note: Dimension must be in the order: z,x,y
    feature measurements for individual organelles (within a masked cell)

    :param org_bboxdata: 3D data in padded region of interest
    :param ref_centroid: location of cell or dna centroid
    :param cellobj:
    :param d2wbbox:

    :return:
    Returns the following metrics
    :param organellecounts: Count of organelles within boundingbox
    :param centroids: center of mass (with all voxels weighted equally) giving the geometric centroid.
    :param volumes: volume of all organelles within cell bounding box.
    :param xspans: X-spans of all organelles within cell bounding box.
    :param yspans: Y-spans of all organelles within cell bounding box.
    :param zspans: Z-spans of all organelles within cell bounding box.
    :param maxferets: maximum feret of all organelles within cell bounding box.
    :param meanferets: mean ferets of all organelles within cell bounding box.
    :param minferets: minimum ferets of all organelles within cell bounding box.
    :param mipareas: Maximum intensity projection of all organelles within cell bounding box.
    :param orientations3D: (r,theta,phi) values for all organelles within cell bounding box.
    :param z_distributions: Z-distribution from the cell centroid of all organelles within cell bounding box.
    :param radial_distribution2ds: 2D radial distribution from cell centroid of all organelles within cell bounding box.
    :param radial_distribution3ds: 3D radial distribution from cell centroid of all organelles within cell bounding box.
    :param meanvolume: mean volume per cell bounding box.


    """
    mno = ep.MAX_ORGANELLE_PER_CELL
    # mno = 1 # temporary
    # centerz = ep.Z_FRAMES_PER_STACK//2+1

    volumes, xspans, yspans, zspans, maxferets, meanferets, minferets, mipareas, sphericities = np.nan * np.ones(
        mno), np.nan * np.ones(mno), np.nan * np.ones(mno), np.nan * np.ones(mno), np.nan * np.ones(
        mno), np.nan * np.ones(
        mno), np.nan * np.ones(mno), np.nan * np.ones(mno), np.nan * np.ones(mno)
    z_distributions, radial_distribution3ds, radial_distribution2ds = np.nan * np.ones(mno), np.nan * np.ones(
        mno), np.nan * np.ones(mno)
    organellelabel, organellecounts = label(org_bboxdata > 0)
    org_df = pd.DataFrame(np.arange(1, organellecounts + 1, 1), columns=['organelle_index'])

    centroids, orientations_3d = np.nan * np.ones((mno, 3)), np.nan * np.ones((mno, 3))
    # orgcentroid = np.multiply(np.asarray(center_of_mass(bboxdata)), np.array([ZSCALE, XSCALE, YSCALE])) # NOT Cellcentroid

    for index, row in org_df.iterrows():
        if index < mno:
            org_index = int(row['organelle_index'])
            organelle_obj = organellelabel == org_index
            bboxcrop = find_objects(organelle_obj)
            gfpslices = bboxcrop[0]  # slices for gfp channel
            # All properties obtained from calcs are already scaled
            _, volume, xspan, yspan, zspan, maxferet, meanferet, minferet, miparea, _ = calculate_object_properties(
                organelle_obj[gfpslices],
                small_organelle=True)
            # distribution calculations
            centroid_rel = organellecentroid_samerefframe(organelle_obj)
            gfp_c_rel = centroid_rel - ref_centroid
            # centroid location needs to be relative to the cell based slices
            z_dist = gfp_c_rel[0]  # distance from centroid of the cell
            # print("centroid_rel",centroid_rel," cell_centroid", cell_centroid,
            #       (gfp_c_rel)/(ep.ZSCALE, ep.XSCALE, ep.YSCALE), gfp_c_rel)
            radial_distribution2d = (gfp_c_rel[1] ** 2 + gfp_c_rel[2] ** 2) ** (1 / 2)
            radial_distribution3d = (gfp_c_rel[0] ** 2 + gfp_c_rel[1] ** 2 + gfp_c_rel[2] ** 2) ** (1 / 2)
            orientation3D = orientation_3d(organelle_obj[gfpslices])
            # orientations - PCA
            centroids[index, :] = np.array(centroid_rel)
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
            orientations_3d[index, :] = np.array(orientation3D)


        else:
            print(f"more than {mno} organelles found: {organellecounts}")
    try:
        meanvolume = np.nanmean(volumes)
    except RuntimeWarning:
        meanvolume = np.nan
    # except Exception as e:
    #     print(e, traceback.format_exc())
    return organellecounts, centroids, volumes, xspans, yspans, zspans, maxferets, meanferets, minferets, mipareas, \
           orientations_3d, z_distributions, radial_distribution2ds, radial_distribution3ds, meanvolume


if __name__ == "__main__":
    pass
