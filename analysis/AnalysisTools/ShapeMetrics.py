import warnings

import numpy as np
import pandas as pd
from imea import measure_2d
from scipy.ndimage.measurements import label, find_objects, center_of_mass
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.morphology import octahedron
from skimage.measure import marching_cubes, mesh_surface_area
from sklearn.decomposition import PCA
from aicsimageio.writers import OmeTiffWriter

from analysis.AnalysisTools import conv_hull
from analysis.AnalysisTools import experimentalparams as ep

XSCALE, YSCALE, ZSCALE = ep.XSCALE, ep.YSCALE, ep.ZSCALE
VOLUMESCALE = ep.VOLUMESCALE
AREASCALE = ep.AREASCALE


def get_edge_connectivity(slices: slice, max_z: int):
    """
    Returns tags indicating connectivity of 3d object to top or bottom or both. Such data should be
    excluded from calculations due to the possibility of it being cut off.

    Args:
        slices: slice object
        max_z: max z based on z dimension of original stack

    Returns:
        tags indicating connectivity of 3d object to top or bottom or both.
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

    Args:
        bboximage: stack dimensions are in order z,x,y

    Returns:
        r, theta, phi values
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

    Args:
        bboxdata: 3D data in region of interest (bounding box)
        usephull: Use pseudo hull for rotation based calculations instead of the entire object
        debug: use when debugging
        small_organelle: is channel gfp?

    Returns:
        centroid, volume, xspan, yspan, zspan, maxferet, minferet measurements
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

    Args:
        ip_bbox: input bounding box
        m: size of padding along all directions

    Returns:
        uniformly padded bounding box
    """
    assert isinstance(m, int), "number of dilations must be an integer"
    # dilate_boundary
    n_dim_pad = tuple([(m, m)] * ip_bbox.ndim)
    op_bbox = np.pad(ip_bbox, pad_width=n_dim_pad)
    return op_bbox


def dilate_boundary_zxy(bbox, m=0, dilatexyonly=True):
    """
    performs 'binary' dilation on integer matrix.

    Args:
        bbox: Dilate boundary along zxy
        m: number of dilations
        dilatexyonly: only dilate along x and y directions.

    Returns:
        dilated bounding box
    """
    assert isinstance(m, int), "number of dilations must be an integer"
    # print("before", bbox.shape, np.unique(bbox), np.count_nonzero(bbox), m)
    assert len(np.unique(bbox)) <= 2, f"Input bounding box must contain upto 2 values, currently{len(np.unique(bbox))}"
    assert bbox.ndim == 3, ""

    structuring_element = np.zeros((3, 3, 3), dtype=int)
    if dilatexyonly:
        structuring_element[1, 1, :] = 1
        structuring_element[1, :, 1] = 1
    else:
        structuring_element = octahedron(1)
    val = max(np.unique(bbox))
    nbbox = binary_dilation(bbox > 0, structure=structuring_element, iterations=m) * val
    # print("after", bbox.shape, np.unique(bbox), np.count_nonzero(bbox))
    # print(f"")
    # dilate object
    return nbbox


def pad_3d_slice(ip_slice_obj, pad_length, stackshape):
    """
    pads a 3d slice taking into account the limits of the original stack.
    This roundabout method is required to avoid passing the entire stack
    as argument which slows calculations down. Takes into account stack
    borders where such padding is not possible.

    Args:
        ip_slice_obj: input slice object
        pad_length: length of padding
        stackshape: np.shape of stack

    Returns:
        modified_slice_obj, actual diffs in slice , ideal shifted slice
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
        shifted_slice = slice(pad_length, ip_slice.stop - ip_slice.start + pad_length, ip_slice.step)
        ideal_shifted_slice.append(shifted_slice)
    return tuple(modified_slice_obj), slicediffs, tuple(ideal_shifted_slice)


def phantom_pad(bbox, slicediff):
    """
    Pads the box with given pad dimensions

    Args:
        bbox: bounding box
        slicediff: actual diffs in slice - caused by stack borders. This object can be obtained from pad_3d_slice

    Returns:
        pads bounding box by given slicediff
    """
    assert bbox.ndim == len(slicediff), f"dimensions for bbox and slicediff must match, " \
                                        f"currently bbox:{bbox.ndim}, slicediff:{len(slicediff)}"
    op_bbox = np.pad(bbox, pad_width=slicediff)
    return op_bbox


def z_dist_top_bottom_extrema(org_bbox, cell_bbox):
    """
    Takes the minimum coordinate in cell and subtracts from all organelle values - this provides the distance
    from bottom of the cell. Returns a mean and standard deviation.

    Args:
        org_bbox: organelle bounding box
        cell_bbox: cell bounding box

    Returns:
        z_dists_bot_mean, z_dists_bot_std, z_dists_top_mean, z_dists_top_std
        - z distance from bot and top and their standard deviations
    """
    cell_voxels = np.transpose(np.nonzero(cell_bbox))
    lowest_z_coord = cell_voxels[:, 0].min()
    highest_z_coord = cell_voxels[:, 0].max()

    organelle_voxels = np.transpose(np.nonzero(org_bbox))
    # 0.5 voxel correction term to account for voxel width from center
    z_distances_bot = np.abs(organelle_voxels[:, 0] - lowest_z_coord) + 0.5
    z_distances_top = np.abs(highest_z_coord - organelle_voxels[:, 0]) + 0.5
    z_dists_bot_mean = z_distances_bot.mean() * ZSCALE
    z_dists_bot_std = z_distances_bot.std() * ZSCALE
    z_dists_top_mean = z_distances_top.mean() * ZSCALE
    z_dists_top_std = z_distances_top.std() * ZSCALE
    return z_dists_bot_mean, z_dists_bot_std, z_dists_top_mean, z_dists_top_std


def z_dist_top_bottom_surface(org_bbox, cell_bbox):
    """
    Takes the minimum coordinate in cell and subtracts from all organelle values - this provides the distance
    from bottom of the cell. Returns a mean and standard deviation.

    Args:
        org_bbox: organelle bounding box
        cell_bbox: cell bounding box
    Returns:
        z_dists_bot_surface_mean, z_dists_bot_surface_std, z_dists_top_surface_mean, z_dists_top_surface_std
        - z distance from bottom and top surfaces and their standard deviations
    """
    cellshape = cell_bbox.shape

    z_bot_coord_at_xy = np.full((cellshape[1], cellshape[2]), np.inf)
    z_top_coord_at_xy = np.full((cellshape[1], cellshape[2]), np.inf)
    for x in range(cellshape[1]):
        for y in range(cellshape[2]):
            z_indices = np.nonzero(cell_bbox[:, x, y])[0]
            if z_indices.size > 0:
                z_bot_coord_at_xy[x, y] = z_indices.min()
                z_top_coord_at_xy[x, y] = z_indices.max()
    organelle_voxels = np.transpose(np.nonzero(org_bbox))
    z_distances_bot = []
    z_distances_top = []
    # print(organelle_voxels.shape)
    # 0.5 added for correction
    for z, x, y in organelle_voxels:
        min_z = z_bot_coord_at_xy[x, y]
        max_z = z_top_coord_at_xy[x, y]
        if (min_z != np.inf) and (max_z != np.inf):
            z_distances_bot.append(z - min_z + 0.5)
            z_distances_top.append(max_z - z + 0.5)
            # print(f"z:{z}, x:{x}, y:{y}, minz:{min_z}, maxz:{max_z}\t")
    z_distances_bot = np.asarray(z_distances_bot) * ZSCALE
    z_distances_top = np.asarray(z_distances_top) * ZSCALE
    # print(f"zbot: {z_distances_bot}, ztop: {z_distances_top}")
    z_dists_bot_surface_mean = z_distances_bot.mean()
    z_dists_bot_surface_std = z_distances_bot.std()
    z_dists_top_surface_mean = z_distances_top.mean()
    z_dists_top_surface_std = z_distances_top.std()
    return z_dists_bot_surface_mean, z_dists_bot_surface_std, z_dists_top_surface_mean, z_dists_top_surface_std


def distance_from_wall_2d(org_bbox, cell_bbox, returnmap=False, axis=0, usescale=True, scales=None, temppath=""):
    """
    calculates the mean and standard deviation of distance of each pixel from the wall for each frame layer-by-layer
    Data must be in the form : Z, X, Y -> axis 0 is assumed to be z.

    Args:
        org_bbox: bounding box with segmented organelle, pre-dilated if m_dilations !=0
        cell_bbox: bounding box with corresponding segmented cell, undilated
        returnmap : returns Euclidean distance transform map
        axis : axis along which all frames are considered.
        usescale : scales Euclidean distance map based on resolution
        scales : scaling factors

    Returns:
        mean and std of distance of each pixel from cell border
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
        # Combine edt and inverse edt. An entire voxel is labelled 0
        ed_map = ed_map_in - ed_map_out

        mask2d = org2d > 0
        org_map = ed_map * mask2d
        org_map_n[z, :, :] = org_map
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

    Args:
        org_bbox : bounding box with segmented organelle
        cell_bbox : bounding box with corresponding segmented cell
        returnmap : returns euclidean distance transform map
        usescale : use scaling factor
        scales : scaling factor

    Returns:
        mean and std of distance of each pixel from cell border
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
    # distance map for cell
    ed_map_in = distance_transform_edt(cell_bbox, sampling=scales)
    ed_map_out = distance_transform_edt(cell_bbox_inv, sampling=scales)
    # Combine edt and inverse edt
    ed_map = ed_map_in - ed_map_out
    mask = org_bbox > 0
    org_map = ed_map * mask

    org_map_nonzero = org_map[np.nonzero(org_bbox)]
    m = np.mean(org_map_nonzero)  # * minscale
    s = np.std(org_map_nonzero)  # * minscale
    if returnmap:
        return m, s, org_map, cell_bbox
    return m, s


def distance_from_centroid_2d(org_bbox, cell_centroid):
    """
    calculates the mean and standard deviation of distance of each pixel from the wall

    Args:
        org_bbox: bounding box with segmented organelle
        cell_bbox: bounding box with corresponding segmented cell
    Returns:
        mean and std of distance of each pixel from cell border
    """
    org_points = np.asarray(np.where(org_bbox > 0)).T  # coordinates of all relevant points
    dists = np.sum(np.square(org_points - cell_centroid))  # euclidean distance for all points
    m, s = dists.mean(), dists.std()
    return m, s


def getsphericity(bboxdata, volume):
    """
    Returns sphericity of segmented region in bounding box by converting to mesh first.

    Args:
        bboxdata: segmented region
        volume: volume of segmented region
    Returns:
        Sphericity value
    """

    bboxdata = bboxdata.squeeze()
    # assert bboxdata.ndim == 3, f"sphericity inputs must be 3 dimensional, currently: {bboxdata.ndim} dimensional"
    assert bboxdata.ndim == 3
    verts, faces, normals, values = marching_cubes(bboxdata, 0)  # levelset set to 0 for outermost contour
    surface_area = mesh_surface_area(verts, faces)
    sphericity = (36 * np.pi * volume ** 2) ** (1. / 3.) / surface_area
    return sphericity


def organellecentroid_samerefframe(bboxdata):
    """
    Calculate centroid using saved axis scales on segmented object in bounding box

    Args:
        bboxdata: bounding box with binary segmentation

    Returns:
        centroid coordinates scaled
    """
    bboxdatabw = (bboxdata > 0)
    centroid = np.multiply(np.asarray(center_of_mass(bboxdatabw)), np.array([ZSCALE, XSCALE, YSCALE]))
    return centroid


def calculate_multiorganelle_properties(org_bboxdata, ref_centroid):
    """
    Note: Dimension must be in the order: z,x,y
    feature measurements for individual organelles (within a masked cell)

    Args:
        org_bboxdata: 3D data in padded region of interest
        ref_centroid: location of cell or dna centroid

    Returns:
        organellecount: Count of organelles within boundingbox.
        centroids: center of mass (with all voxels weighted equally) giving the geometric centroid.
        volumes: volume of all organelles within cell bounding box.
        xspans: X-spans of all organelles within cell bounding box.
        yspans: Y-spans of all organelles within cell bounding box.
        zspans: Z-spans of all organelles within cell bounding box.
        maxferets: maximum feret of all organelles within cell bounding box.
        meanferets: mean ferets of all organelles within cell bounding box.
        minferets: minimum ferets of all organelles within cell bounding box.
        mipareas: Maximum intensity projection of all organelles within cell bounding box.
        orientations3D: (r,theta,phi) values for all organelles within cell bounding box.
        z_distributions: Z-distribution from the cell centroid of all organelles within cell bounding box.
        radial_distribution2ds: 2D radial distribution from cell centroid of all organelles within cell bounding box.
        radial_distribution3ds: 3D radial distribution from cell centroid of all organelles within cell bounding box.
        meanvolume: mean volume per cell bounding box.


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
