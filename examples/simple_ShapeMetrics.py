import os
import re
import numpy as np
import pandas as pd
import warnings
from analysis.stackio import stackio
from scipy.ndimage.measurements import label, find_objects, center_of_mass
from skimage.measure import marching_cubes, mesh_surface_area
from sklearn.decomposition import PCA
from aicsimageio.writers import OmeTiffWriter
from aicsimageio import AICSImage
from analysis.AnalysisTools import conv_hull
from analysis.AnalysisTools import experimentalparams as ep

XSCALE_822, YSCALE_822, ZSCALE_822 = 0.0705908, 0.0705908, 0.2300000
VOLUMESCALE_822 = XSCALE_822 * YSCALE_822 * ZSCALE_822
if __name__ == "__main__":
    path_tomseg = ""
    path_cellseg = ""
    savepath_tmm20 = ""
    flist_tom = [f for f in os.listdir(path_tomseg) if os.path.isfile(os.path.join(path_tomseg, f))]
    flist_cell = [f for f in os.listdir(path_cellseg) if os.path.isfile(os.path.join(path_cellseg, f))]
    total = 0
    # print(len(flist_cell), len(flist_tom))
    maxcells = 150
    max_org_per_cell = 250
    line_names = sorted(['1085A1', '1085A2', '1097F1', '48B1', '48B2', 'BBS10B1', 'BBS16B2', 'D3C', 'LCA5A1', 'LCA5A2',
                         'LCA5B2', 'TJP11'])
    transwells = ['1', '2']
    FOVs = ['1', '2']
    max_line_name, max_transwells, max_FOVs = len(line_names), len(transwells), len(FOVs)
    print(f'max_line_name, max_transwells, FOVs, {max_line_name, max_transwells, FOVs}')
    tomVolumes = np.full((max_line_name, max_transwells, max_FOVs, maxcells, max_org_per_cell), np.nan)
    # tomVolumes = np.full((maxcells, max_org_per_cell), np.nan)
    cellVolumes = np.full((max_line_name, max_transwells, max_FOVs, maxcells), np.nan)
    tomCount = np.full((max_line_name, max_transwells, max_FOVs, maxcells), np.nan)
    for i, t in enumerate((sorted(flist_tom))):
        splittom = re.split(r'[_.-]', t)
        line_name, transwell, chname, FOV = splittom[0], splittom[1], splittom[3], splittom[8]
        for j, c in enumerate(sorted(flist_cell)):
            try:
                splitcell = re.split(r'[_.-]', c)
                if (splittom[0], splittom[1], splittom[3], splittom[8]) == (
                        splitcell[0], splitcell[1], splitcell[3], splitcell[8]):
                    total += 1
                    ln, twn, fv = line_names.index(line_name), transwells.index(transwell), FOVs.index(FOV)
                    # print(i + 1, j + 1, total, line_name, transwell, chname, FOV)

                    docalculations = True
                    if docalculations:
                        img_T = stackio.opensegmentedstack(os.path.join(path_tomseg, t))
                        img_Z = stackio.opensegmentedstack(os.path.join(path_cellseg, c), whiteonblack=False)
                        img_C = np.zeros((img_T.shape)) + img_Z
                        lbltom = stackio.getlabelledstack(img_T)
                        lblcell = stackio.getlabelledstack(img_C)
                        # OmeTiffWriter.save(data=lblcell.astype(int), uri="Cell_lbl.tiff", overwrite_file=True)
                        obj_df = pd.DataFrame(np.arange(1, len(np.unique(lblcell)), 1),
                                              columns=['object_index'])
                        for index, row in obj_df.iterrows():
                            try:
                                cellinputdict = {}
                                obj_index = int(row['object_index'])
                                objs = lblcell == obj_index
                                bbox_cell = find_objects(objs)
                                # print(bbox_cell, flush=True)
                                slices = bbox_cell[0]
                                CellObject = objs[slices]
                                bboxdatabw = (CellObject > 0)
                                cellVolumes[ln, twn, fv, index] = np.count_nonzero(bboxdatabw) * VOLUMESCALE_822
                                GFPObjects = (lbltom[slices] & CellObject)
                                organellelabel, organellecounts = label(GFPObjects > 0)
                                org_df = pd.DataFrame(np.arange(1, np.unique(organellecounts), 1),
                                                      columns=['organelle_index'])
                                # print(organellecounts, len(np.unique(lblcell + 1)))
                                for index_t, row_t in org_df.iterrows():
                                    # max_line_name, max_transwells, FOVs, maxcells
                                    tomVolumes[ln, twn, fv, index, index_t] = np.count_nonzero(
                                        bboxdatabw) * VOLUMESCALE_822
                                    tomCount[ln, twn, fv, index] = organellecounts
                            except Exception as err:
                                print("inerr", bbox_cell, np.unique(lblcell), obj_index)
                                print(err)
                                # exit()
            except Exception as e:
                print(e)
                print(re.split(r'[_.-]', t), len(re.split(r'[_.-]', t)))
                print(re.split(r'[_.-]', c), len(re.split(r'[_.-]', c)))
                # exit()
    channels = ["Cell", "TOM", "TOM"]
    properties = ["Volume", "Volume", "Count"]
    data = [cellVolumes, tomVolumes, tomCount]
    for i, (channel, propertyname, values_mat) in enumerate(zip(channels, properties, data)):
        filename = f"{channel}_{propertyname}.npz"
        fpath = os.path.join(savepath_tmm20, filename)
        stackio.saveproperty(values_mat, filepath=fpath, type="npz")
        stackio.saveproperty(values_mat, filepath=fpath, type="npz")
        loaded = stackio.loadproperty(fpath)
        success = stackio.checksavedfileintegrity(loaded, values_mat)
        if success:
            print(f"SAVE SUCCESSFUL FOR {filename}\t\tNo. of Datapoints: {np.count_nonzero(~np.isnan(values_mat))}")
        else:  # (2, 4, 1, 5, 6, 1000, 50)
            print(loaded.files, loaded[loaded.files[0]].shape, propertyname.shape)
