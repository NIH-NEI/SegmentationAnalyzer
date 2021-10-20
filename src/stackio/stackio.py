from src.AnalysisTools import types
import traceback

import numpy as np
from aicsimageio import AICSImage  # ,#omeTifWriter
from tifffile import imread

from src.AnalysisTools import types


#
# def aimg_simple(name):
#     seg = AICSImage(name)
#     seg = 1 * (seg.data > 0)
#     return np.squeeze(seg)
#
#
# def aimgproc(name):
#     seg = AICSImage(name)
#     seg = seg.data.astype(np.int16)
#     seg = seg.max() - seg  # invert
#     seg = seg.squeeze()
#     seg = seg // seg.max()
#     return seg


def opensegmentedstack(name: types.PathLike, binary: bool = True, whiteonblack: types.SegmentationLike = "default", debug=False):
    """
    TODO: test this
    opens segmented stacks. whiteonblack
    :param debug:
    :param name:
    :param binary:
    :param whiteonblack:
    :return:
    """
    try:
        if debug:
            print(f"opening: {name}, binary: {binary},", end="")
        if binary:
            seg = AICSImage(name)
            if whiteonblack == "default":
                whiteonblack = True
            seg = 1 * (seg.data > 0)
        else:
            seg = imread(name)
            # print(type(seg), seg.shape)
            if whiteonblack == "default":
                whiteonblack = False
            seg = seg.astype(np.int16)
            seg = seg.max() - seg  # invert
            seg = seg.squeeze()
            seg = seg // seg.max()
        seg = seg.squeeze()
        if debug:
            print(f" whiteonblack: {whiteonblack} ||\t DONE, shape = ", seg.shape)
        return seg
    except Exception as e:
        print(e, traceback.format_exc())
