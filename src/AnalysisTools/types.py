from pathlib import Path
from typing import List, Union

import numpy as np

PathLike = Union[str, Path]
SegmentationLike = Union[str, bool]
Stacklike = np.ndarray
ArrayLike = Union[list, tuple, np.ndarray, set]
strlist = List[str]
# TODO: nestedlistlike ?
