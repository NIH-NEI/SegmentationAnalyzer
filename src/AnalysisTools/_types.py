from pathlib import Path
from typing import List, Union

import numpy as np

PathLike = Union[str, Path]
TupleLike = Union[tuple, List]
SegmentationLike = Union[str, bool]
Stacklike = np.ndarray
ArrayLike = Union[list, tuple, np.ndarray]#, set]
strList = List[str]
# TODO: nestedlistlike ?
