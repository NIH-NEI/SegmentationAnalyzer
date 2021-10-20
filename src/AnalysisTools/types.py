from pathlib import Path
from typing import List, NamedTuple, Optional, Union
import numpy as np

PathLike = Union[str, Path]
SegmentationLike = Union[str, bool]
Stacklike = np.ndarray  # ndim = 3?
ArrayLike = Union[np.ndarray]
