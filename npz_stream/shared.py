from dataclasses import dataclass
import numpy as np


@dataclass(kw_only=True)
class ArraySpec:
  dtype: np.dtype
  fortran_order: bool
  shape: tuple[int, ...]
