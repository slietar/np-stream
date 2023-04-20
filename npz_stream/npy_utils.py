import struct

import numpy as np

from .utils import ArrayShape


MAGIC_PREFIX = b"\x93NUMPY"
MAGIC_LEN = len(MAGIC_PREFIX) + 2
ARRAY_ALIGN = 64
BUFFER_SIZE = 2 ** 18
GROWTH_AXIS_MAX_DIGITS = 21

def encode_npy_header(*, dtype: np.dtype, fortran_order: bool, partial: bool = False, shape: ArrayShape):
  options = {
    'descr': (dtype.descr if dtype.names is not None else dtype.str),
    'fortran_order': fortran_order,
    'shape': ((0, shape) if partial else shape)
  }

  header = repr(options).encode()

  if partial:
    header += b" " * (GROWTH_AXIS_MAX_DIGITS - 1)


  hlen = len(header) + 1
  padlen = ARRAY_ALIGN - ((MAGIC_LEN + struct.calcsize("<I") + hlen) % ARRAY_ALIGN)

  header_prefix = MAGIC_PREFIX + b"\x03\x00" + struct.pack("<I", hlen + padlen)
  return header_prefix + header + (b" " * padlen) + b"\n"
