import contextlib
from typing import IO, Literal, cast
from zlib import crc32

import numpy as np
import numpy.compat


ArrayShape = tuple[int, ...]
ArrayOrder = Literal['C', 'F']
CastingRule = Literal['equiv', 'no', 'safe', 'same_kind', 'unsafe']


def combine_crc32(a_data: bytes, b_crc32: int, b_size: int):
  return crc32(a_data + bytes(b_size)) ^ b_crc32 ^ crc32(bytes(len(a_data) + b_size))

def isfortran(arr: np.ndarray, /):
  return (not arr.flags.c_contiguous) and arr.flags.f_contiguous

@contextlib.contextmanager
def isolate_cursor(file: IO[bytes], /):
  position = file.tell()

  try:
    yield
  finally:
    file.seek(position)

def iter_array_chunks(arr: np.ndarray, *, fortran_order: bool):
  buffer_size = max(16 * 1024 ** 2 // arr.itemsize, 1)

  for chunk in np.nditer(
    arr,
    buffersize=buffer_size,
    flags=['buffered', 'external_loop', 'zerosize_ok'],
    order=('F' if fortran_order else 'C')
  ):
    yield cast(bytes, chunk.tobytes()) # type: ignore

def write_array(file: IO[bytes], arr: np.ndarray, *, fortran_order: bool):
  if numpy.compat.isfileobj(file):
    (arr.T if fortran_order else arr).tofile(file)
  else:
    for chunk in iter_array_chunks(arr, fortran_order=fortran_order):
      file.write(chunk)
