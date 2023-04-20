import math
from typing import IO, Optional

import numpy as np

from .npy_utils import encode_npy_header
from .utils import ArrayOrder, ArrayShape, isfortran, isolate_cursor, write_array


class NpyStreamWriter:
  """
  An uncompressed array stream writer.
  """

  def __init__(
    self,
    file: IO[bytes],
    /,
    *,
    dtype: Optional[np.dtype | str] = None,
    order: Optional[ArrayOrder] = None,
    shape: Optional[ArrayShape] = None
  ):
    """
    Creates an uncompressed array stream writer.

    Parameters
      dtype: The data type. Will be inferred from the first array provided to write() if missing. Required if flush() is called without any preceding call to write().
      file: The output binary file.
      order: The memory order of the stored array. Either `C` for C-contiguous array or `F` for a Fortran-contiguous array. Will be inferred from the first array provided to write() if missing, defaulting to `C` if that array is neither C-contiguous or Fortran-contiguous, if it is both, or if flush() is called without any preceding call to write().
      shape: The shape of the appended arrays. Will be inferred from the first array provided to write() if missing. Required if flush() is called without any preceding call to write().
    """

    self._dtype = np.dtype(dtype) if dtype is not None else None
    self._fortran_order = (order == 'F') if order is not None else None
    self._shape = shape

    self._file = file
    self._length = 0
    self._start_offset = file.tell()

  def flush(self):
    """
    Finishes writing the array and flushes the written data to the underlying file.

    This function does not close the underlying file.
    """

    if self._dtype is None:
      raise ValueError("Unknown dtype")
    if self._shape is None:
      raise ValueError("Unknown shape")

    with isolate_cursor(self._file):
      self._file.seek(self._start_offset)
      self._file.write(encode_npy_header(
        dtype=self._dtype,
        fortran_order=(not not self._fortran_order),
        shape=((*self._shape, self._length) if self._fortran_order else (self._length, *self._shape))
      ))

    self._file.flush()

  def write(self, arr: np.ndarray, /):
    """
    Appends an array to the output file.

    Parameters
      arr: The array to append. It must have the same dtype and shape as previous arrays provided to write(), or match the parameters provided to __init__(), if any.
    """

    if self._fortran_order is None:
      self._fortran_order = isfortran(arr)

    if self._dtype is None:
      self._dtype = arr.dtype
    elif arr.dtype != self._dtype:
      raise ValueError("Invalid dtype")

    if self._shape is None:
      self._shape = arr.shape
    elif arr.shape != self._shape:
      raise ValueError("Invalid shape")

    if self._length < 1:
      self._file.write(b"\x00" * len(encode_npy_header(
        dtype=self._dtype,
        fortran_order=self._fortran_order,
        partial=False,
        shape=self._shape
      )))

    self._length += 1

    write_array(self._file, arr, fortran_order=self._fortran_order)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.flush()


class NpyStreamReader:
  """
  An uncompressed array stream reader.
  """

  def __init__(self, file: IO[bytes], /):
    """
    Creates an uncompressed array stream reader.

    Parameters
      file: The input binary file.
    """

    version = np.lib.format.read_magic(file)
    shape, self._fortran_order, self._dtype = np.lib.format._read_array_header(file, version) # type: ignore

    self._data_offset = file.tell()
    self._file = file

    if self._fortran_order:
      *self._shape, self._length = shape
    else:
      self._length, *self._shape = shape

    self._stride = self._dtype.itemsize * math.prod(self._shape)

  def __getitem__(self, index: int, /):
    self._file.seek(self._data_offset + index * self._stride)
    data = self._file.read(self._stride)

    return np.frombuffer(data, dtype=self._dtype).reshape(self._shape, order=('F' if self._fortran_order else 'C'))

  def __iter__(self):
    for index in range(self._length):
      yield self[index]

  def __len__(self):
    return self._length


__all__ = [
  'NpyStreamReader',
  'NpyStreamWriter'
]
