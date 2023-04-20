import io
import struct
import zlib
from typing import IO, Optional
from zipfile import ZipFile
from zlib import crc32

import numpy as np

from .npy import NpyStreamReader, encode_npy_header
from .utils import (ArrayOrder, ArrayShape, combine_crc32, isfortran,
                    isolate_cursor, iter_array_chunks)
from .zip import encode_zip, get_zip_offset


class NpzStreamWriter:
  """
  A compressed array stream writer.

  See NpyStreamWriter for details.
  """

  def __init__(
    self,
    file: IO[bytes],
    /,
    *,
    dtype: Optional[np.dtype | str] = None,
    name: str = "arr_0.npy",
    order: Optional[ArrayOrder] = None,
    shape: Optional[ArrayShape] = None
  ):
    """
    Creates a compressed array stream writer.

    Parameters
      name: The name of the file in the resulting archive.
    """

    self._dtype = np.dtype(dtype) if dtype is not None else None
    self._fortran_order = (order == 'F') if order is not None else None
    self._shape = shape

    self._file = file
    self._length = 0
    self._name = name

    self._compressor = zlib.compressobj(2)
    self._compressor_initialized = False

    # Test
    # self.__data = bytes()

  def _init(self):
    assert self._dtype is not None
    assert self._fortran_order is not None
    assert self._shape is not None

    npy_header_size = len(encode_npy_header(
      dtype=self._dtype,
      fortran_order=self._fortran_order,
      partial=True,
      shape=self._shape
    ))

    self._crc32 = crc32(bytes(npy_header_size))
    self._size_compressed = npy_header_size + struct.calcsize("<BHH")
    self._size_uncompressed = npy_header_size

    self._zip_start_offset = self._file.tell()
    bytes_written = self._file.write(bytes(get_zip_offset(file_name=self._name)))

    self._npy_start_offset = self._file.tell()
    bytes_written += self._file.write(bytes(self._size_compressed))

    return bytes_written

  def flush(self):
    if self._dtype is None:
      raise ValueError("Unknown dtype")
    if self._shape is None:
      raise ValueError("Unknown shape")

    if self._fortran_order is None:
      self._fortran_order = False

    bytes_written = 0

    if self._length < 1:
      bytes_written += self._init()

    npy_header = encode_npy_header(
      dtype=self._dtype,
      fortran_order=self._fortran_order,
      shape=((*self._shape, self._length) if self._fortran_order else (self._length, *self._shape))
    )

    npy_header_size = len(npy_header)

    # self.__data = npy_header + self.__data
    self._crc32 = combine_crc32(npy_header, self._crc32, self._size_uncompressed - npy_header_size)

    compressed_data = self._compressor.flush()

    if not self._compressor_initialized:
      compressed_data = compressed_data[2:]
      self._compressor_initialized = True

    self._size_compressed += len(compressed_data)
    bytes_written += self._file.write(compressed_data)

    self._file.seek(-4, io.SEEK_CUR)
    bytes_written += self._file.write(struct.pack(">I", 0))

    bytes_written += encode_zip(
      self._file,
      compressed_size=self._size_compressed,
      compression='deflate',
      crc32=self._crc32,
      file_name=self._name,
      start_offset=self._zip_start_offset,
      uncompressed_size=self._size_uncompressed
    )

    with isolate_cursor(self._file):
      self._file.seek(self._npy_start_offset)
      bytes_written += self._file.write(struct.pack("<BHH", 0b00, npy_header_size, npy_header_size ^ 0xffff))
      bytes_written += self._file.write(npy_header)

    return bytes_written

  def write(self, arr: np.ndarray, /):
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

    bytes_written = 0

    if self._length < 1:
      bytes_written += self._init()

    for chunk in iter_array_chunks(arr, fortran_order=self._fortran_order):
      compressed_chunk = self._compressor.compress(chunk)

      if not self._compressor_initialized:
        compressed_chunk = compressed_chunk[2:]
        self._compressor_initialized = True

      self._crc32 = crc32(chunk, self._crc32)
      self._size_compressed += len(compressed_chunk)

      bytes_written += self._file.write(compressed_chunk)

    self._size_uncompressed += arr.data.nbytes
    self._length += 1

    # print(f"Writing {(self._size_compressed / self._size_uncompressed * 100):.2f}%")

    # self.__data += arr.data
    return bytes_written

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.flush()


class NpzStreamReader:
  """
  A compressed array stream reader.
  """

  def __init__(self, file: IO[bytes], /):
    """
    Creates a compressed array stream reader.

    Parameters
      file: The input binary file.
    """

    self._file = ZipFile(file)

  def __getitem__(self, name: str):
    return NpyStreamReader(self._file.open(name + ".npy"))


__all__ = [
  'NpzStreamReader',
  'NpzStreamWriter'
]
