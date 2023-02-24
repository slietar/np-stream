from dataclasses import dataclass
import functools
import gzip
from io import BufferedIOBase, IOBase, RawIOBase
import io
import operator
import struct
import time
from typing import IO, BinaryIO, Literal, Optional
from zipfile import ZipFile
from zlib import crc32
import zlib
import numpy as np

from .npy import encode_npy_header
from .shared import ArraySpec
from .zip import encode_zip, get_zip_offset


NPY_HEADER_SIZE = 128


class NpyStreamWriter:
  def __init__(self, file: IOBase, /, *, name: str = "arr_0"):
    self._file = file
    self._length = 0
    self._spec: Optional[ArraySpec] = None

  def flush(self):
    assert self._spec

    self._file.seek(0)
    self._file.write(encode_npy_header(ArraySpec(
      dtype=self._spec.dtype,
      fortran_order=self._spec.fortran_order,
      shape=((*self._spec.shape, self._length) if self._spec.fortran_order else (self._length, *self._spec.shape))
    )))

    self._file.flush()

  def write(self, arr: np.ndarray, /):
    self._length += 1

    spec = ArraySpec(
      dtype=arr.dtype,
      fortran_order=np.isfortran(arr),
      shape=arr.shape
    )

    if self._spec is None:
      self._spec = spec
      self._file.write(b"\x00" * 128)
    else:
      assert spec == self._spec

    self._file.write(arr.data)


class NpzStreamWriter:
  def __init__(self, file: BufferedIOBase | RawIOBase, /, *, name: str = "arr_0.npy"):
    self._file = file
    self._length = 0
    self._name = name
    self._spec: Optional[ArraySpec] = None

    self._compressor = zlib.compressobj(2)
    self._crc32 = crc32(bytes(NPY_HEADER_SIZE))
    self._size_compressed = NPY_HEADER_SIZE + struct.calcsize("<BHH")
    self._size_uncompressed = NPY_HEADER_SIZE

    self._zip_start_offset = file.tell()
    self._file.write(bytes(get_zip_offset(file_name=self._name)))

    self._npy_start_offset = file.tell()
    self._file.write(bytes(self._size_compressed))

    # Test
    # self.__data = bytes()

  def flush(self):
    assert self._spec

    npy_header = encode_npy_header(ArraySpec(
      dtype=self._spec.dtype,
      fortran_order=self._spec.fortran_order,
      shape=((*self._spec.shape, self._length) if self._spec.fortran_order else (self._length, *self._spec.shape))
    ))

    # self.__data = npy_header + self.__data
    self._crc32 = combine_crc32(npy_header, self._crc32, self._size_uncompressed - NPY_HEADER_SIZE)

    compressed_data = self._compressor.flush()
    self._size_compressed += len(compressed_data)

    self._file.write(compressed_data)
    self._file.seek(-4, io.SEEK_CUR)
    self._file.write(struct.pack(">I", 0))

    encode_zip(
      self._file,
      compressed_size=self._size_compressed,
      compression='deflate',
      crc32=self._crc32,
      file_name=self._name,
      start_offset=self._zip_start_offset,
      uncompressed_size=self._size_uncompressed
    )

    self._file.seek(self._npy_start_offset)
    self._file.write(struct.pack("<BHH", 0b00, NPY_HEADER_SIZE, NPY_HEADER_SIZE ^ 0xffff))
    self._file.write(npy_header)

  def write(self, arr: np.ndarray, /):
    spec = ArraySpec(
      dtype=arr.dtype,
      fortran_order=np.isfortran(arr),
      shape=arr.shape
    )

    if self._spec is None:
      self._spec = spec
    else:
      assert spec == self._spec

    self._crc32 = crc32(arr.data, self._crc32)
    compressed_data = self._compressor.compress(arr.data)

    if self._length < 1:
      compressed_data = compressed_data[2:]

    self._size_compressed += len(compressed_data)
    self._size_uncompressed += arr.data.nbytes

    self._length += 1

    # print(f"Writing {(self._size_compressed / self._size_uncompressed * 100):.2f}%")

    # self.__data += arr.data
    return self._file.write(compressed_data)


class NpzStreamReader:
  def __init__(self, file: IO[bytes], /):
    self._file = ZipFile(file)

  def __getitem__(self, name: str):
    file = self._file.open(name + '.npy')
    # print(file.read(10))
    version = np.lib.format.read_magic(file)
    print(version)

    shape, fortran_order, dtype = np.lib.format._read_array_header(file, version, max_header_size=np.lib.format._MAX_HEADER_SIZE) # type: ignore

    return NpzStreamFile(
      file,
      dtype=dtype,
      fortran_order=fortran_order,
      offset=file.tell(),
      shape=shape
    )

    # return memmap(
    #   file,
    #   dtype=dtype,
    #   offset=file.tell(),
    #   order=('F' if fortran_order else 'C'),
    #   shape=shape
    # )


class NpzStreamFile:
  def __init__(self, file: IO[bytes], /, *, dtype: np.dtype, fortran_order: bool, offset: int, shape: tuple[int, ...]):
    self._file = file
    self._offset = offset

    self._dtype = dtype
    self._fortran_order = fortran_order
    self._shape = shape

    if self._fortran_order:
      *self._inner_shape, self._length = shape
    else:
      self._length, *self._inner_shape = shape

  @property
  def _stride(self):
    return functools.reduce(operator.mul, self._inner_shape, 1) * self._dtype.itemsize

  def __getitem__(self, index: int):
    self._file.seek(self._offset + index * self._stride)
    data = self._file.read(self._stride)

    return np.frombuffer(data, dtype=self._dtype).reshape(self._inner_shape, order=('F' if self._fortran_order else 'C'))

  def __iter__(self):
    for index in range(self._length):
      yield self[index]

  def __len__(self):
    return self._length


# class memmap(np.ndarray):
#   # __array_priority__ = -100.0
#   _file: IO[bytes]
#
#   def __new__(cls, file: IO[bytes], *, dtype: np.dtype, offset: int = 0, order: Literal['C', 'F'], shape: tuple[int, ...]):
#     self = super().__new__(cls, dtype=dtype, order=order, shape=shape)
#     self._file = file
#
#     return self


def combine_crc32(a_data: bytes, b_crc32: int, b_size: int):
  return crc32(a_data + bytes(b_size)) ^ b_crc32 ^ crc32(bytes(len(a_data) + b_size))
