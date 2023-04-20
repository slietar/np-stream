from io import BufferedWriter
import io
import math
import struct
from typing import IO, Literal, Optional, cast
from zipfile import ZipFile
from zlib import crc32
import zlib
import numpy as np
import numpy.compat

from .npy import encode_npy_header
from .zip import encode_zip, get_zip_offset


ArrayShape = tuple[int, ...]
ArrayOrder = Literal['C', 'F']


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

    end_offset = self._file.tell()

    self._file.seek(self._start_offset)
    self._file.write(encode_npy_header(
      dtype=self._dtype,
      fortran_order=(not not self._fortran_order),
      shape=((*self._shape, self._length) if self._fortran_order else (self._length, *self._shape))
    ))

    self._file.seek(end_offset)
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
    self._file.write(bytes(get_zip_offset(file_name=self._name)))

    self._npy_start_offset = self._file.tell()
    self._file.write(bytes(self._size_compressed))

  def flush(self):
    if self._dtype is None:
      raise ValueError("Unknown dtype")
    if self._shape is None:
      raise ValueError("Unknown shape")

    if self._fortran_order is None:
      self._fortran_order = False

    if self._length < 1:
      self._init()

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
    self._file.write(struct.pack("<BHH", 0b00, npy_header_size, npy_header_size ^ 0xffff))
    self._file.write(npy_header)

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

    if self._length < 1:
      self._init()

    bytes_written = 0

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


def combine_crc32(a_data: bytes, b_crc32: int, b_size: int):
  return crc32(a_data + bytes(b_size)) ^ b_crc32 ^ crc32(bytes(len(a_data) + b_size))

def isfortran(arr: np.ndarray, /):
  return (not arr.flags.c_contiguous) and arr.flags.f_contiguous

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
