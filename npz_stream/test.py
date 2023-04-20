from io import BytesIO
from tempfile import TemporaryFile
from unittest import TestCase
import numpy as np
import unittest

from . import NpyStreamReader, NpyStreamWriter, NpzStreamReader, NpzStreamWriter


class NpyWriterTest(TestCase):
  def test_default(self):
    with TemporaryFile() as file:
      arr = np.random.rand(1000, 3, 6)

      with NpyStreamWriter(file) as writer:
        for index in range(arr.shape[0]):
          writer.write(arr[index, :, :])

      file.seek(0)
      self.assertTrue(np.array_equal(np.load(file), arr))

  def test_bytesio(self):
    arr = np.random.rand(1000, 3, 6)
    file = BytesIO()

    with NpyStreamWriter(file) as writer:
      for index in range(arr.shape[0]):
        writer.write(arr[index, :, :])

    self.assertEqual(
      file.tell(),
      len(file.getbuffer())
    )

    file.seek(0)
    self.assertTrue(np.array_equal(np.load(file), arr))

  def test_accept_fortran_order(self):
    with TemporaryFile() as file:
      arr = np.random.rand(1000, 3, 6)

      with NpyStreamWriter(file, order='C') as writer:
        for index in range(arr.shape[0]):
          writer.write(np.asfortranarray(arr[index, :, :]))

      file.seek(0)
      loaded: np.ndarray = np.load(file)

      self.assertTrue(np.array_equal(loaded, arr))
      self.assertTrue(loaded.flags.c_contiguous)

  def test_produce_fortran_order(self):
    with TemporaryFile() as file:
      arr = np.random.rand(1000, 3, 6)

      with NpyStreamWriter(file, order='F') as writer:
        for index in range(arr.shape[0]):
          writer.write(arr[index, :, :])

      file.seek(0)
      loaded: np.ndarray = np.load(file)

      self.assertTrue(np.array_equal(loaded, np.moveaxis(arr, 0, -1)))
      self.assertTrue(loaded.flags.f_contiguous)

  def test_cast(self):
    with TemporaryFile() as file:
      shape = (3, 4)

      with NpyStreamWriter(file, dtype='f4') as writer:
        writer.write(np.zeros(shape, dtype='f2'))
        writer.write(np.zeros(shape, dtype='f4'))
        writer.write(np.zeros(shape, dtype='u2'))

        with self.assertRaises(TypeError):
          writer.write(np.zeros(shape, dtype='f8'))

        with self.assertRaises(TypeError):
          writer.write(np.zeros(shape, dtype='u4'))

      file.seek(0)
      np.load(file)


  def test_no_entries(self):
    shape = (3, 6)

    with TemporaryFile() as file:
      with NpyStreamWriter(file, dtype='f8', shape=shape):
        pass

      file.seek(0)
      loaded: np.ndarray = np.load(file)

      self.assertEqual(loaded.shape, (0, *shape))
      self.assertEqual(loaded.dtype, np.dtype('f8'))

  def test_empty_axis(self):
    length = 10
    shape = (0, 6)

    with TemporaryFile() as file:
      with NpyStreamWriter(file) as writer:
        for _ in range(length):
          writer.write(np.zeros(shape))

      file.seek(0)
      loaded: np.ndarray = np.load(file)

      self.assertEqual(loaded.shape, (length, *shape))

  def test_empty_shape(self):
    with TemporaryFile() as file:
      arr = np.random.rand(1000)

      with NpyStreamWriter(file) as writer:
        for index in range(arr.shape[0]):
          writer.write(arr[index])

      file.seek(0)
      loaded: np.ndarray = np.load(file)

      self.assertTrue(np.array_equal(loaded, arr))


class NpyReaderTest(TestCase):
  def test_default(self):
    with TemporaryFile() as file:
      length = 1000
      arr = np.random.rand(length, 3, 4)

      np.save(file, arr)
      file.seek(0)

      reader = NpyStreamReader(file)
      count = 0

      for index, entry in enumerate(reader):
        self.assertTrue(np.array_equal(arr[index, :, :], entry))
        count += 1

      for index in range(length):
        self.assertTrue(np.array_equal(arr[index, :, :], reader[index]))

      self.assertEqual(count, length)
      self.assertEqual(len(reader), length)

  def test_fortran_order(self):
    with TemporaryFile() as file:
      length = 1000
      arr = np.asfortranarray(np.random.rand(3, 4, length))

      np.save(file, arr)
      file.seek(0)

      reader = NpyStreamReader(file)
      count = 0

      for index, entry in enumerate(reader):
        self.assertTrue(np.array_equal(arr[:, :, index], entry))
        count += 1

      self.assertEqual(count, length)


class NpzWriterTest(TestCase):
  def test_default(self):
    with TemporaryFile() as file:
      arr = np.random.rand(1000, 3, 6)

      with NpzStreamWriter(file) as writer:
        for index in range(arr.shape[0]):
          writer.write(arr[index, :, :])

      file.seek(0)
      self.assertTrue(np.array_equal(np.load(file)['arr_0'], arr))

  def test_bytesio(self):
    arr = np.random.rand(1000, 3, 6)
    file = BytesIO()
    bytes_written = 0

    writer = NpzStreamWriter(file)

    for index in range(arr.shape[0]):
      bytes_written += writer.write(arr[index, :, :])

    bytes_written += writer.flush()
    file_size = len(file.getbuffer())

    # Using >= as certain bytes could have been written twice.
    self.assertGreaterEqual(bytes_written, file_size)
    self.assertEqual(file.tell(), file_size)

    file.seek(0)
    self.assertTrue(np.array_equal(np.load(file)['arr_0'], arr))

  def test_no_entries(self):
    shape = (3, 6)

    with TemporaryFile() as file:
      with NpzStreamWriter(file, dtype='f8', shape=shape):
        pass

      file.seek(0)
      loaded: np.ndarray = np.load(file)['arr_0']

      self.assertEqual(loaded.shape, (0, *shape))
      self.assertEqual(loaded.dtype, np.dtype('f8'))

  def test_empty_axis(self):
    length = 10
    shape = (0, 6)

    with TemporaryFile() as file:
      with NpzStreamWriter(file) as writer:
        for _ in range(length):
          writer.write(np.zeros(shape))

      file.seek(0)
      loaded: np.ndarray = np.load(file)['arr_0']

      self.assertEqual(loaded.shape, (length, *shape))

  def test_empty_shape(self):
    with TemporaryFile() as file:
      arr = np.random.rand(1000)

      with NpzStreamWriter(file) as writer:
        for index in range(arr.shape[0]):
          writer.write(arr[index])

      file.seek(0)
      loaded: np.ndarray = np.load(file)['arr_0']

      self.assertTrue(np.array_equal(loaded, arr))


class NpzReaderTest(TestCase):
  def test_default(self):
    with TemporaryFile() as file:
      length = 1000
      arr = np.random.rand(length, 3, 4)

      np.savez_compressed(file, arr)
      file.seek(0)

      reader = NpzStreamReader(file)['arr_0']
      count = 0

      for index, entry in enumerate(reader):
        self.assertTrue(np.array_equal(arr[index, :, :], entry))
        count += 1

      for index in range(length):
        self.assertTrue(np.array_equal(arr[index, :, :], reader[index]))

      self.assertEqual(count, length)
      self.assertEqual(len(reader), length)


if __name__ == '__main__':
  unittest.main()
