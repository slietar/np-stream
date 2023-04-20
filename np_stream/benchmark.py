from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

import numpy as np

from . import NpzStreamWriter


input = np.load("bf.npz")["arr_0"]

dtype = 'f4'
length = 10
shape = (500, 2000)

uncompressed_files_size = 0
compressed_files_size = 0

with (
  TemporaryDirectory() as uncompressed_files_dir_path,
  TemporaryDirectory() as compressed_file_dir_path,
  TemporaryFile() as compressed_bundle_file
):
  writer = NpzStreamWriter(compressed_bundle_file, compression=dict(level=2))

  for index in range(length):
    print(index)

    # data = np.random.normal(0, 1, shape).astype(dtype)
    # data = np.zeros(shape, dtype)
    data: np.ndarray = input[index, :, :]

    uncompressed_entry_path = Path(uncompressed_files_dir_path) / f"arr{index}.npy"
    compressed_entry_path = Path(compressed_file_dir_path) / f"arr{index}.npz"

    np.save(uncompressed_entry_path.open("wb"), data)
    np.savez_compressed(compressed_entry_path.open("wb"), data)
    writer.write(data)

    uncompressed_files_size += uncompressed_entry_path.stat().st_size
    compressed_files_size += compressed_entry_path.stat().st_size

  writer.flush()
  compressed_bundle_size = compressed_bundle_file.tell()

print(
  uncompressed_files_size,
  compressed_files_size,
  compressed_bundle_size
)
