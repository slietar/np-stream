# np-stream

This Python package provides a replacement for the `np.load()`, `np.save()` and `np.savez_compressed()` NumPy functions with support for streaming data on one dimension.

For uncompressed arrays with a known size, use [`np.memmap()`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) instead.


## Installation

```sh
$ pip install np-stream
```


## Usage

### Writing uncompressed data

The dimension which lists entries is added first for C-contiguous arrays (the default when creating arrays) and last otherwise. In this example, the array is C-contiguous, therefore its final shape is `(100, 6, 2)`.

```py
from np_stream import NpyStreamWriter
import numpy as np

with NpyStreamWriter(file) as writer:
  for index in range(100):
    writer.write(np.random.rand(6, 2))
```

Alternatively, the writer can be used without a context manager. The call to `flush()` at the end is mandatory.

```py
writer = NpyStreamWriter(file, shape=entry_shape)

for index in range(100):
  writer.write(np.random.rand(6, 2))

writer.flush()
```

### Writing compressed data

To produced compressed data, just replace `NpyStreamWriter` with `NpzStreamWriter` and create an `.npz` instead of an `.npy` file. The writer uses Deflate compression.

```diff
- from np_stream import NpyStreamWriter
+ from np_stream import NpzStreamWriter

  entry_shape = (6, 2)

- with NpyStreamWriter(file, shape=entry_shape) as writer:
+ with NpzStreamWriter(file, shape=entry_shape) as writer:
    for index in range(100):
      writer.write(np.random.rand(*entry_shape))
```

### Reading uncompressed data

```py
reader = NpyStreamReader(file)

for entry in reader:
  print(entry)
```

### Reading compressed data

```py
reader = NpzStreamReader(file)["arr_0"]

for entry in reader:
  print(entry)
```


## Similar packages

- [npy-append-array](https://github.com/xor2k/npy-append-array)
