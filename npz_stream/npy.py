import struct

from .shared import ArraySpec


MAGIC_PREFIX = b"\x93NUMPY"
MAGIC_LEN = len(MAGIC_PREFIX) + 2
ARRAY_ALIGN = 64
BUFFER_SIZE = 2 ** 18
GROWTH_AXIS_MAX_DIGITS = 21

def encode_npy_header(spec: ArraySpec):
  options = {
    'descr': (spec.dtype.descr if spec.dtype.names is not None else spec.dtype.str),
    'fortran_order': spec.fortran_order,
    'shape': spec.shape
  }

  header = repr(options).encode('utf-8')

  header += b" " * ((GROWTH_AXIS_MAX_DIGITS - len(repr(spec.shape[-1 if spec.fortran_order else 0]))) if len(options['shape']) > 0 else 0)


  hlen = len(header) + 1
  padlen = ARRAY_ALIGN - ((MAGIC_LEN + struct.calcsize("<I") + hlen) % ARRAY_ALIGN)

  header_prefix = MAGIC_PREFIX + b"\x03\x00" + struct.pack("<I", hlen + padlen)
  return header_prefix + header + (b" " * padlen) + b"\n"
