from io import BufferedIOBase, BytesIO, RawIOBase
import io
from struct import Struct
import struct
from typing import Literal, Optional

local_file_header_struct = Struct("<IHHHHHIIIHH")


def get_zip_offset(*, file_name: str):
  file_name_encoded = file_name.encode('utf-8')
  return local_file_header_struct.size + len(file_name_encoded)

def encode_zip(
  file: BufferedIOBase | RawIOBase,
  *,
  compressed_size: int,
  compression: Optional[Literal['deflate', 'lzma']] = None,
  crc32: int,
  file_name: str,
  start_offset: int,
  uncompressed_size: int
):
  compression_method = { None: 0, 'deflate': 8, 'lzma': 14 }[compression]
  file_name_encoded = file_name.encode('utf-8')

  local_file_header = local_file_header_struct.pack(
    0x04034b50, # local file header signature
    20, # version needed to extract
    0, # general purpose bit flag
    compression_method, # compression method
    0, # last mod file time
    0, # last mod file date
    (crc32 & 0xffffffff), # crc-32
    compressed_size, # compressed size
    uncompressed_size, # uncompressed size
    len(file_name_encoded), # file name length
    0 # extra field length
  ) + file_name_encoded

  central_directory_file_header = struct.pack(
    "<IHHHHHHIIIHHHHHII",
    0x02014b50, # central directory file header signature
    20, # version made by
    20, # version needed to extract
    0, # general purpose bit flag
    compression_method, # compression method (0 = none)
    0, # last mod file time
    0, # last mod file date
    (crc32 & 0xffffffff), # crc-32
    compressed_size, # compressed size
    uncompressed_size, # uncompressed size
    len(file_name_encoded), # file name length
    0, # extra field length
    0, # file comment length
    0, # disk number where file starts
    0, # internal file attributes
    0, # external file attributes
    0 # relative offset of local file header
  ) + file_name_encoded

  end_of_central_directory_record = struct.pack(
    "<IHHHHIIH",
    0x06054b50, # end of central directory record signature
    0, # number of this disk
    0, # number of the disk with the start of the central directory
    1, # total number of entries in the central directory on this disk
    1, # total number of entries in the central directory
    len(central_directory_file_header), # size of the central directory
    len(local_file_header) + compressed_size, # offset of start of central directory with respect to the starting disk number
    0 # comment length
  )

  # print(file.tell(), len(local_file_header) + compressed_size)

  file.write(central_directory_file_header)
  file.write(end_of_central_directory_record)

  file.seek(start_offset)
  file.write(local_file_header)
