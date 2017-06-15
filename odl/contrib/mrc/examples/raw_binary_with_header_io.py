"""Example for the usage of raw binary reader/writer with header.

The code below defines a simple file format for 2D images using
    - shape
    - origin
    - pixel size

First, a file is written using `FileWriterRawBinaryWithHeader`. This
requires a header in a certain format.
Then, the same file is read again using a file specification and the
`FileReaderRawBinaryWithHeader`. The specification is given as a
sequence of dictionaries with a certain structure.
"""

from __future__ import print_function

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tempfile

from odl.contrib.mrc import (
    FileReaderRawBinaryWithHeader, FileWriterRawBinaryWithHeader)


# --- Writing --- #

# Create some test data. We arbitrarily define origin and pixel size.
# In practice, these could come from a `DiscreteLp` space as `mid_pt`
# and `cell_sides` properties.
image = scipy.misc.ascent()
shape = np.array(image.shape, dtype='int32')
origin = np.array([-1.0, 0.0], dtype='float32')
px_size = np.array([0.1, 0.1], dtype='float32')
# To make it storable as binary data, we take the string version of the data
# type with a fixed size of 10 characters and encode it as array of single
# bytes.
dtype = np.fromstring(str(image.dtype).ljust(10), dtype='S1')

# Create the header
# Use an OrderedDict for the header to have a predictable order when
# looping through it
header = OrderedDict()
header['shape'] = {'offset': 0, 'value': shape}
header['origin'] = {'offset': 8, 'value': origin}
header['px_size'] = {'offset': 16, 'value': px_size}
header['dtype'] = {'offset': 24, 'value': dtype}

# Use a temporary file for the output to keep the workspace clean.
# The writer can be used as a context manager like `open`:
tmp_file = tempfile.NamedTemporaryFile()

with FileWriterRawBinaryWithHeader(tmp_file, header) as writer:
    # Write header and data to the file
    writer.write(image)

    # Print some stuff to see that the sizes are correct
    tmp_file.seek(0, 2)  # last position
    file_size = tmp_file.tell()
    print('File size ({}) = Image size ({}) + Header size ({})'
          ''.format(file_size, image.nbytes, writer.header_size))


# --- Reading --- #

# We build a specification for our file format that mirrors `header`.
header_fields = [
    {'name': 'shape', 'offset': 0, 'size': 8, 'dtype': 'int32'},
    {'name': 'origin', 'offset': 8, 'size': 8, 'dtype': 'float32'},
    {'name': 'px_size', 'offset': 16, 'size': 8, 'dtype': 'float32'},
    {'name': 'dtype', 'offset': 24, 'size': 10, 'dtype': 'S1'}
]

# Read header and data from the newly created file. Again, we can use
# the reader in form of a context manager:
with FileReaderRawBinaryWithHeader(tmp_file, header_fields) as reader:
    header_file, image_file = reader.read()  # Read header and data in one go

# Check that everything has been restored correctly
shape_file = header_file['shape']['value']
origin_file = header_file['origin']['value']
px_size_file = header_file['px_size']['value']
dtype_file = header_file['dtype']['value']

print('shape   -- original {}, from file {}'.format(shape, shape_file))
print('origin  -- original {}, from file {}'.format(origin, origin_file))
print('px_size -- original {}, from file {}'.format(px_size, px_size_file))
print('dtype   -- original {}, from file {}'
      ''.format(str(image.dtype), ''.join(dtype_file.astype(str))))

# Compare the plots of the original and recovered images
plt.figure()
plt.title('Original image')
plt.imshow(image, cmap='Greys_r')
plt.figure()
plt.title('Image from file')
plt.imshow(image, cmap='Greys_r')
plt.show()
