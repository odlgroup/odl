# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

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
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import scipy
import tempfile

import odl


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
dtype = np.fromiter(str(image.dtype).ljust(10), dtype='S1')

# Create the header
# Use an OrderedDict for the header to have a predictable order when
# looping through it
header = OrderedDict()
header['shape'] = {'offset': 0, 'value': shape}
header['origin'] = {'offset': 8, 'value': origin}
header['px_size'] = {'offset': 16, 'value': px_size}
header['dtype'] = {'offset': 24, 'value': dtype}

# Initialize the writer with a file and the header. We use a temporary
# file in order to keep the workspace clean.
file = tempfile.NamedTemporaryFile()
writer = odl.tomo.data.FileWriterRawBinaryWithHeader(file, header)

# Write header and data to the file
writer.write(image)

# Print some stuff to see that the sizes are correct
print('File size ({}) = Image size ({}) + Header size ({})'
      ''.format(file.seek(0, 2), image.nbytes, writer.header_bytes))


# --- Reading --- #

# We build a specification for our file format that mirrors `header`.
header_fields = [
    {'name': 'shape', 'offset': 0, 'size': 8, 'dtype': 'int32'},
    {'name': 'origin', 'offset': 8, 'size': 8, 'dtype': 'float32'},
    {'name': 'px_size', 'offset': 16, 'size': 8, 'dtype': 'float32'},
    {'name': 'dtype', 'offset': 24, 'size': 10, 'dtype': 'S1'}
]

# Now we create a reader and read from our newly created file.
# TODO: make this simpler after fixing the properties
reader = odl.tomo.data.FileReaderRawBinaryWithHeader(
    file, header_fields, set_attrs=False)
reader.header_bytes = writer.header_bytes

# Read header and data in one go
header_file, image_file = reader.read()

# Check that everything has been reconstructed correctly
shape_file = header_file['shape']['value']
origin_file = header_file['origin']['value']
px_size_file = header_file['px_size']['value']
dtype_file = header_file['dtype']['value']

print('shape   -- original {}, from file {}'.format(shape, shape_file))
print('origin  -- original {}, from file {}'.format(origin, origin_file))
print('px_size -- original {}, from file {}'.format(px_size, px_size_file))
print('dtype   -- original {}, from file {}'
      ''.format(str(image.dtype), ''.join(dtype_file.astype(str))))

if plt is not None:
    plt.figure()
    plt.title('Original image')
    plt.imshow(image, cmap='Greys_r')
    plt.figure()
    plt.title('Image from file')
    plt.imshow(image, cmap='Greys_r')
    plt.show()
