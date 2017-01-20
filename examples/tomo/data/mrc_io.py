"""Example for the usage of the MRC reader/writer.

The code below first reads an existing MRC file from disk and prints
some attributes read from the header. If possible, the data is displayed.
Then, a new header is created using a utility function, and a new
(temporary) file is written to disk.

This simulates the start and end of a reconstruction workflow, where
first projection data is read from disk, and finally the reconstructed
volume is written back to disk, using a different header.
"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile

from odl.tomo.data import (
    FileReaderMRC, FileWriterMRC, mrc_header_from_params)


# --- Reading --- #

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'mrc', 'test.mrc')

# File readers can be used as context managers like `open`. As argument,
# either a file stream or a file name string can be used.

with FileReaderMRC(file_path) as reader:
    # Get header and data
    header, data = reader.read()

    # Print some interesting header information conveniently available
    # as reader attributes.
    print('Data shape: ', reader.data_shape)
    print('Data dtype: ', reader.data_dtype)
    print('Data axis ordering: ', reader.data_axis_order)
    print('Header size (bytes): ', reader.header_size)
    print('Additional text labels: ')
    print('')
    for label in reader.labels:
        if label.strip():
            print(repr(label))
    print('')

# Check if the values are correctly reconstructed
values = np.zeros([10, 20, 30], dtype='int16')
values[:, 5:15, 20:25] = 1
print('Values from file all equal to manually created array? ',
      np.array_equal(values, data))

# Plot the data to see that it also looks correct. We take the 5th slice
# along the first axis, which is in the middle.
plt.figure()
plt.title('A standing rectangle, middle right')
plt.imshow(data[5], cmap='Greys_r', interpolation='none')
plt.show()


# --- Writing --- #

# Create some data -- float32 is supported by MRC. Data must be 3D.
new_data = np.ones((50, 100, 200), dtype='float32')

# Create a minimal header. All parameters except these here have default
# values.
header = mrc_header_from_params(new_data.shape, new_data.dtype, kind='volume')

# Write the stuff to a temporary (MRC) file
out_file = tempfile.TemporaryFile()

with FileWriterMRC(out_file, header) as writer:
    # Write both header and data to the file
    writer.write(new_data)

    # Check if the file size is consistent with header and data sizes
    print('File size ({}) = Header size ({}) + Data size ({})'
          ''.format(writer.file.seek(0, 2), writer.header_size,
                    new_data.nbytes))
