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

"""Tests for the raw binary I/O routines.

Header-related things are tested in the MRC tests, so we only test
the raw binary data I/O here.
"""

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from itertools import permutations
import pytest
import numpy as np
import tempfile

from odl.tomo.data import (
    FileWriterRawBinaryWithHeader, FileReaderRawBinaryWithHeader)
from odl.util.testutils import all_equal, simple_fixture


# --- pytest fixtures --- #


axis_order = simple_fixture(
    name='axis_order',
    params=list(permutations((0, 1, 2))))

shape = simple_fixture(
    name='shape',
    params=[(5, 10, 20), (1, 5, 6), (10, 1, 1), (1, 1, 1)])

order = simple_fixture(name='order', params=['F', 'C'])


# --- Tests --- #


def test_uncompr_bin_io_without_header(shape, floating_dtype, order):
    """Test I/O bypassing the header processing."""
    dtype = np.dtype(floating_dtype)
    file = tempfile.TemporaryFile()

    # data is f(x, y, z) = z
    data = np.ones(shape, dtype=dtype)
    data *= np.arange(shape[2], dtype=dtype)[None, None, :]

    with FileWriterRawBinaryWithHeader(file) as writer:
        writer.write_data(data, reshape_order=order)

    file.seek(0, 2)  # asserts file is still open
    file_size = file.tell()
    assert file_size == data.nbytes

    section_size_bytes = int(np.prod(shape[:2])) * dtype.itemsize
    flat_data = data.ravel(order)

    with FileReaderRawBinaryWithHeader(file, dtype=dtype) as reader:

        # whole file, should work
        file_data = reader.read_data(reshape_order=order)
        assert np.array_equal(file_data, flat_data)

        file_data = reader.read_data(dstart=0, dend=file_size,
                                     reshape_order=order)
        assert np.array_equal(file_data, flat_data)

        # read an arbitrary section ('F' ordering only, otherwise stuff is
        # not contiguous)
        if order == 'F':
            isection = int(np.random.randint(low=0, high=shape[2]))
            flat_section = data[..., isection].ravel()

            section_start = isection * section_size_bytes
            section_end = section_start + section_size_bytes
            file_section = reader.read_data(dstart=section_start,
                                            dend=section_end)
            assert np.array_equal(file_section, flat_section)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
