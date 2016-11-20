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

"""Tests for the MRC I/O routines."""

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from itertools import permutations
import pytest
import numpy as np
import tempfile

from odl.tomo.data import mrc_header_from_params, FileWriterMRC, FileReaderMRC
from odl.util.testutils import all_equal, simple_fixture


# --- pytest fixtures --- #


mode_dtype_params = [(0, 'int8'), (1, 'int16'), (2, 'float32'), (6, 'uint16')]
mode_dtype_ids = [" mode = {p[0]}, dtype = '{p[1]}' ".format(p=p)
                  for p in mode_dtype_params]


@pytest.fixture(scope='module', ids=mode_dtype_ids, params=mode_dtype_params)
def mrc_mode_dtype(request):
    mode, dtype = request.param
    return mode, np.dtype(dtype)


axis_order = simple_fixture(
    name='axis_order',
    params=list(permutations((0, 1, 2))))

ncells = simple_fixture(
    name='ncells',
    params=[(5, 10, 20), (1, 5, 6), (10, 1, 1), (1, 1, 1)])


ispg_kind_params = [(0, 'projections'), (1, 'volume')]
ispg_kind_ids = [" ispg = {p[0]}, kind = '{p[0]}' ".format(p=p)
                 for p in ispg_kind_params]


@pytest.fixture(scope='module', ids=ispg_kind_ids, params=ispg_kind_params)
def ispg_kind(request):
    return request.param


def test_mrc_header_from_params_defaults(ncells, mrc_mode_dtype, ispg_kind):
    """Test the utility function for the minimal required parameters."""
    mode, dtype = mrc_mode_dtype
    true_ispg, kind = ispg_kind
    header = mrc_header_from_params(ncells, dtype, kind)

    # Check values of all header entries
    nx = header['nx']['value']
    ny = header['ny']['value']
    nz = header['nz']['value']
    assert all_equal([nx, ny, nz], ncells)
    assert header['mode']['value'] == mode
    mx = header['mx']['value']
    my = header['my']['value']
    mz = header['mz']['value']
    assert all_equal([mx, my, mz], ncells)
    cella = header['cella']['value']
    assert all_equal(cella, np.ones(3) * ncells)
    mapc = header['mapc']['value']
    mapr = header['mapr']['value']
    maps = header['maps']['value']
    assert all_equal([mapc, mapr, maps], [1, 2, 3])
    dmin = header['dmin']['value']
    dmax = header['dmax']['value']
    dmean = header['dmean']['value']
    rms = header['rms']['value']
    assert all_equal([dmin, dmax, dmean, rms], [1.0, 0.0, -1.0, -1.0])
    ispg = header['ispg']['value']
    assert ispg == true_ispg
    nsymbt = header['nsymbt']['value']
    assert nsymbt == 0
    exttype = header['exttype']['value']
    assert np.array_equal(exttype, np.fromiter('    ', dtype='S1'))
    nversion = header['nversion']['value']
    assert nversion == 20140
    origin = header['origin']['value']
    assert np.array_equal(origin, np.zeros(3, dtype=int))
    map = header['map']['value']
    assert np.array_equal(map, np.fromiter('MAP ', dtype='S1'))
    machst = header['machst']['value']
    assert np.array_equal(machst, np.fromiter(b'DD  ', dtype='S1'))
    nlabl = header['nlabl']['value']
    assert nlabl == 0
    label = header['label']['value']
    assert np.array_equal(label, np.zeros([10, 80], dtype='S1'))

    # Check all data types
    int32_vars = [nx, ny, nz, mx, my, mz, mapc, mapr, maps, ispg, nsymbt,
                  nversion, origin, nlabl]
    for v in int32_vars:
        assert v.dtype == np.dtype('int32')

    float32_vars = [cella, dmin, dmax, dmean, rms]
    for v in float32_vars:
        assert v.dtype == np.dtype('float32')

    string_vars = [exttype, map, machst, label]
    for v in string_vars:
        assert v.dtype == np.dtype('S1')


def test_mrc_header_from_params_kwargs():
    """Test the utility function for the minimal required parameters."""
    ncells = (10, 20, 30)
    dtype = np.dtype('int8')
    kind = 'projections'
    kwargs = {}
    kwargs['extent'] = [10.0, 1.5, 0.1]
    kwargs['axis_order'] = (2, 0, 1)
    kwargs['dmin'] = 1.0
    kwargs['dmax'] = 5.0
    kwargs['dmean'] = 2.0
    kwargs['rms'] = 0.5
    kwargs['mrc_version'] = (2014, 1)
    kwargs['text_labels'] = ['label 1', '   label 2   ']
    header = mrc_header_from_params(ncells, dtype, kind, **kwargs)

    # Check values of the header entries set by kwargs
    cella = header['cella']['value']
    assert np.allclose(cella, np.array([10.0, 1.5, 0.1]))
    mapc = header['mapc']['value']
    mapr = header['mapr']['value']
    maps = header['maps']['value']
    assert all_equal([mapc, mapr, maps], (3, 1, 2))
    dmin = header['dmin']['value']
    dmax = header['dmax']['value']
    dmean = header['dmean']['value']
    rms = header['rms']['value']
    assert all_equal([dmin, dmax, dmean, rms], [1.0, 5.0, 2.0, 0.5])
    nversion = header['nversion']['value']
    assert nversion == 20141
    nlabl = header['nlabl']['value']
    assert nlabl == 2
    label = header['label']['value']
    true_label = np.zeros([10, 80], dtype='S1')
    true_label[0] = np.fromiter('label 1'.ljust(80), dtype='S1')
    true_label[1] = np.fromiter('   label 2   '.ljust(80), dtype='S1')
    assert np.array_equal(label, true_label)

    # Check all data types
    int32_vars = [mapc, mapr, maps, nversion, nlabl]
    for v in int32_vars:
        assert v.dtype == np.dtype('int32')

    float32_vars = [cella, dmin, dmax, dmean, rms]
    for v in float32_vars:
        assert v.dtype == np.dtype('float32')

    string_vars = [label]
    for v in string_vars:
        assert v.dtype == np.dtype('S1')


def test_mrc_io(ncells, mrc_mode_dtype, ispg_kind, axis_order):
    """Test reading and writing MRC files and the class properties."""
    _, dtype = mrc_mode_dtype
    _, kind = ispg_kind

    # Data shape is the inverse permutation of nx, ny, nz
    data_shape = tuple(ncells[ax] for ax in np.argsort(axis_order))
    header = mrc_header_from_params(ncells, dtype, kind,
                                    axis_order=axis_order)

    # Test writer properties using standard class construction
    file = tempfile.TemporaryFile()
    writer = FileWriterMRC(file, header)

    assert writer.header_size == 1024  # Standard MRC header size
    assert all_equal(writer.num_cells, ncells)
    assert all_equal(writer.data_shape, data_shape)
    assert writer.data_dtype == dtype
    assert all_equal(writer.data_axis_order, axis_order)

    # Test writing some data (that all data types can represent).
    data = np.random.randint(0, 10, size=data_shape).astype(dtype)
    writer.write_data(data)

    # Check file size
    writer.file.seek(0, 2)
    file_size = writer.file.tell()
    assert file_size == writer.header_size + data.nbytes

    # Check flat arrays
    file.seek(1024)
    raw_data = np.fromfile(file, dtype=dtype)
    flat_data = np.transpose(data, axes=axis_order).reshape(-1, order='F')
    assert np.array_equal(raw_data, flat_data)

    # Write everything using the context manager syntax
    with FileWriterMRC(file, header) as writer:
        writer.write(data)

    # Read from the same file using the reader with standard constructor
    reader = FileReaderMRC(file)
    reader.read_header()
    assert writer.header_size == 1024  # Standard MRC header size
    assert all_equal(reader.num_cells, ncells)
    assert all_equal(reader.data_shape, data_shape)
    assert reader.data_dtype == dtype
    assert reader.data_kind == kind
    assert all_equal(reader.data_axis_order, axis_order)
    assert np.allclose(reader.cell_sides_angstrom, 1.0)
    assert all_equal(reader.mrc_version, (2014, 0))
    assert reader.extended_header_type == '    '
    assert reader.labels == ()


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
