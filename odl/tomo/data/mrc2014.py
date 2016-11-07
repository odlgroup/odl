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

"""Specification and reader for the MRC2014 file format."""

import numpy as np

from odl.tomo.data.uncompr_bin import (
    FileReaderUncompressedBinary, standardized_fields, spec_fields)


__all__ = ('FileReaderMRC',)


MRC_2014_SPEC = """
+---------+-------+---------+--------+-------------------------------+
|Long word|Byte   |Data type|Name    |Description                    |
+=========+=======+=========+========+===============================+
|1        |1-4    |Int32    |NX      |Number of columns              |
+---------+-------+---------+--------+-------------------------------+
|2        |5-8    |Int32    |NY      |Number of rows                 |
+---------+-------+---------+--------+-------------------------------+
|3        |9-12   |Int32    |NZ      |Number of sections             |
+---------+-------+---------+--------+-------------------------------+
|4        |13-16  |Int32    |MODE    |Data type                      |
+---------+-------+---------+--------+-------------------------------+
|...      |       |         |        |                               |
+---------+-------+---------+--------+-------------------------------+
|8        |29-32  |Int32    |MX      |Number of intervals along X of |
|         |       |         |        |the "unit cell"                |
+---------+-------+---------+--------+-------------------------------+
|9        |33-36  |Int32    |MY      |Number of intervals along Y of |
|         |       |         |        |the "unit cell"                |
+---------+-------+---------+--------+-------------------------------+
|10       |37-40  |Int32    |MZ      |Number of intervals along Z of |
|         |       |         |        |the "unit cell"                |
+---------+-------+---------+--------+-------------------------------+
|11-13    |41-52  |Float32  |CELLA   |Cell dimension in angstroms    |
+---------+-------+---------+--------+-------------------------------+
|...      |       |         |        |                               |
+---------+-------+---------+--------+-------------------------------+
|20       |77-80  |Float32  |DMIN    |Minimum density value          |
+---------+-------+---------+--------+-------------------------------+
|21       |81-84  |Float32  |DMAX    |Maximum density value          |
+---------+-------+---------+--------+-------------------------------+
|22       |85-88  |Float32  |DMEAN   |Mean density value             |
+---------+-------+---------+--------+-------------------------------+
|23       |89-92  |Int32    |ISPG    |Space group number 0, 1, or 401|
+---------+-------+---------+--------+-------------------------------+
|24       |93-96  |Int32    |NSYMBT  |Number of bytes in extended    |
|         |       |         |        |header                         |
+---------+-------+---------+--------+-------------------------------+
|...      |       |         |        |                               |
+---------+-------+---------+--------+-------------------------------+
|27       |105-108|Char     |EXTTYPE |Extended header type           |
+---------+-------+---------+--------+-------------------------------+
|28       |109-112|Int32    |NVERSION|Format version identification  |
|         |       |         |        |number                         |
+---------+-------+---------+--------+-------------------------------+
|...      |       |         |        |                               |
+---------+-------+---------+--------+-------------------------------+
|50-52    |197-208|Float32  |ORIGIN  |Origin in X, Y, Z used in      |
|         |       |         |        |transform                      |
+---------+-------+---------+--------+-------------------------------+
|53       |209-212|Char     |MAP     |Character string 'MAP' to      |
|         |       |         |        |identify file type             |
+---------+-------+---------+--------+-------------------------------+
|54       |213-216|Char     |MACHST  |Machine stamp                  |
+---------+-------+---------+--------+-------------------------------+
|55       |217-220|Float32  |RMS     |RMS deviation of map from mean |
|         |       |         |        |density                        |
+---------+-------+---------+--------+-------------------------------+
"""
# TODO: add nlabl stuff

MRC_HEADER_BYTES = 1024
MRC_SPEC_KEYS = {
    'id': 'Long word',
    'byte_range': 'Byte',
    'dtype': 'Data type',
    'name': 'Name',
    'description': 'Description'}

# Add more if needed
MRC_DTYPE_TO_NPY_DTYPE = {
    'Float32': np.dtype('float32'),
    'Int32': np.dtype('int32'),
    'Char': np.dtype('S1')}

MRC_MODE_TO_NPY_DTYPE = {
    0: np.dtype('uint8'),
    1: np.dtype('int16'),
    2: np.dtype('float32'),
    6: np.dtype('uint16')}

ANGSTROM = 1e-10


class FileReaderMRC(FileReaderUncompressedBinary):

    """Reader for the MRC file format(s).

    By default, the MRC2014 format is used, see ``MRC_2014_SPEC`` for
    details. See also [Che+2015]_ or the `explanations on the CCP4 homepage
    <http://www.ccpem.ac.uk/mrc_format/mrc2014.php>`_ for the
    text of the specification.

    References
    ----------
    [Che+2015] Cheng, A, Henderson, R, Mastronarde, D, Ludtke, S J,
    Schoenmakers, R H M, Short, J, Marabini, R, Dallakyan, S, Agard, D,
    and Winn, M. *MRC2014: Extensions to the MRC format header for electron
    cryo-microscopy and tomography*. Journal of Structural Biology,
    129 (2015), pp 146--150.
    """

    def __init__(self, file, header_fields=None):
        """Initialize a new instance.

        Parameters
        ----------
        file : file-like or str
            Stream or filename from which to read the data. The stream
            is allowed to be already opened in 'rb' mode.
        header_fields : sequence of dicts, optional
            Definition of the fields in the header (per row), each
            containing key-value pairs for the following keys:

            - ``'name'`` : Label for the field.
            - ``'offset_bytes'`` : Start of the field in bytes.
            - ``'size_bytes'`` : Size of the field in bytes.
            - ``'dtype'`` : Data type in Numpy- or Numpy-readable format.
            - ``'dshape'`` (optional) : The array of values is reshaped to
              this shape.
            - ``'description'`` : A human-readable description of the field.

            For the default ``None``, the MRC2014 format is used, see
            ``MRC2014_SPEC``.
        """
        if header_fields is None:
            header_fields = standardized_fields(
                spec_fields(MRC_2014_SPEC, id_key=MRC_SPEC_KEYS['id']),
                keys=MRC_SPEC_KEYS, dtype_map=MRC_DTYPE_TO_NPY_DTYPE)
        super().__init__(file, header_fields)

    def _set_attrs_from_header(self):
        """Set the following attributes of ``self`` from ``self.header``:

            - ``data_shape`` : Shape of the (full) data.
            - ``data_dtype`` : Data type of the data.
            - ``cell_sides`` : Size of the unit cell in meters.
            - ``mrc_version`` : ``(major, minor)`` tuple encoding the version
              of the MRC specification used to create the file.
        """
        # data_shape
        nx = self.header['nx']['value']
        ny = self.header['ny']['value']
        nz = self.header['nz']['value']
        self.data_shape = tuple(np.array([nx, ny, nz], dtype=int).squeeze())

        # data_dtype
        mode = int(self.header['mode']['value'])
        try:
            self.data_dtype = MRC_MODE_TO_NPY_DTYPE[mode]
        except KeyError:
            raise ValueError('data mode {} not supported'.format(mode))

        # cell_sides
        self.cell_sides_angstrom = np.asarray(self.header['cella']['value'],
                                              dtype=float)
        self.cell_sides = self.cell_sides_angstrom * ANGSTROM

        # mrc_version
        nversion = self.header['nversion']['value']
        maj_ver, min_ver = nversion // 10, nversion % 10
        self.mrc_version = (maj_ver, min_ver)

        # header_bytes
        extra_header_bytes = self.header['nsymbt']['value']
        self.header_bytes = MRC_HEADER_BYTES + extra_header_bytes

    # TODO: read extended header for the standard flavors, see the spec
    # homepage
