# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Specification and reader for the MRC2014 file format."""

from __future__ import print_function, division, absolute_import
from builtins import int, object
from collections import OrderedDict
from itertools import permutations
import numpy as np
import struct
import warnings

from odl.contrib.mrc.uncompr_bin import (
    FileReaderRawBinaryWithHeader, FileWriterRawBinaryWithHeader,
    header_fields_from_table)


__all__ = ('FileReaderMRC', 'FileWriterMRC', 'mrc_header_from_params')


# The standard header
MRC_2014_SPEC_TABLE = """
+---------+--------+----------+--------+-------------------------------+
|Long word|Byte    |Data type |Name    |Description                    |
+=========+========+==========+========+===============================+
|1        |1-4     |Int32     |NX      |Number of columns              |
+---------+--------+----------+--------+-------------------------------+
|2        |5-8     |Int32     |NY      |Number of rows                 |
+---------+--------+----------+--------+-------------------------------+
|3        |9-12    |Int32     |NZ      |Number of sections             |
+---------+--------+----------+--------+-------------------------------+
|4        |13-16   |Int32     |MODE    |Data type                      |
+---------+--------+----------+--------+-------------------------------+
|8        |29-32   |Int32     |MX      |Number of intervals along X of |
|         |        |          |        |the "unit cell"                |
+---------+--------+----------+--------+-------------------------------+
|9        |33-36   |Int32     |MY      |Number of intervals along Y of |
|         |        |          |        |the "unit cell"                |
+---------+--------+----------+--------+-------------------------------+
|10       |37-40   |Int32     |MZ      |Number of intervals along Z of |
|         |        |          |        |the "unit cell"                |
+---------+--------+----------+--------+-------------------------------+
|11-13    |41-52   |Float32   |CELLA   |Cell dimension in angstroms    |
|         |        |          |        |(whole volume)                 |
+---------+--------+----------+--------+-------------------------------+
|17       |65-68   |Int32     |MAPC    |axis corresponding to columns  |
|         |        |          |        |(1,2,3 for X,Y,Z)              |
+---------+--------+----------+--------+-------------------------------+
|18       |69-72   |Int32     |MAPR    |axis corresponding to rows     |
|         |        |          |        |(1,2,3 for X,Y,Z)              |
+---------+--------+----------+--------+-------------------------------+
|19       |73-76   |Int32     |MAPS    |axis corresponding to sections |
|         |        |          |        |(1,2,3 for X,Y,Z)              |
+---------+--------+----------+--------+-------------------------------+
|20       |77-80   |Float32   |DMIN    |Minimum density value          |
+---------+--------+----------+--------+-------------------------------+
|21       |81-84   |Float32   |DMAX    |Maximum density value          |
+---------+--------+----------+--------+-------------------------------+
|22       |85-88   |Float32   |DMEAN   |Mean density value             |
+---------+--------+----------+--------+-------------------------------+
|23       |89-92   |Int32     |ISPG    |Space group number 0, 1, or 401|
+---------+--------+----------+--------+-------------------------------+
|24       |93-96   |Int32     |NSYMBT  |Number of bytes in extended    |
|         |        |          |        |header                         |
+---------+--------+----------+--------+-------------------------------+
|27       |105-108 |String    |EXTTYPE |Extended header type           |
+---------+--------+----------+--------+-------------------------------+
|28       |109-112 |Int32     |NVERSION|Format version identification  |
|         |        |          |        |number                         |
+---------+--------+----------+--------+-------------------------------+
|50-52    |197-208 |Int32     |ORIGIN  |Origin in X, Y, Z used in      |
|         |        |          |        |transform                      |
+---------+--------+----------+--------+-------------------------------+
|53       |209-212 |String    |MAP     |Character string 'MAP' to      |
|         |        |          |        |identify file type             |
+---------+--------+----------+--------+-------------------------------+
|54       |213-216 |String    |MACHST  |Machine stamp                  |
+---------+--------+----------+--------+-------------------------------+
|55       |217-220 |Float32   |RMS     |RMS deviation of map from mean |
|         |        |          |        |density                        |
+---------+--------+----------+--------+-------------------------------+
|56       |221-224 |Int32     |NLABL   |Number of labels being used    |
+---------+--------+----------+--------+-------------------------------+
|57-256   |225-1024|String(80)|LABEL   |10 80-character text labels    |
+---------+--------+----------+--------+-------------------------------+
"""
MRC_HEADER_SIZE = 1024

MRC_SPEC_KEYS = {
    'id': 'Long word',
    'byte_range': 'Byte',
    'dtype': 'Data type',
    'name': 'Name',
    'description': 'Description'}

MRC_DTYPE_TO_NPY_DTYPE = {
    'Float32': np.dtype('float32'),
    'Int32': np.dtype('int32'),
    'String': np.dtype('S1')}

MRC_MODE_TO_NPY_DTYPE = {
    0: np.dtype('int8'),
    1: np.dtype('int16'),
    2: np.dtype('float32'),
    6: np.dtype('uint16')}
NPY_DTYPE_TO_MRC_MODE = {v: k for k, v in MRC_MODE_TO_NPY_DTYPE.items()}

ANGSTROM_IN_METERS = 1e-10
MICRON_IN_METERS = 1e-6


def print_mrc2014_spec():
    """Print the MRC2014 specification table.

    The specification table is as follows:
    """
    print(MRC_2014_SPEC_TABLE)


print_mrc2014_spec.__doc__ += MRC_2014_SPEC_TABLE


# Extended header (first section) for the `FEI1` type
MRC_FEI_EXT_HEADER_SECTION = """
+---------+---------+---------+---------------+------------------------------+
|Long word|Byte     |Data type|Name           |Description                   |
+=========+=========+=========+===============+==============================+
|1        |1025-1028|Float32  |A_TILT         |Alpha tilt, in degrees        |
+---------+---------+---------+---------------+------------------------------+
|2        |1029-1032|Float32  |B_TILT         |Beta tilt, in degrees         |
+---------+---------+---------+---------------+------------------------------+
|3        |1033-1036|Float32  |X_STAGE        |Stage x position. Normally in |
|         |         |         |               |SI units (meters), but some   |
|         |         |         |               |older files may be in         |
|         |         |         |               |micrometers.(values larger    |
|         |         |         |               |than 1)                       |
+---------+---------+---------+---------------+------------------------------+
|4        |1037-1040|Float32  |Y_STAGE        |Stage y position              |
+---------+---------+---------+---------------+------------------------------+
|5        |1041-1044|Float32  |Z_STAGE        |Stage z position              |
+---------+---------+---------+---------------+------------------------------+
|6        |1045-1048|Float32  |X_SHIFT        |Stage x shift. For units see  |
|         |         |         |               |remarks on X_STAGE            |
+---------+---------+---------+---------------+------------------------------+
|7        |1049-1052|Float32  |Y_SHIFT        |Stage y shift                 |
+---------+---------+---------+---------------+------------------------------+
|8        |1053-1056|Float32  |DEFOCUS        |Defocus as read from the      |
|         |         |         |               |microscope. For units see     |
|         |         |         |               |remarks on X_STAGE.           |
+---------+---------+---------+---------------+------------------------------+
|9        |1057-1060|Float32  |EXP_TIME       |Exposure time in seconds      |
+---------+---------+---------+---------------+------------------------------+
|10       |1061-1064|Float32  |MEAN_INT       |Mean value of the image       |
+---------+---------+---------+---------------+------------------------------+
|11       |1065-1068|Float32  |TILT_AXIS      |Orientation of the tilt axis  |
|         |         |         |               |in the image in degrees.      |
|         |         |         |               |Vertical to the top is 0      |
|         |         |         |               |degrees, the direction of     |
|         |         |         |               |positive rotation is          |
|         |         |         |               |anti-clockwise.               |
+---------+---------+---------+---------------+------------------------------+
|12       |1069-1072|Float32  |PIXEL_SIZE     |Pixel size of the images in SI|
|         |         |         |               |units (meters)                |
+---------+---------+---------+---------------+------------------------------+
|13       |1073-1076|Float32  |MAGNIFICATION  |Magnification used for        |
|         |         |         |               |recording the images          |
+---------+---------+---------+---------------+------------------------------+
|14       |1077-1080|Float32  |HT             |Value of the high tension in  |
|         |         |         |               |SI units (volts)              |
+---------+---------+---------+---------------+------------------------------+
|15       |1081-1084|Float32  |BINNING        |The binning of the CCD or STEM|
|         |         |         |               |acquisition                   |
+---------+---------+---------+---------------+------------------------------+
|16       |1085-1088|Float32  |APPLIED_DEFOCUS|The intended application      |
|         |         |         |               |defocus in SI units (meters), |
|         |         |         |               |as defined for example in the |
|         |         |         |               |tomography parameters view    |
+---------+---------+---------+---------------+------------------------------+
"""
MRC_FEI_SECTION_SIZE = 128
MRC_FEI_NUM_SECTIONS = 1024


def print_fei_ext_header_spec():
    """Print the specification table of an FEI extended header section.

    The specification table is as follows:
    """
    print(MRC_FEI_EXT_HEADER_SECTION)


print_fei_ext_header_spec.__doc__ += MRC_FEI_EXT_HEADER_SECTION


class MRCHeaderProperties(object):

    """Mixin class adding MRC header-based properties to I/O classes."""

    print_mrc2014_spec = staticmethod(print_mrc2014_spec)
    print_fei_ext_header_spec = staticmethod(print_fei_ext_header_spec)

    @property
    def header_size(self):
        """Total size of `file`'s header (including extended) in bytes.

        The size of the header is determined from `header`. If this is not
        possible (i.e., before the header has been read), 0 is returned.

        If the header contains an ``'nsymbt'`` entry (size of the extra
        header in bytes), its value is added to the regular header size.
        """
        standard_header_size = MRC_HEADER_SIZE

        try:
            extra_header_size = int(self.header['nsymbt']['value'])
        except KeyError:
            extra_header_size = 0

        return standard_header_size + extra_header_size

    @property
    def data_shape(self):
        """Shape tuple of the whole data block as determined from `header`.

        If no header is available (i.e., before it has been initialized),
        or any of the header entries ``'nx', 'ny', 'nz'`` is missing,
        -1 is returned, which makes reshaping a no-op.
        Otherwise, the returned shape is ``(nx, ny, nz)``.

        Note: this is the shape of the data as defined by the header.
        For a non-trivial axis ordering, the shape of actual data will
        be different.

        See Also
        --------
        data_storage_shape
        data_axis_order
        """
        if not self.header:
            return -1
        try:
            nx = self.header['nx']['value']
            ny = self.header['ny']['value']
            nz = self.header['nz']['value']
        except KeyError:
            return -1
        else:
            return tuple(int(n) for n in (nx, ny, nz))

    @property
    def data_storage_shape(self):
        """Shape tuple of the data as stored in the file.

        If no header is available (i.e., before it has been initialized),
        or any of the header entries ``'nx', 'ny', 'nz'`` is missing,
        -1 is returned, which makes reshaping a no-op.
        Otherwise, the returned shape is a permutation of `data_shape`,
        i.e., ``(nx, ny, nz)``, according to `data_axis_order` in the
        following way::

            data_shape[i] == data_storage_shape[data_axis_order[i]]

        See Also
        --------
        data_shape
        data_axis_order
        """
        if self.data_shape == -1:
            return -1
        else:
            return tuple(self.data_shape[ax]
                         for ax in np.argsort(self.data_axis_order))

    @property
    def data_dtype(self):
        """Data type of the data block as determined from `header`.

        If no header is available (i.e., before it has been initialized),
        or the header entry ``'mode'`` is missing, the data type gained
        from the ``dtype`` argument in the initializer is returned.
        Otherwise, it is determined from ``mode``.
        """
        if not self.header:
            return self._init_data_dtype
        try:
            mode = int(self.header['mode']['value'])
        except KeyError:
            return self._init_data_dtype
        else:
            try:
                return MRC_MODE_TO_NPY_DTYPE[mode]
            except KeyError:
                raise ValueError('data mode {} not supported'.format(mode))

    @property
    def data_kind(self):
        """String ``'volume'``, ``'projections'`` or ``'unknown'``.

        The value is determined from the ``'ispg'`` header entry.
        """
        ispg = self.header['ispg']['value']
        if ispg == 0:
            return 'projections'
        elif ispg == 1:
            return 'volume'
        else:
            return 'unknown'

    @property
    def data_axis_order(self):
        """Permutation of ``(0, 1, 2)``.

        The value is determined from the ``'mapc', 'mapr', 'maps'``
        header entries and determines the order of the axes of a
        dataset (`data_shape`) when stored in a file
        (`data_storage_shape`)::

            data_shape[i] == data_storage_shape[data_axis_order[i]]

        For example, if ``data_axis_order == (2, 0, 1)`` then the
        data axis 2 comes first in storage, axis 0 comes second and
        axis 1 comes last.

        If no header is available, (i.e., before it has been initialized),
        or one of the header entries ``'mapc', 'mapr', 'maps'`` is missing,
        the identity permutation ``(0, 1, 2)`` is returned.

        See Also
        --------
        data_shape
        data_storage_shape
        """
        if not self.header:
            return (0, 1, 2)
        try:
            mapc = self.header['mapc']['value']
            mapr = self.header['mapr']['value']
            maps = self.header['maps']['value']
        except KeyError:
            return (0, 1, 2)
        else:
            axis_order = tuple(int(m) - 1 for m in [mapc, mapr, maps])
            if (sorted(axis_order) != [0, 1, 2]):
                # Ignore invalid entries in the header, e.g. 0, 0, 0.
                # Some MRC files out there are like that.
                warnings.warn('invalid axis mapping {}, using (0, 1, 2)'
                              ''.format(tuple(m + 1 for m in axis_order)),
                              RuntimeWarning)
                axis_order = (0, 1, 2)
            return axis_order

    @property
    def cell_sides_angstrom(self):
        """Array of sizes of a unit cell in Angstroms.

        The value is determined from the ``'cella'`` entry in `header`.
        """
        return np.asarray(
            self.header['cella']['value'], dtype=float) / self.data_shape

    @property
    def cell_sides(self):
        """Array of sizes of a unit cell in meters.

        The value is determined from the ``'cella'`` entry in `header`.
        """
        return self.cell_sides_angstrom * ANGSTROM_IN_METERS

    @property
    def mrc_version(self):
        """Version tuple of the MRC file.

        The value is determined from the ``'nversion'`` header entry.
        """
        nversion = int(self.header['nversion']['value'])
        return nversion // 10, nversion % 10

    @property
    def extended_header_size(self):
        """Size of the extended header in bytes.

        The value is determined from the header entry ``'nsymbt'``.
        """
        return int(self.header['nsymbt']['value'])

    @property
    def extended_header_type(self):
        """Type of the extended header.

        The value is determined from the header entry ``'exttype'``.
        See `the specification homepage
        <http://www.ccpem.ac.uk/mrc_format/mrc2014.php>`_ for possible
        values.
        """
        return ''.join(self.header['exttype']['value'].astype(str))

    @property
    def labels(self):
        """Return the 10-tuple of text labels from `header`.

        The value is determined from the header entries ``'nlabl'`` and
        ``'label'``.
        """
        label_array = self.header['label']['value']
        labels = tuple(''.join(row.astype(str)) for row in label_array)

        try:
            nlabels = int(self.header['nlabl']['value'])
        except KeyError:
            nlabels = len(labels)

        # Check if there are nontrivial labels after the number given in
        # the header. If yes, ignore the 'nlabl' information and return
        # all labels.
        if any(label.strip() for label in labels[nlabels:]):
            return labels
        else:
            return labels[:nlabels]


class FileReaderMRC(MRCHeaderProperties, FileReaderRawBinaryWithHeader):

    """Reader for the MRC file format.

    By default, the MRC2014 format is used, see `print_mrc_2014_spec` for
    details. See also [Che+2015] or the `explanations on the CCP4 homepage
    <http://www.ccpem.ac.uk/mrc_format/mrc2014.php>`_ for the
    text of the specification.

    References
    ----------
    [Che+2015] Cheng, A et al. *MRC2014: Extensions to the MRC format header
    for electron cryo-microscopy and tomography*. Journal of Structural
    Biology, 129 (2015), pp 146--150.
    """

    def __init__(self, file, header_fields=None):
        """Initialize a new instance.

        Parameters
        ----------
        file : file-like or str
            Stream or filename from which to read the data. The stream
            is allowed to be already opened in ``'rb'`` mode.
        header_fields : sequence of dicts, optional
            Definition of the fields in the header (per row), each
            containing key-value pairs for the following keys:

            - ``'name'`` : Label for the field.
            - ``'offset'`` : Start of the field in bytes.
            - ``'size'`` : Size of the field in bytes.
            - ``'dtype'`` : Data type in Numpy- or Numpy-readable format.
            - ``'dshape'`` (optional) : The array of values is reshaped to
              this shape.
            - ``'description'`` (optional) : A human-readable description
              of the field.

            For the default ``None``, the MRC2014 format is used, see
            `print_mrc2014_spec`.
        """
        if header_fields is None:
            header_fields = header_fields_from_table(
                spec_table=MRC_2014_SPEC_TABLE,
                keys=MRC_SPEC_KEYS,
                dtype_map=MRC_DTYPE_TO_NPY_DTYPE)

        # `MRCHeaderProperties` has no `__init__`, so this calls
        # `FileReaderRawBinaryWithHeader.__init__`
        super(FileReaderMRC, self).__init__(file, header_fields)

    def read_extended_header(self, groupby='field', force_type=''):
        """Read the extended header according to `extended_header_type`.

        Currently, only the FEI extended header format is supported.
        See `print_fei_ext_header_spec` or `this homepage`_ for the format
        specification.

        The extended header usually has one header section per
        image (slice), in case of the FEI header 128 bytes each, with
        a total of 1024 sections.

        Parameters
        ----------
        groupby : {'field', 'section'}, optional
            How to group the values in the extended header sections.

            ``'field'`` : make an array per section field, e.g.::

                'defocus': [dval1, dval2, ..., dval1024],
                'exp_time': [tval1, tval2, ..., tval1024],
                ...

            ``'section'`` : make a dictionary for each section, e.g.::

                {'defocus': dval1, 'exp_time': tval1},
                {'defocus': dval2, 'exp_time': tval2},
                ...

            If the number of images is smaller than 1024, the last values are
            all set to zero.

        force_type : string, optional
            If given, this value overrides the `extended_header_type`
            from `header`.

            Currently supported: ``'FEI1'``

        Returns
        -------
        ext_header: `OrderedDict` or tuple
            For ``groupby == 'field'``, a dictionary with the field names
            as keys, like in the example.
            For ``groupby == 'section'``, a tuple of dictionaries as
            shown above.
            The returned data structures store no offsets, in contrast
            to the regular header.

        See Also
        --------

        References
        ----------
        .. _this homepage:
           http://www.2dx.unibas.ch/documentation/mrc-software/fei-\
extended-mrc-format-not-used-by-2dx
        """
        ext_header_type = str(force_type).upper() or self.extended_header_type
        if ext_header_type != 'FEI1':
            raise ValueError("extended header type '{}' not supported"
                             "".format(self.extended_header_type))

        groupby, groupby_in = str(groupby).lower(), groupby

        ext_header_len = int(self.header['nsymbt']['value'])
        if ext_header_len % MRC_FEI_SECTION_SIZE:
            raise ValueError('extended header length {} from header is '
                             'not divisible by extended header section size '
                             '{}'.format(ext_header_len, MRC_FEI_SECTION_SIZE))

        num_sections = ext_header_len // MRC_FEI_SECTION_SIZE
        if num_sections != MRC_FEI_NUM_SECTIONS:
            raise ValueError('calculated number of sections ({}) not equal to '
                             'expected number of sections ({})'
                             ''.format(num_sections, MRC_FEI_NUM_SECTIONS))

        section_fields = header_fields_from_table(
            MRC_FEI_EXT_HEADER_SECTION, keys=MRC_SPEC_KEYS,
            dtype_map=MRC_DTYPE_TO_NPY_DTYPE)

        # Make a list for each field and append the values for that
        # field. Then create an array from that list and store it
        # under the field name.
        ext_header = OrderedDict()
        for field in section_fields:
            value_list = []
            field_offset = field['offset']
            field_dtype = field['dtype']
            field_dshape = field['dshape']

            # Compute some parameters
            num_items = int(np.prod(field_dshape))
            size_bytes = num_items * field_dtype.itemsize
            fmt = '{}{}'.format(num_items, field_dtype.char)

            for section in range(num_sections):
                # Get the bytestring from the right position in the file,
                # unpack it and append the value to the list.
                start = section * MRC_FEI_SECTION_SIZE + field_offset
                self.file.seek(start)
                packed_value = self.file.read(size_bytes)
                value_list.append(struct.unpack(fmt, packed_value))

            ext_header[field['name']] = np.array(value_list, dtype=field_dtype)

        if groupby == 'field':
            return ext_header
        elif groupby == 'section':
            # Transpose the data and return as tuple.
            return tuple({key: ext_header[key][i] for key in ext_header}
                         for i in range(num_sections))
        else:
            raise ValueError("`groupby` '{}' not understood"
                             "".format(groupby_in))

    def read_data(self, dstart=None, dend=None, swap_axes=True):
        """Read the data from `file` and return it as Numpy array.

        Parameters
        ----------
        dstart : int, optional
            Offset in bytes of the data field. By default, it is equal
            to ``header_size``. Backwards indexing with negative values
            is also supported.
            Use a value larger than the header size to extract a data subset.
        dend : int, optional
            End position in bytes until which data is read (exclusive).
            Backwards indexing with negative values is also supported.
            Use a value different from the file size to extract a data subset.
        swap_axes : bool, optional
            If ``True``, use `data_axis_order` to swap the axes in the
            returned array. In that case, the shape of the array may no
            longer agree with `data_storage_shape`.

        Returns
        -------
        data : `numpy.ndarray`
            The data read from `file`.
        """
        data = super(FileReaderMRC, self).read_data(dstart, dend)
        data = data.reshape(self.data_shape, order='F')
        if swap_axes:
            data = np.transpose(data, axes=self.data_axis_order)
            assert data.shape == self.data_shape
        return data


class FileWriterMRC(MRCHeaderProperties, FileWriterRawBinaryWithHeader):

    """Writer for the MRC file format.

    See [Che+2015] or the `explanations on the CCP4 homepage
    <http://www.ccpem.ac.uk/mrc_format/mrc2014.php>`_ for the
    text of the specification.

    References
    ----------
    [Che+2015] Cheng, A et al. *MRC2014: Extensions to the MRC format header
    for electron cryo-microscopy and tomography*. Journal of Structural
    Biology, 129 (2015), pp 146--150.
    """

    def write_data(self, data, dstart=None, swap_axes=True):
        """Write ``data`` to `file`.

        Parameters
        ----------
        data : `array-like`
            Data that should be written to `file`.
        dstart : non-negative int, optional
            Offset in bytes of the start position of the written data.
            If provided, reshaping and axis swapping of ``data`` is
            skipped.
            For ``None``, `header_size` is used.
        swap_axes : bool, optional
            If ``True``, use the ``'mapc', 'mapr', 'maps'`` header entries
            to swap the axes in the ``data`` before writing. Use ``False``
            only if the data is already consistent with the final axis
            order.
        """
        if dstart is None:
            shape = self.data_shape
            dstart = int(self.header_size)
        elif dstart < 0:
            raise ValueError('`dstart` must be non-negative, got {}'
                             ''.format(dstart))
        else:
            shape = -1
            dstart = int(dstart)

        if dstart < self.header_size:
            raise ValueError('invalid `dstart`, resulting in absolute '
                             '`dstart` < `header_size` ({} < {})'
                             ''.format(dstart, self.header_size))

        data = np.asarray(data, dtype=self.data_dtype).reshape(shape)
        if swap_axes:
            # Need to argsort here since `data_axis_order` tells
            # "which axis comes from where", which is the inverse of what the
            # `transpose` function needs.
            data = np.transpose(data, axes=np.argsort(self.data_axis_order))
            assert data.shape == self.data_storage_shape

        data = data.reshape(-1, order='F')
        self.file.seek(dstart)
        data.tofile(self.file)


def mrc_header_from_params(shape, dtype, kind, **kwargs):
    """Create a minimal MRC2014 header from the given parameters.

    Parameters
    ----------
    shape : 3-sequence of ints
        3D shape of the stored data. The values are used as
        ``'nx', 'ny', 'nz'`` header entries, in this order. Note that
        this is different from the actual data storage shape for
        non-trivial ``axis_order``.
    dtype : {'int8', 'int16', 'float32', 'uint16'}
        Data type specifier as understood by `numpy.dtype`. It is
        translated to a ``'mode'`` header entry. See `this page
        <http://www.ccpem.ac.uk/mrc_format/mrc2014.php>`_ for valid
        modes.
    kind : {'volume', 'projections'}
        Interpretation of the 3D data, either as single 3D volume or as
        a stack of 2D projections. The value is used for the ``'ispg'``
        header entry.
    extent : 3-sequence of floats, optional
        Size of the 3D volume in meters. The values are used for
        the ``'cella'`` header entry.
        Default: ``shape``, resulting in ``(1, 1, 1)`` unit cells
    axis_order : permutation of ``(0, 1, 2)`` optional
        Order of the data axes as they should appear in the stored file.
        The values are used for the ``'mapc', 'mapr', 'maps'`` header
        entries.
        Default: ``(0, 1, 2)``
    dmin, dmax : float, optional
        Minimum and maximum values of the data, used for header entries
        ``'dmin'`` and ``'dmax'``, resp.
        Default: 1.0, 0.0. These values indicate according to [Che+2015]
        that the values are considered as undetermined.
    dmean, rms : float, optional
        Mean and variance of the data, used for header entries ``'dmean'``
        and ``'rms'``, resp.
        Default: ``min(dmin, dmax) - 1, -1.0``. These values indicate
        according to [Che+2015] that the values are considered as
        undetermined.
    mrc_version : 2-tuple of int, optional
        Version identifier for the MRC file, used for the ``'nversion'``
        header entry.
        Default: ``(2014, 0)``
    text_labels : sequence of strings, optional
        Maximal 10 strings with 80 characters each, used for the
        ``'nlabl'`` and ``'label'`` header entries.
        Default: ``[]``

    Returns
    -------
    header : `OrderedDict`
        Header stored in an ordered dictionary, where each entry has the
        following form::

            'name': {'value': value_as_array,
                     'offset': offset_in_bytes
                     'description': description_string}

        All ``'value'``'s are `numpy.ndarray`'s with at least one
        dimension.

    References
    ----------
    [Che+2015] Cheng, A et al. *MRC2014: Extensions to the MRC format header
    for electron cryo-microscopy and tomography*. Journal of Structural
    Biology, 129 (2015), pp 146--150.
    """
    # Positional args
    shape = [int(n) for n in shape]
    kind, kind_in = str(kind).lower(), kind
    if kind not in ('volume', 'projections'):
        raise ValueError("`kind '{}' not understood".format(kind_in))

    # Keyword args
    extent = kwargs.pop('extent', shape)
    axis_order = kwargs.pop('axis_order', (0, 1, 2))
    if tuple(axis_order) not in permutations((0, 1, 2)):
        raise ValueError('`axis_order` must be a permutation of (0, 1, 2), '
                         'got {}'.format(axis_order))
    dmin = kwargs.pop('dmin', 1.0)
    dmax = kwargs.pop('dmax', 0.0)
    dmean = kwargs.pop('dmean', min(dmin, dmax) - 1.0)
    rms = kwargs.pop('rms', -1.0)
    mrc_version = kwargs.pop('mrc_version', (2014, 0))
    if len(mrc_version) != 2:
        raise ValueError('`mrc_version` must be a sequence of length 2, got '
                         '{}'.format(mrc_version))

    # Text labels: fill each label up with whitespace to 80 characters.
    # Create the remaining labels as 80 * '\x00'
    text_labels_in = kwargs.pop('text_labels', [])
    nlabl = len(text_labels_in)
    if nlabl > 10:
        raise ValueError('expexted maximum of 10 labels, got {} labels'
                         ''.format(nlabl))
    text_labels = [str(label).ljust(80) for label in text_labels_in]
    if any(len(label) > 80 for label in text_labels):
        raise ValueError('labels cannot have more than 80 characters each')

    # Convert to header-friendly form. Names are required to match
    # exactly the header field names, and all of them must exist,
    # so that `eval` below succeeds for all fields.
    nx, ny, nz = [np.array(n, dtype='int32').reshape([1]) for n in shape]
    mode = np.array(NPY_DTYPE_TO_MRC_MODE[np.dtype(dtype)],
                    dtype='int32').reshape([1])
    mx, my, mz = nx, ny, nz
    cella = np.array(extent).reshape([3]).astype('float32')
    mapc, mapr, maps = [np.array(m, dtype='int32').reshape([1]) + 1
                        for m in axis_order]
    dmin, dmax, dmean, rms = [np.array(x, dtype='float32').reshape([1])
                              for x in (dmin, dmax, dmean, rms)]
    ispg = 1 if kind == 'volume' else 0
    ispg = np.array(ispg, dtype='int32', ndmin=1)
    nsymbt = np.array([0], dtype='int32')
    exttype = np.fromstring('    ', dtype='S1')
    nversion = np.array(10 * mrc_version[0] + mrc_version[1],
                        dtype='int32').reshape([1])
    origin = np.zeros(3, dtype='int32')
    map = np.fromstring('MAP ', dtype='S1')
    # TODO: no idea how to properly choose the machine stamp
    machst = np.fromiter(b'DD  ', dtype='S1')
    nlabl = np.array(nlabl, dtype='int32').reshape([1])
    label = np.zeros((10, 80), dtype='S1')  # ensure correct size
    for i, label_i in enumerate(text_labels):
        label[i] = np.fromstring(label_i, dtype='S1')

    # Make the header
    # We use again the specification to set the values
    header_fields = header_fields_from_table(
        MRC_2014_SPEC_TABLE, MRC_SPEC_KEYS, MRC_DTYPE_TO_NPY_DTYPE)

    header = OrderedDict()
    for field in header_fields:
        header[field['name']] = {'offset': field['offset'],
                                 'value': eval(field['name'])}

    return header
