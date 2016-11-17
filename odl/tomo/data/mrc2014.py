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

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import int, super

from collections import OrderedDict
import numpy as np

from odl.tomo.data.uncompr_bin import (
    FileReaderRawBinaryWithHeader, FileWriterRawBinaryWithHeader,
    header_fields_from_table)


__all__ = ('FileReaderMRC', 'FileWriterMRC', 'mrc_header_from_params')


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
|...      |        |          |        |                               |
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
|...      |        |          |        |                               |
+---------+--------+----------+--------+-------------------------------+
|17       |65-68   |Int32     |MAPC    |axis corresponding to sections |
|         |        |          |        |(1,2,3 for X,Y,Z)              |
+---------+--------+----------+--------+-------------------------------+
|18       |69-72   |Int32     |MAPR    |axis corresponding to sections |
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
|...      |        |          |        |                               |
+---------+--------+----------+--------+-------------------------------+
|27       |105-108 |String    |EXTTYPE |Extended header type           |
+---------+--------+----------+--------+-------------------------------+
|28       |109-112 |Int32     |NVERSION|Format version identification  |
|         |        |          |        |number                         |
+---------+--------+----------+--------+-------------------------------+
|...      |        |          |        |                               |
+---------+--------+----------+--------+-------------------------------+
|50-52    |197-208 |Float32   |ORIGIN  |Origin in X, Y, Z used in      |
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


def print_mrc2014_spec():
    """Print the MRC2014 specification table.

    The specification table is as follows:
    """
    print(MRC_2014_SPEC_TABLE)

print_mrc2014_spec.__doc__ += MRC_2014_SPEC_TABLE


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
    'String': np.dtype('S1')}

MRC_MODE_TO_NPY_DTYPE = {
    0: np.dtype('int8'),
    1: np.dtype('int16'),
    2: np.dtype('float32'),
    6: np.dtype('uint16')}
NPY_DTYPE_TO_MRC_MODE = {v: k for k, v in MRC_MODE_TO_NPY_DTYPE.items()}

ANGSTROM_IN_METERS = 1e-10


class FileReaderMRC(FileReaderRawBinaryWithHeader):

    """Reader for the MRC file format.

    By default, the MRC2014 format is used, see `print_mrc_2014_spec` for
    details. See also [Che+2015]_ or the `explanations on the CCP4 homepage
    <http://www.ccpem.ac.uk/mrc_format/mrc2014.php>`_ for the
    text of the specification.

    References
    ----------
    [Che+2015] Cheng, A et al. *MRC2014: Extensions to the MRC format header
    for electron cryo-microscopy and tomography*. Journal of Structural
    Biology, 129 (2015), pp 146--150.
    """

    def __init__(self, file, header_fields=None, **kwargs):
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
        set_attrs : bool, optional
            If ``True``, set attributes of ``self`` from the header for
            convenient access. This can fail for non-standard
            ``header_fields``, in which case ``False`` should be chosen.
            Default: ``True``
        """
        if header_fields is None:
            header_fields = header_fields_from_table(
                spec_table=MRC_2014_SPEC_TABLE,
                keys=MRC_SPEC_KEYS,
                dtype_map=MRC_DTYPE_TO_NPY_DTYPE)

        super().__init__(file, header_fields, **kwargs)

    print_mrc2014_spec = staticmethod(print_mrc2014_spec)

    @property
    def header_size(self):
        """Size of `file`'s header in bytes.

        The size of the header is determined from `header`. If this is not
        possible (i.e., before the header has been read), 0 is returned.

        If the header contains an ``'nsymbt'`` entry (size of the extra
        header in bytes), its value is added to the regular header size.
        """
        standard_header_size = super().header_size
        if standard_header_size == 0:
            return standard_header_size

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

        If no header is available, (i.e., before it has been initialized),
        or one of the header entries ``'mapc', 'mapr', 'maps'`` is missing,
        the identity permutation ``(1, 2, 3)`` is returned.
        Otherwise, value is determined from the ``'mapc', 'mapr', 'maps'``
        header entries.
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
            return tuple(int(m) - 1 for m in (mapc, mapr, maps))

    @property
    def cell_sides_angstrom(self):
        """Array of sizes of a unit cell in Angstroms.

        The value is determined from the ``'cella'`` entry in `header`.
        """
        return (np.asarray(self.header['cella']['value'], dtype=float) /
                self.data_shape)

    @property
    def cell_sides(self):
        """Array of sizes of a unit cell in meters.

        The value is determined from the ``'cella'`` entry in `header`.
        """
        return self.cell_sides_angstrom * ANGSTROM_IN_METERS

    # TODO: origin

    @property
    def mrc_version(self):
        """Version tuple of the MRC file.

        The value is determined from the ``'nversion'`` header entry.
        """
        nversion = int(self.header['nversion']['value'])
        return nversion // 10, nversion % 10

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
        if any(label for label in labels[nlabels:]):
            return labels
        else:
            return labels[:nlabels]

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
            longer agree with `data_shape`.

        Returns
        -------
        data : `numpy.ndarray`
            The data read from `file`.
        """
        data = super().read_data(dstart, dend, reshape_order='F')
        if swap_axes:
            data = np.transpose(data, axes=self.data_axis_order)
        return data

    # TODO: read extended header for the standard flavors, see the spec
    # homepage


class FileWriterMRC(FileWriterRawBinaryWithHeader):

    """Writer for the MRC file format.

    See [Che+2015]_ or the `explanations on the CCP4 homepage
    <http://www.ccpem.ac.uk/mrc_format/mrc2014.php>`_ for the
    text of the specification.

    References
    ----------
    [Che+2015] Cheng, A et al. *MRC2014: Extensions to the MRC format header
    for electron cryo-microscopy and tomography*. Journal of Structural
    Biology, 129 (2015), pp 146--150.
    """

    print_mrc2014_spec = staticmethod(print_mrc2014_spec)

    @property
    def data_shape(self):
        """Shape tuple of the whole data block as determined from `header`.

        If any of the header entries ``'nx', 'ny', 'nz'`` is missing,
        -1 is returned, which makes reshaping a no-op.
        Otherwise, the returned shape is ``(nx, ny, nz)``.
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
    def data_dtype(self):
        """Data type of the data block as determined from `header`.

        If the header entry ``'mode'`` is missing, ``None`` is returned.
        Otherwise, it is determined from ``mode``.
        """
        if not self.header:
            return None
        try:
            mode = int(self.header['mode']['value'])
        except KeyError:
            return None
        else:
            try:
                return MRC_MODE_TO_NPY_DTYPE[mode]
            except KeyError:
                raise ValueError('data mode {} not supported'.format(mode))

    @property
    def data_axis_order(self):
        """Permutation of ``(0, 1, 2)``.

        If one of the header entries ``'mapc', 'mapr', 'maps'`` is missing,
        the identity permutation ``(1, 2, 3)`` is returned.
        Otherwise, value is determined from the ``'mapc', 'mapr', 'maps'``
        header entries.
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
            return tuple(int(m) - 1 for m in (mapc, mapr, maps))

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
            reshape = True
            dstart = int(self.header_size)
        elif dstart < 0:
            raise ValueError('`dstart` must be non-negative, got {}'
                             ''.format(dstart))
        else:
            reshape = True
            dstart = int(dstart)

        if dstart < self.header_size:
            raise ValueError('invalid `dstart`, resulting in absolute '
                             '`dstart` < `header_size` ({} < {})'
                             ''.format(dstart, self.header_size))

        data = np.asarray(data, dtype=self.data_dtype)
        if reshape:
            shape = tuple(self.data_shape[ax] for ax in self.data_axis_order)
            data = data.reshape(shape)
            if swap_axes:
                data = np.transpose(data, axes=self.data_axis_order)

        data = data.reshape(-1, order='F')
        self.file.seek(dstart)
        data.tofile(self.file)


def mrc_header_from_params(shape, dtype, kind, **kwargs):
    """Create a minimal MRC2014 header from the given parameters.

    Parameters
    ----------
    shape : 3-sequence of ints
        3D shape of the stored data. The values are used as
        ``'nx', 'ny', 'nz'`` header entries, in this order.
    dtype : {'int8', 'int16', 'float32', 'uint16'}
        Data type specifier as understood by `numpy.dtype`. It is
        translated to a ``'mode'`` header entry. See `this page
        <http://www.ccpem.ac.uk/mrc_format/mrc2014.php>`_ for valid
        modes.
    kind : {'volume', 'projections'}
        Interpretation of the 3D data, either as single 3D volume or as
        a stack of 2D projections. The value is used for the ``'ispg'``
        header entry.
    cell_sides : 3-sequence of floats, optional
        Size of the 3D unit cell in meters. The values are used for
        the ``'cella'`` header entry.
        Default: ``(1, 1, 1)``
    axis_order : permutation of ``(0, 1, 2)`` optional
        Order of the data axes as they should appear in the stored file.
        The values are used for the ``'mapc', 'mapr', 'maps'`` header
        entries.
        Default: ``(0, 1, 2)``
    dmin, dmax : float, optional
        Minimum and maximum values of the data, used for header entries
        ``'dmin'`` and ``'dmax'``, resp.
        Default: 1.0, 0.0. These values indicate according to [Che+2015]_
        that the values are considered as undetermined.
    dmean, rms : float, optional
        Mean and variance of the data, used for header entries ``'dmean'``
        and ``'rms'``, resp.
        Default: ``min(dmin, dmax) - 1, -1.0``. These values indicate
        according to [Che+2015]_ that the values are considered as
        undetermined.
    mrc_version : 2-tuple of int, optional
        Version identifier for the MRC file, used for the ``'nversion'``
        header entry.
        Default: ``(2014, 0)``
    text_labels : sequence of strings
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
    kind, kind_in = str(kind).lower(), kind
    if kind not in ('volume', 'projections'):
        raise ValueError("`kind '{}' not understood".format(kind_in))

    # Keyword args
    cell_sides = kwargs.pop('cell_sides', (1.0, 1.0, 1.0))
    axis_order = kwargs.pop('axis_order', (0, 1, 2))
    if (len(axis_order) != 3 or
            any(axis_order.count(i) != 1 for i in (0, 1, 2))):
        raise ValueError('`axis_order` must be a permutation of (0, 1, 2), '
                         'got {}'.format(axis_order))
    dmin = kwargs.pop('dmin', 1.0)
    dmax = kwargs.pop('dmax', 0.0)
    dmean = kwargs.pop('dmean', min(dmin, dmax) - 1.0)
    rms = kwargs.pop('dmean', -1.0)
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
    nx, ny, nz = [np.array(shape[ax], dtype='int32').reshape([1])
                  for ax in axis_order]
    mode = np.array(NPY_DTYPE_TO_MRC_MODE[np.dtype(dtype)],
                    dtype='int32').reshape([1])
    mx, my, mz = nx, ny, nz
    cella = np.array(cell_sides, dtype='float32').reshape([3])
    cella *= [int(n) for n in (nx, ny, nz)]
    mapc, mapr, maps = [np.array(m, dtype='int32').reshape([1]) + 1
                        for m in axis_order]
    dmin, dmax, dmean, rms = [np.array(x, dtype='float32').reshape([1])
                              for x in (dmin, dmax, dmean, rms)]
    ispg = 1 if kind == 'volume' else 0
    ispg = np.array(ispg, dtype='int32', ndmin=1)
    nsymbt = np.array([0], dtype='int32')
    exttype = np.fromiter('    ', dtype='S1')
    nversion = np.array(10 * mrc_version[0] + mrc_version[1],
                        dtype='int32').reshape([1])
    origin = np.zeros(3, dtype='float32')
    map = np.fromiter('MAP ', dtype='S1')
    # TODO: no idea how to properly choose the machine stamp
    machst = np.fromiter(b'DD  ', dtype='S1')
    nlabl = np.array(nlabl, dtype='int32').reshape([1])
    label = np.zeros((10, 80), dtype='S1')  # ensure correct size
    for i, label_i in enumerate(text_labels):
        label[i] = np.fromiter(label_i, dtype='S1')

    # Make the header
    # We use again the specification to set the values
    header_fields = header_fields_from_table(
        MRC_2014_SPEC_TABLE, MRC_SPEC_KEYS, MRC_DTYPE_TO_NPY_DTYPE)

    header = OrderedDict()
    for field in header_fields:
        header[field['name']] = {'offset': field['offset'],
                                 'value': eval(field['name'])}

    return header
