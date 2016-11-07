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

"""Utilities and class for reading uncompressed binary files with header."""

import csv
import numpy as np
import struct


__all__ = ('FileReaderUncompressedBinary',)


# TODO: add rest
DTYPE_MAP_NPY2STRUCT = {
    np.dtype('float32'): 'f',
    np.dtype('int32'): 'i',
    np.dtype('int16'): 'h',
    np.dtype('int8'): 'b',
    np.dtype('uint64'): 'L',
    np.dtype('uint32'): 'I',
    np.dtype('uint16'): 'H',
    np.dtype('uint8'): 'B',
    np.dtype('S1'): 'b'}


def reformat_spec(spec):
    """Return the reformatted specification table.

    The given specification is assumed to be in
    `reST grid table format
    <http://docutils.sourceforge.net/docs/user/rst/quickref.html#tables>`_.

    Parameters
    ----------
    spec : str
        Specification given as a string containing a definition table.

    Returns
    -------
    lines : tuple of strings
        Table lines with leading and trailing '|' stripped and lines
        containing '...' removed.
    """
    return tuple(line[1:-1].rstrip() for line in spec.splitlines()
                 if line.startswith('|') and '...' not in line)


def spec_fields(spec, id_key):
    """Read the specification and return a list of fields.

    The given specification is assumed to be in
    `reST grid table format
    <http://docutils.sourceforge.net/docs/user/rst/quickref.html#tables>`_.

    Parameters
    ----------
    spec : str
        Specification given as a string containing a definition table.
    id_key : str
        Dictionary key (= column header) for the ID labels in ``spec``.

    Returns
    -------
    fields : tuple of dicts
        Field list of the specification with combined multi-line entries.
        Each field corresponds to one (multi-)line of the spec.
    """
    spec_lines = reformat_spec(spec)

    # Guess the CSV dialect and read the table, producing an iterable
    dialect = csv.Sniffer().sniff(spec_lines[0], delimiters='|')
    reader = csv.DictReader(spec_lines, dialect=dialect)

    # Read the fields as dictionaries and transform keys and values to
    # lowercase.
    fields = []
    for row in reader:
        new_row = {}
        if row[id_key].strip():
            # Start of a new field, indicated by a nontrivial ID entry
            for key, val in row.items():
                new_row[key.strip()] = val.strip()
            fields.append(new_row)
        else:
            # We have the second row of a multi-line field. We
            # append all stripped values to the corresponding existing entry
            # value with an extra space.

            if not fields:
                # Just to make sure that this situation did not happen at
                # the very beginning of the table
                continue

            for key, val in row.items():
                fields[-1][key.strip()] += (' ' + val).rstrip()

    return tuple(fields)


def standardized_fields(field_list, keys, dtype_map):
    """Convert the field keys and values to standard format.

    Data type is piped through `numpy.dtype`, and name is converted
    to lowercase. All keys are converted to lowercase.

    The standardized fields are as follows:

    +--------------+---------+------------------------------------------+
    |Name          |Data type|Description                               |
    +==============+=========+==========================================+
    |'name'        |string   |Name of the element                       |
    +--------------+---------+------------------------------------------+
    |'offset_bytes'|int      |Offset of the current element in bytes    |
    +--------------+---------+------------------------------------------+
    |'size_bytes'  |int      |Size of the current element in bytes      |
    +--------------+---------+------------------------------------------+
    |'dtype'       |type     |Data type of the current element as       |
    |              |         |defined by Numpy                          |
    +--------------+---------+------------------------------------------+
    |'description' |string   |Description of the element (optional)     |
    +--------------+---------+------------------------------------------+
    |'dshape'      |tuple    |For multi-elements: number of elements per|
    |              |         |dimension. Optional for single elements.  |
    +--------------+---------+------------------------------------------+

    Parameters
    ----------
    field_list : sequence of dicts
        Dictionaries describing the field elements, as returned by, e.g.,
        `mrc_spec_fields`.
    keys : dict
        Dictionary with the following entries for the column headers in
        the specification table:

            - ``'id'``
            - ``'byte_range'``
            - ``'dtype'``
            - ``'name'``
            - ``'description'``

    dtype_map : dict
        Mapping from the data type specifiers in the specification table
        to NumPy data types.

    Returns
    -------
    standardized_fields : tuple of dicts
        The standardized fields according to the above table.
    """
    # Parse the fields and represent them in a unfied way
    conv_list = []
    for row, field in enumerate(field_list):
        new_field = {}

        # Name and description: lowercase name, copy description
        new_field['name'] = field[keys['name']].lower()
        new_field['description'] = field[keys['description']]

        # Get offset from ID range
        num_range = field[keys['id']].split('-')
        nstart = int(num_range[0])
        nend = int(num_range[-1])  # 0 for range of type 3-

        # Get byte range and set start
        byte_range = field[keys['byte_range']].split('-')
        byte_start = int(byte_range[0]) - 1
        byte_end = int(byte_range[-1]) - 1
        new_field['offset_bytes'] = byte_start

        # Data type: transform to Numpy format and get shape from its
        # itemsize and the byte range
        dtype = dtype_map[field[keys['dtype']]]
        byte_size = byte_end - byte_start + 1

        if hasattr(dtype, 'itemsize') and byte_size % dtype.itemsize:
            raise ValueError(
                'in row {}: byte range {} not a multiple of itemsize {} '
                'of the data type {}.'
                ''.format(row + 1, field[keys['byte_range']], dtype.itemsize,
                          field[keys['dtype']]))

        new_field['dtype'] = dtype
        new_field['size_bytes'] = byte_size
        # Assuming 1d arrangement of multiple elements
        # TODO: find way to handle 2d fields
        if nend:
            new_field['dshape'] = (nend - nstart + 1,)
        elif hasattr(dtype, 'itemsize'):
            new_field['dshape'] = (byte_size / dtype.itemsize,)
        else:
            new_field['dshape'] = (1,)

        conv_list.append(new_field)

    return tuple(conv_list)


class FileReaderUncompressedBinary(object):

    """Reader for uncompressed binary files including a header."""

    def __init__(self, file, header_fields):
        """Initialize a new instance.

        Parameters
        ----------
        file : file-like or str
            Stream or filename from which to read the data. The stream
            is allowed to be already opened in 'rb' mode.
        header_fields : sequence of dicts
            Definition of the fields in the header (per row), each
            containing key-value pairs for the following keys:

            - ``'name'`` : Label for the field.
            - ``'offset_bytes'`` : Start of the field in bytes.
            - ``'size_bytes'`` : Size of the field in bytes.
            - ``'dtype'`` : Data type in Numpy- or Numpy-readable format.
            - ``'dshape'`` (optional) : The array of values is reshaped to
              this shape.
            - ``'description'`` : A human-readable description of the field.
        """
        try:
            f = open(file, 'rb')
        except TypeError:
            f = file

        if f.mode != 'rb':
            raise ValueError("`file` must be opened in 'rb' mode, but mode "
                             "'is {}'".format(f.mode))

        self.file = f

        # Initialize some attributes to default values, plus the header
        self.data_shape = -1  # Makes reshape a no-op
        self.header_bytes = None
        self.header_fields = header_fields
        self.data = None
        if self.header_fields:
            self.header = self.read_header()
        else:
            self.header = None

    @classmethod
    def from_raw_file(cls, file, header_bytes, dtype):
        """Construct a reader from a raw file w/o header spec.

        Readers constructed with this method can use `read_data`, but
        not `read_header`.

        Parameters
        ----------
        file : file-like or str
            Stream or filename from which to read the data. The stream
            is allowed to be already opened in 'rb' mode.
        header_bytes : int
            Size of the header in bytes.
        dtype :
            Data type specifier for the data field. It must be
            understood by the `numpy.dtype` constructor.

        Returns
        -------
        reader : `MRCFileReader`
            Raw reader for the given MRC file.
        """
        try:
            f = open(file, 'rb')
        except TypeError:
            f = file

        if f.mode != 'rb':
            raise ValueError("`file` must be opened in 'rb' mode, got '{}'"
                             "".format(f.mode))

        header_bytes = int(header_bytes)
        if header_bytes < 0:
            raise ValueError('`header_bytes` must be nonnegative, got {}.'
                             ''.format(header_bytes))

        filesize_bytes = f.seek(0, 2)
        data_bytes = filesize_bytes - header_bytes
        f.seek(0)

        if header_bytes >= filesize_bytes:
            raise ValueError('`header_bytes` is larger or equal to file size '
                             '({} >= {})'.format(header_bytes, filesize_bytes))

        if dtype is None:
            raise TypeError('`dtype` cannot be `None`')
        dtype = np.dtype(dtype)
        data_size = data_bytes / dtype.itemsize

        instance = cls(f, header_fields=[])
        instance.data_dtype = dtype
        instance.data_shape = (data_size,)
        instance.header_bytes = header_bytes
        return instance

    def read_header(self):
        """Read the header from the reader's file.

        The header is also stored in the ``self.header`` attribute.

        Returns
        -------
        header : dict
            Header of ``self.file`` stored in a dictionary, where each
            entry has the following form::

                'name': {'value': value, 'description': description}

            All ``value``'s are `numpy.ndarray`'s with at least one
            dimension. If a ``shape`` is given in ``self.header_fields``,
            the resulting array is reshaped accordingly.
        """
        # Read all fields except data
        header = {}
        for field in self.header_fields:
            # Get all the values from the dictionary
            name = field['name']
            if name == 'data':
                continue
            entry = {'description': field.get('description', '')}
            offset_bytes = field['offset_bytes']
            size_bytes = field.get('size_bytes', None)
            dtype = field['dtype']
            shape = field.get('dshape', None)
            if shape is None:
                shape = -1  # Causes reshape to 1d or no-op

            bytes_per_elem = dtype.itemsize

            if size_bytes is None:
                # Default if 'size_bytes' is omitted
                num_elems = 1
            else:
                num_elems = size_bytes / bytes_per_elem

            if num_elems != int(num_elems):
                raise RuntimeError(
                    "field '{}': `size_bytes` {} and `dtype.itemsize` {} "
                    " result in non-integer number of elements"
                    "".format(name, size_bytes, bytes_per_elem))

            # Create format string for struct module to unpack the binary
            # data
            fmt = str(int(num_elems)) + DTYPE_MAP_NPY2STRUCT[dtype]
            if struct.calcsize(fmt) != size_bytes:
                raise RuntimeError(
                    "field '{}': format '{}' has results in {} bytes, but "
                    "`size_bytes` is {}"
                    "".format(name, fmt, struct.calcsize(fmt), size_bytes))

            self.file.seek(offset_bytes)
            packed_value = self.file.read(size_bytes)
            value = np.array(struct.unpack_from(fmt, packed_value),
                             dtype=dtype)

            if dtype == np.dtype('S1'):
                entry['value'] = ''.join(value.astype(str)).ljust(size_bytes)
            else:
                entry['value'] = value.reshape(shape)
            header[name] = entry

        # Store information gained from the header
        self.header = header
        self._set_attrs_from_header()

        return header

    def _set_attrs_from_header(self):
        """Abstract method for setting attributes from ``self.header``.

        The minimum attributes to set by this method are:

            - ``data_shape`` : Shape of the (full) data.
            - ``data_dtype`` : Data type of the data.

        Subclasses must override this method. They may define additional
        attributes to set.
        """
        raise NotImplementedError

    def read_data(self, dstart=None, dend=None):
        """Read the data from the reader's file and store if desired.

        Parameters
        ----------
        dstart : int, optional
            Offset in bytes of the data field. By default, it is equal
            to ``header_size``. Negative values are added to the file
            size in bytes, to support indexing "backwards".
            Use a value different from ``header_size`` to extract data
            subsets.
        dend : `int, optional`
            End position in bytes until which data is read (exclusive).
            Negative values are added to the file size in bytes, to support
            indexing "backwards". Use a value different from the file size
            to extract data subsets.

        Returns
        -------
        data : `numpy.ndarray`
            The data read from ``self.file``.
        """
        filesize_bytes = self.file.seek(0, 2)
        if dstart is None:
            dstart_abs = int(self.header_bytes)
        elif dstart < 0:
            dstart_abs = filesize_bytes + int(dstart)
        else:
            dstart_abs = int(dstart) + self.header_bytes

        if dend is None:
            dend_abs = int(filesize_bytes)
        elif dend < 0:
            dend_abs = int(dend) + filesize_bytes
        else:
            dend_abs = int(dend) + self.header_bytes

        if dstart_abs >= dend_abs:
            raise ValueError('invalid `dstart` and `dend`, resulting in '
                             'absolute `dstart` >= `dend` ({} >= {})'
                             ''.format(dstart_abs, dend_abs))
        if dstart_abs < self.header_bytes:
            raise ValueError('invalid `dstart`, resulting in absolute '
                             '`dstart` < `header_bytes` ({} < {})'
                             ''.format(dstart_abs, self.header_bytes))
        if dend_abs > filesize_bytes:
            raise ValueError('invalid `dend`, resulting in absolute '
                             '`dend` > `filesize_bytes` ({} < {})'
                             ''.format(dend_abs, filesize_bytes))

        num_elems = (dend_abs - dstart_abs) / self.data_dtype.itemsize
        if num_elems != int(num_elems):
            raise ValueError('trying to read {} bytes, corresponding to '
                             '{} elements of type {}'
                             ''.format(dend_abs - dstart_abs, num_elems,
                                       self.data_dtype))
        self.file.seek(dstart_abs)
        # TODO: use byteorder according to header
        data = np.fromfile(self.file, dtype=self.data_dtype,
                           count=int(num_elems))

        if dstart_abs == self.header_bytes and dend_abs == filesize_bytes:
            # Full dataset read, reshape to stored shape.
            # Use 'F' order in reshaping since it's the native MRC data
            # ordering.
            data = data.reshape(self.data_shape, order='F')

        return data


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
