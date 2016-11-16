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

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import int, object, open, str

import csv
import numpy as np
import struct


__all__ = ('FileReaderUncompressedBinary', 'header_fields_from_table')


def _fields_from_table(spec_table, id_key):
    """Read the specification and return a list of fields.

    The given specification is assumed to be in
    `reST grid table format
    <http://docutils.sourceforge.net/docs/user/rst/quickref.html#tables>`_.

    Parameters
    ----------
    spec_table : str
        Specification given as a string containing a definition table.
    id_key : str
        Dictionary key (= column header) for the ID labels in ``spec``.

    Returns
    -------
    fields : tuple of dicts
        Field list of the specification with combined multi-line entries.
        Each field corresponds to one (multi-)line of the spec.
    """
    # Reformat the table, throwing away lines not starting with '|' or
    # containing '...'.
    spec_lines = [line[1:-1].rstrip() for line in spec_table.splitlines()
                  if line.startswith('|') and '...' not in line]

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


def header_fields_from_table(spec_table, keys, dtype_map):
    """Convert the specification table to a standardized format.

    The specification table is assumed to be in
    `reST grid table format
    <http://docutils.sourceforge.net/docs/user/rst/quickref.html#tables>`_.
    It must have the following 5 columns:

        1. ID : an arbitrary unique identifier, e.g., a number.
        2. Byte range : Bytes in the file covered by this field, given
           as number range (e.g. ``15-24``). The byte values start at
           1 (not 0), and the upper value of the range is included.
        3. Data type : Field values are stored in this format. For multiple
           entries, a shape can be specified immediately after the type
           specifier, e.g., ``Float32(4)`` or ``Int32(2,2)``. It is also
           possible to give an incomplete shape, e.g., ``Int32(2)`` with
           a 24-byte field. In this case, the shape is completed to
           ``(3, 2)`` automatically. By default, the one-dimensional shape
           is determined from the data type and the byte range.
        4. Name : The name of the field as used later (in lowercase) for
           identification.
        5. Description : An explanation of the field.

    The table may also contain rows with ``...``, which are ignored.

    The converted specification is a tuple of dictionaries, each
    corresponding to one (multi-)row (=field) of the original table. Each
    field has key-value pairs for the following keys:

    +--------------+---------+------------------------------------------+
    |Name          |Data type|Description                               |
    +==============+=========+==========================================+
    |'name'        |string   |Name of the element                       |
    +--------------+---------+------------------------------------------+
    |'offset'      |int      |Offset of the current element in bytes    |
    +--------------+---------+------------------------------------------+
    |'size'        |int      |Size of the current element in bytes      |
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
    spec_table : str
        Specification given as a string containing a definition table.
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
        The standardized fields according to the above table, one for
        each (multi-)row.
    """
    field_list = _fields_from_table(spec_table, id_key=keys['id'])

    # Parse the fields and represent them in a unfied way
    conv_list = []
    for field in field_list:
        new_field = {}

        # Name and description: lowercase name, copy description
        new_field['name'] = field[keys['name']].lower()
        new_field['description'] = field[keys['description']]

        # Get byte range and set start
        byte_range = field[keys['byte_range']].split('-')
        offset_bytes = int(byte_range[0]) - 1
        end_bytes = int(byte_range[-1]) - 1
        size_bytes = end_bytes - offset_bytes + 1
        new_field['offset'] = offset_bytes
        new_field['size'] = size_bytes

        # Data type: transform to Numpy format and get shape
        dtype_shape = field[keys['dtype']].split('(')
        dtype = dtype_map[dtype_shape[0]]
        new_field['dtype'] = dtype

        if len(dtype_shape) == 2:
            # Shape was given in data type specification

            # Re-attach left parenthesis that was removed in the split
            dshape = np.atleast_1d(eval('(' + dtype_shape[-1]))
            size_bytes_from_shape = np.prod(dshape) * dtype.itemsize
            if size_bytes_from_shape >= size_bytes:
                raise ValueError(
                    "entry '{}': field size {} from shape {} and "
                    "dtype.itemsize {} larger than field size {} from spec"
                    "".format(field[keys['name']], size_bytes_from_shape,
                              dshape, dtype.itemsize, size_bytes))

            # Try to complete the given shape
            if size_bytes % size_bytes_from_shape:
                raise ValueError(
                    "entry '{}': shape {} cannot be completed consistently "
                    "using field size {} and `dtype.itemsize` {}"
                    "".format(field[keys['name']], dshape, size_bytes,
                              dtype.itemsize))

            dshape = (size_bytes // size_bytes_from_shape,) + tuple(dshape)

        else:
            if size_bytes % dtype.itemsize:
                raise ValueError(
                    "entry '{}': field size {} not a multiple of "
                    "`dtype.itemsize` {}"
                    "".format(field[keys['name']], field[keys['byte_range']],
                              dtype.itemsize, field[keys['dtype']]))
            dshape = (size_bytes // dtype.itemsize,)

        new_field['dshape'] = dshape

        conv_list.append(new_field)

    return tuple(conv_list)


class FileReaderUncompressedBinary(object):

    """Reader for uncompressed binary files using an optional header.

    This class can be used to read header and data from files that contain
    a single binary header followed by a single block of uncompressed binary
    data.
    Alternatively, the header can be bypassed and data blocks can be
    read directly using `read_data` with start and end byte values, which
    allows to read arbitrary portions.
    """

    def __init__(self, file, header_fields=(), dtype=None, **kwargs):
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

        dtype : optional
            Data type of the file's data block. It must be understood by
            the `numpy.dtype` constructor. By default, the data type
            is determined from the file header, or, if no information
            is available there, it is set to ``np.dtype(float)``.
        set_attrs : bool, optional
            If ``True``, set attributes of ``self`` from the header for
            convenient access. This can fail for non-standard
            ``header_fields``, in which case ``False`` should be chosen.
            Default: ``True``

        See Also
        --------
        header_fields_from_table :
            Function to parse a specification table, returning a field
            sequence usable as ``header_fields`` parameter.
        """
        try:
            file = open(file, 'rb')
        except TypeError:
            pass

        if file.mode != 'rb':
            raise ValueError("`file` must be opened in 'rb' mode, but mode "
                             "'is {}'".format(file.mode))

        self.__file = file
        self.set_attrs = bool(kwargs.pop('set_attrs', True))

        # Initialize some attributes to default values, plus the header
        self.data_shape = -1  # Makes reshape a no-op
        self.data_dtype = np.dtype(dtype)
        self.header_bytes = 0
        self.__header_fields = header_fields
        if self.header_fields:
            self.__header = self.read_header()
        else:
            self.__header = None

    @property
    def file(self):
        """File object from which ``self`` reads."""
        return self.__file

    @property
    def header(self):
        """Header as read from `file`, or ``None``."""
        return self.__header

    @property
    def header_fields(self):
        """Tuple of dictionaries defining the fields in `header`."""
        return self.__header_fields

    def read_header(self):
        """Read the header from `file`.

        The header is also stored in the `header` attribute.

        Returns
        -------
        header : dict
            Header of ``self.file`` stored in a dictionary, where each
            entry has the following form::

                'name': {'value': value, 'description': description}

            All ``value``'s are `numpy.ndarray`'s with at least one
            dimension. If a ``shape`` is given in `header_fields`,
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
            offset_bytes = int(field['offset'])
            size_bytes = int(field['size'])
            dtype = np.dtype(field['dtype'])
            shape = field.get('dshape', -1)  # no-op by default

            if size_bytes is None:
                # Default if 'size' is omitted
                num_elems = 1
            else:
                num_elems = size_bytes / dtype.itemsize

            if size_bytes % dtype.itemsize:
                raise RuntimeError(
                    "field '{}': `size` {} and `dtype.itemsize` {} "
                    " result in non-integer number of elements"
                    "".format(name, size_bytes, dtype.itemsize))

            # Create format string for struct module to unpack the binary
            # data
            if np.issubdtype(dtype, np.dtype('S')):
                # Have conversion only for 'S1', so we need to translate
                fmt = (str(int(num_elems) * dtype.itemsize) + 's')
            else:
                # Format character can be obtained as dtype.char
                fmt = str(int(num_elems)) + dtype.char

            if struct.calcsize(fmt) != size_bytes:
                raise RuntimeError(
                    "field '{}': format '{}' results in {} bytes, but "
                    "`size` is {}"
                    "".format(name, fmt, struct.calcsize(fmt), size_bytes))

            self.file.seek(offset_bytes)
            packed_value = self.file.read(size_bytes)

            if np.issubdtype(dtype, np.dtype('S')):
                # Bytestring type, decode instead of unpacking. Replace
                # \x00 characters with whitespace so the final length is
                # correct
                packed_value = packed_value.replace(b'\x00', b' ')
                value = np.fromiter(packed_value.decode().ljust(size_bytes),
                                    dtype=dtype)
                entry['value'] = value.astype(str).reshape(shape)
            else:
                value = np.array(struct.unpack_from(fmt, packed_value),
                                 dtype=dtype)
                entry['value'] = value.reshape(shape)
            header[name] = entry

        # Store information gained from the header
        self.__header = header
        if self.set_attrs:
            self._set_attrs_from_header()

        return header

    def _set_attrs_from_header(self):
        """Abstract method for setting attributes from `header`.

        The minimum attributes to set by this method are:

            - ``data_shape`` : Shape of the (full) data.
            - ``data_dtype`` : Data type of the data.

        Subclasses should override this method. They may define additional
        attributes to set.
        """
        raise NotImplementedError('abstract method')

    def read_data(self, dstart=None, dend=None, reshape_order='C'):
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
            Backwards indexing with negative values  is also supported.
            Use a value different from the file size to extract a data subset.
        reshape_order : {'C', 'F', 'A'}, optional
            Value passed as ``order`` parameter to `numpy.reshape`.
            Reshaping is only done in case the whole data block is read.

        Returns
        -------
        data : `numpy.ndarray`
            The data read from `file`.
        """
        filesize_bytes = self.file.seek(0, 2)  # 2 means "from the end"
        if dstart is None:
            dstart_abs = int(self.header_bytes)
        elif dstart < 0:
            dstart_abs = filesize_bytes + int(dstart)
        else:
            dstart_abs = int(dstart)

        if dend is None:
            dend_abs = int(filesize_bytes)
        elif dend < 0:
            dend_abs = int(dend) + filesize_bytes
        else:
            dend_abs = int(dend)

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
        # Numpy determines byte order by itself
        data = np.fromfile(self.file, dtype=self.data_dtype,
                           count=int(num_elems))

        if dstart_abs == self.header_bytes and dend_abs == filesize_bytes:
            # Full dataset read, reshape to stored shape.
            data = data.reshape(self.data_shape, order=reshape_order)

        return data


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
