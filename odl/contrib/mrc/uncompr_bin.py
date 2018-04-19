# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities and class for reading uncompressed binary files with header."""

from __future__ import absolute_import, division, print_function

import csv
import struct
from builtins import int, object
from collections import OrderedDict

import numpy as np

__all__ = ('FileReaderRawBinaryWithHeader',
           'FileWriterRawBinaryWithHeader',
           'header_fields_from_table')


def _fields_from_table(spec_table, id_key):
    """Read a specification and return a list of fields.

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
    # Reformat the table, throwing away lines not starting with '|'
    spec_lines = [line[1:-1].rstrip() for line in spec_table.splitlines()
                  if line.startswith('|')]

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
           The data type must map to a NumPy data type (``dtype_map``).
        4. Name : The name of the field as used later (in lowercase) for
           identification.
        5. Description : An explanation of the field.

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


class FileReaderRawBinaryWithHeader(object):

    """Reader for uncompressed binary files using an optional header.

    This class can be used to read header and data from files that contain
    a single binary header followed by a single block of uncompressed binary
    data.
    Alternatively, the header can be bypassed and data blocks can be
    read directly using `read_data` with start and end byte values, which
    allows to read arbitrary portions.

    An instance of this class can also be used as context manager, i.e.::

        with FileReaderRawBinaryWithHeader(file, header_fields) as reader:
            header, data = reader.read()
    """

    def __init__(self, file, header_fields=(), dtype=None):
        """Initialize a new instance.

        Parameters
        ----------
        file : file-like or str
            Stream or filename from which to read the data. A stream
            must to be open in ``'rb'`` mode.
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

            If empty, the header size is set 0, and the data type is
            assumed to be ``dtype``. Use this in conjunction with
            parametrized `read_data` to bypass the header and read
            arbitrary data portions.

        dtype : optional
            Data type of the file's data block. It must be understood by
            the `numpy.dtype` constructor. By default, the data type
            is determined from the file header, or, if no information
            is available there, it is set to ``np.dtype(float)``.

        See Also
        --------
        header_fields_from_table :
            Function to parse a specification table, returning a field
            sequence usable as ``header_fields`` parameter.
        """
        # Pick out the file object from temp files
        file = getattr(file, 'file', file)
        # Need those attrs in subsequent code
        file_attrs = ('mode', 'seek', 'read', 'readinto', 'close')
        is_file = all(hasattr(file, attr) for attr in file_attrs)
        if is_file:
            self.__file = file
            self.__owns_file = False
        else:
            self.__file = open(file, 'rb', buffering=0)
            self.__owns_file = True

        if 'b' not in self.file.mode:
            raise ValueError("`file` must be opened in binary mode, "
                             "but mode 'is {}'".format(self.file.mode))

        try:
            iter(header_fields)
        except TypeError:
            raise TypeError('`header_fields` must be iterable, got '
                            '{!r}'.format(header_fields))
        self.__header_fields = header_fields

        # Set default values for some attributes
        self._init_data_dtype = np.dtype(dtype)
        self.__header = OrderedDict()

    @property
    def file(self):
        """File object from which ``self`` reads."""
        return self.__file

    def __enter__(self):
        """Initializer for the context manager."""
        return self

    def __exit__(self, *exc):
        """Cleanup before on exiting the context manager."""
        if self.__owns_file:
            self.file.close()

    @property
    def header_size(self):
        """Size of `file`'s header in bytes.

        The size of the header is determined from `header`. If this is not
        possible (i.e., before the header has been read), 0 is returned.
        """
        if not self.header:
            return 0

        # Determine header size by finding the largest offset and the
        # value of the corresponding entry. The header size is the
        # offset plus the size of the entry.
        max_entry = max(self.header.values(),
                        key=lambda val: val['offset'])
        return max_entry['offset'] + max_entry['value'].nbytes

    @property
    def data_storage_shape(self):
        """Shape of the whole data block as stored in `file`.

        This is a default implementation always returning -1, which makes
        reshaping a no-op.
        Subclasses should override this property with an implementation
        that returns the data shape from the header.
        """
        return -1

    @property
    def data_dtype(self):
        """Data type of the data block in `file`.

        This is a default implementation returning the data type gained
        from the ``dtype`` argument in the initializer.
        Subclasses should override this property with an implementation
        that returns the data type from the header.
        """
        return self._init_data_dtype

    @property
    def header(self):
        """Header as read from `file`, or an empty dictionary."""
        return self.__header

    @property
    def header_fields(self):
        """Tuple of dictionaries defining the fields in `header`."""
        return self.__header_fields

    def read(self):
        """Return header and data from `file`.

        Returns
        -------
        header : `OrderedDict`
            The header as read from `file`.
        data : `numpy.ndarray`
            The data block from `file`, reshaped according to
            `data_storage_shape`.

        See Also
        --------
        read_header
        read_data
        """
        return self.read_header(), self.read_data()

    def read_header(self):
        """Read the header from `file`.

        The header is also stored in the `header` attribute.

        Returns
        -------
        header : `OrderedDict`
            Header from `file`, stored in an ordered dictionary, where each
            entry has the following form::

                'name': {'value': value_as_array,
                         'offset': offset_in_bytes
                         'description': description_string}

            All ``'value'``'s are `numpy.ndarray`'s with at least one
            dimension. If a ``'shape'`` is given in `header_fields`,
            the resulting array is reshaped accordingly.

        See Also
        --------
        read_data
        """
        # Read all fields except data. We use an OrderedDict such that
        # the order is the same as in `header_fields`. This makes it simple
        # to write later on in the correct order, based only on `header`.
        header = OrderedDict()
        for field in self.header_fields:
            # Get all the values from the dictionary
            name = field['name']
            if name == 'data':
                continue
            entry = {'description': field.get('description', '')}
            offset_bytes = int(field['offset'])
            entry['offset'] = offset_bytes
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
                entry['value'] = value.reshape(shape)
            else:
                value = np.array(struct.unpack_from(fmt, packed_value),
                                 dtype=dtype)
                entry['value'] = value.reshape(shape)

            header[name] = entry

        # Store information gained from the header
        self.__header = header

        return header

    def read_data(self, dstart=None, dend=None):
        """Read data from `file` and return it as Numpy array.

        Parameters
        ----------
        dstart : int, optional
            Offset in bytes of the data field. By default, it is taken to
            be the header size as determined from reading the header.
            Backwards indexing with negative values is also supported.
            Use a value larger than the header size to extract a data subset.
        dend : int, optional
            End position in bytes until which data is read (exclusive).
            Backwards indexing with negative values is also supported.
            Use a value different from the file size to extract a data subset.

        Returns
        -------
        data : `numpy.ndarray`
            The data read from `file`.

        See Also
        --------
        read_header
        """
        self.file.seek(0, 2)  # 2 means "from the end"
        filesize_bytes = self.file.tell()
        if dstart is None:
            dstart_abs = int(self.header_size)
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
        if dstart_abs < self.header_size:
            raise ValueError('invalid `dstart`, resulting in absolute '
                             '`dstart` < `header_size` ({} < {})'
                             ''.format(dstart_abs, self.header_size))
        if dend_abs > filesize_bytes:
            raise ValueError('invalid `dend`, resulting in absolute '
                             '`dend` > `filesize_bytes` ({} < {})'
                             ''.format(dend_abs, filesize_bytes))

        num_elems = (dend_abs - dstart_abs) / self.data_dtype.itemsize
        if num_elems != int(num_elems):
            raise ValueError(
                'trying to read {} bytes, which is not a multiple of '
                'the itemsize {} of the data type {}'
                ''.format(dend_abs - dstart_abs, self.data_dtype.itemsize,
                          self.data_dtype))
        self.file.seek(dstart_abs)
        array = np.empty(int(num_elems), dtype=self.data_dtype)
        self.file.readinto(array.data)
        return array


class FileWriterRawBinaryWithHeader(object):

    """Writer for uncompressed binary files using an optional header.

    This class can be used to write a single binary header followed by a
    single block of uncompressed binary data to a file.
    Alternatively, the header can be bypassed and data blocks can be
    written directly using `write_data`, which allows to write arbitrary
    portions.
    """

    def __init__(self, file, header=None):
        """Initialize a new instance.

        Parameters
        ----------
        file : file-like or str
            Stream or filename to which to write the data. A stream
            must be open in a writable mode.
        header : `OrderedDict`, optional
            Header in form of an ordered dictionary, where each entry has
            the following form::

                'name': {'value': value_as_array,
                         'offset': offset_in_bytes
                         'description': description_string}

            All ``'value'``'s must be `numpy.ndarray`'s. Their size and the
            ``'offset'`` entry determine the space that the value occupies
            in `file`'s header.

            For ``None``, no header is written.

        Notes
        -----
        **Important:** There is no check if the writes from the provided
        header are consistent, i.e., that they don't overwrite each other.
        Ensuring this is the user's responsibility.

        See Also
        --------
        header_fields_from_table :
            Function to parse a specification table, returning a field
            sequence usable as ``header_fields`` parameter.
        FileReaderRawBinaryWithHeader.read_header
        """
        if header is None:
            header = OrderedDict()

        # Pick out the file object from temp files
        file = getattr(file, 'file', file)
        # Need those attrs in subsequent code
        file_attrs = ('mode', 'seek', 'write', 'close')
        is_file = all(hasattr(file, attr) for attr in file_attrs)
        if is_file:
            self.__file = file
            self.__owns_file = False
        else:
            self.__file = open(file, 'wb', buffering=0)
            self.__owns_file = True

        if 'b' not in self.file.mode:
            raise ValueError("`file` must be opened in binary mode, "
                             "but mode 'is {}'".format(self.file.mode))

        if not isinstance(header, dict):
            raise TypeError('`header` must be a dictionary, got {!r}'
                            ''.format(header))
        self.__header = header

    @property
    def file(self):
        """File object from which ``self`` reads."""
        return self.__file

    def __enter__(self):
        """Initializer for the context manager."""
        return self

    def __exit__(self, *exc):
        """Cleanup before on exiting the context manager."""
        if self.__owns_file:
            self.file.close()

    @property
    def header_size(self):
        """Size of `header` in bytes."""
        if not self.header:
            return 0

        # Determine header size by finding the largest offset and the
        # value of the corresponding entry. The header size is the
        # offset plus the size of the entry.
        max_entry = max(self.header.values(),
                        key=lambda val: val['offset'])
        return max_entry['offset'] + max_entry['value'].nbytes

    @property
    def header(self):
        """Header dictionary to be written to `file`."""
        return self.__header

    def write(self, data):
        """Write `header` and provided ``data`` to `file`.

        Parameters
        ----------
        data : `array-like`
            Data that should be written to `file`.
        """
        self.write_header()
        self.write_data(data)

    def write_header(self):
        """Write `header` to `file`.

        See Also
        --------
        write_data
        """
        for properties in self.header.values():
            value = properties['value']
            offset_bytes = int(properties['offset'])
            self.file.seek(offset_bytes)
            value.tofile(self.file)

    def write_data(self, data, dstart=None, reshape_order='C'):
        """Write ``data`` to `file`.

        Parameters
        ----------
        data : `array-like`
            Data that should be written to `file`.
        dstart : non-negative int, optional
            Offset in bytes of the start position of the written data.
            By default, it is taken to be `header_size`.
        reshape_order : {'C', 'F', 'A'}, optional
            Value passed as ``order`` parameter to `numpy.reshape`.
            Reshaping is only done in case the whole data block is read.

        See Also
        --------
        write_header
        """
        data = np.asarray(data).reshape(-1, order=reshape_order)
        if dstart is None:
            dstart = int(self.header_size)
        elif dstart < 0:
            raise ValueError('`dstart` must be non-negative, got {}'
                             ''.format(dstart))
        else:
            dstart = int(dstart)

        if dstart < self.header_size:
            raise ValueError('invalid `dstart`, resulting in absolute '
                             '`dstart` < `header_size` ({} < {})'
                             ''.format(dstart, self.header_size))

        self.file.seek(dstart)
        data.tofile(self.file)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
