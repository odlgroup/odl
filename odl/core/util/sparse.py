# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

# pylint: disable=line-too-long

"""
Sparse matrix representation for creating product space operators.
"""

import numpy as np

__all__ = ('COOMatrix',)


class COOMatrix():
    """
    Custom coo matrix representation for creating product space operators.

    The columns, rows and data are stored in separate lists such that
    A[i[k], j[k]] = data[k].

    Note that, the class is only used as a container and does not provide
    any matrix operations. Further, no checks are performed on the data thus
    duplicate and out-of-order indices are allowed and the user is responsible
    for ensuring the correct shape of the matrix.

    """

    def __init__(self, data, index, shape):

        # type check
        if len(data) != len(index[0]) or len(data) != len(index[1]):
            raise ValueError('data and index must have the same length')

        self.__data = data
        self.__row_index = np.asarray(index[0])
        self.__col_index = np.asarray(index[1])
        self.__shape = shape

    @property
    def row(self):
        """Return the row indices of the matrix."""
        return self.__row_index

    @property
    def col(self):
        """Return the column indices of the matrix."""
        return self.__col_index

    @property
    def shape(self):
        """Return the shape of the matrix."""
        return self.__shape

    @property
    def data(self):
        """Return the data of the matrix."""
        return self.__data

    def __repr__(self):
        return ( f"COO matrix({self.data},"
                + f"({self.__row_index}, {self.__col_index}), {self.shape})" )
