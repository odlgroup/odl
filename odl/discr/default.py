# Copyright 2014, 2015 The ODL development group
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

# pylint: disable=abstract-method

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from __future__ import unicode_literals
from builtins import super, str
from future import standard_library
standard_library.install_aliases()

from odl.discr.discretization import LinearSpaceDiscretization
from odl.discr.operators import GridCollocation, NearestInterpolation
from odl.space.default import L2


_supported_interp = ('nearest')


class DiscreteL2(LinearSpaceDiscretization):

    """Discretization of an :math:`L^2` space."""

    def __init__(self, l2space, grid, dspace, interp='nearest', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        l2space : `L2`
            The continuous space to be discretized
        dspace : `FnBase`, same `field` as `l2space`
            The space of elements used for data storage
        grid : `TensorGrid`
            The sampling grid for the discretization. Must be contained
            in `l2space.domain`.
        interp : `string`, optional
            The interpolation type to be used for discretization.

            'nearest' : use nearest-neighbor interpolation (default)

            'linear' : use linear interpolation (not implemented)

        kwargs : {'order'}
            'order' : 'C' or 'F'  (Default: 'C')
                The axis ordering in the data storage
        """
        if not isinstance(l2space, L2):
            raise TypeError('{} is not an `L2` type space.'.format(l2space))

        interp = str(interp)
        if interp not in _supported_interp:
            raise ValueError('{} is not among the supported interpolation'
                             'types {}.'.format(interp, _supported_interp))

        order = kwargs.pop('order', 'C')
        restriction = GridCollocation(l2space, grid, dspace, order=order)
        if interp == 'nearest':
            extension = NearestInterpolation(l2space, grid, dspace,
                                             order=order)
        else:
            raise NotImplementedError

        super().__init__(l2space, dspace, restriction, extension)
