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
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import super, str
from future import standard_library
standard_library.install_aliases()

from odl.discr.discretization import LinearSpaceDiscretization
from odl.discr.grid import TensorGrid
from odl.discr.operators import RawGridCollocation, RawNearestInterpolation
from odl.space.cartesian import Ntuples, Rn, Cn
from odl.space.default import L2
from odl.space.function import FunctionSet
from odl.utility.utility import errfmt


class DiscreteL2(LinearSpaceDiscretization):

    """Discretization of an :math:`L^2` space."""

    def __init__(self, l2space, rn_or_cn, interp='nearest'):
        """Initialize a new instance.

        Parameters
        ----------
        l2space : `L2`
            The continuous space to be discretized
        rn_or_cn : `Rn` or `Cn`, same `field` as `l2space`
            The space of elements used for data storage. If `l2space`
            is a complex space, a `Cn` space must be given, and
            analogously a `Rn` space if `l2space` is real.
        interp : `string`, optional  (Default: 'nearest')
            The interpolation type to be used for discretization
        """
        if not isinstance(l2space, L2):
            raise TypeError('{} is not an `L2` type space.'.format(l2space))

        # TODO: replace with a check allowing other implementations of
        # Rn or Cn type spaces
        if not isinstance(rn_or_cn, (Rn, Cn)):
            raise TypeError('{} is not an `Rn` or `Cn` type space.'
                            ''.format(l2space))

        interp = str(interp)
        if interp not in ('nearest', 'linear'):
            raise ValueError('{} is not a supported interpolation type.'
                             ''.format(interp))
