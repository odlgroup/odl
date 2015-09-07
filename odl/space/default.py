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

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from __future__ import unicode_literals
from builtins import super
from future import standard_library
standard_library.install_aliases()

# ODL imports
from odl.discr.grid import TensorGrid
from odl.space.cartesian import Rn, Cn
from odl.space.function import FunctionSpace
from odl.space.set import RealNumbers, ComplexNumbers
try:
    from odl.space.cuda import CudaRn
    CUDA_AVAILABLE = True
except ImportError:
    CudaRn = None
    CUDA_AVAILABLE = False


class L2(FunctionSpace):
    """The space of square integrable functions on some domain."""

    def __init__(self, domain, field=RealNumbers()):
        super().__init__(domain, field)

    def _inner(self, v1, v2):
        """Inner product, not computable in continuous spaces."""
        raise NotImplementedError('inner product not computable in the'
                                  'non-discretized space {}.'.format(self))

    def equals(self, other):
        """Test if `other` is equal to this space."""
        return isinstance(other, L2) and super().equals(other)

    def discretize(self, grid, interp='nearest', **kwargs):
        """Discretize the space with an interpolation dictionary.

        Parameters
        ----------
        grid : `TensorGrid`
            Sampling grid underlying the discretization. Must be
            contained in this space's domain.
        interp : `string`, optional
            The interpolation type to be used for discretization.

            'nearest' : use nearest-neighbor interpolation (default)

            'linear' : use linear interpolation

        kwargs : {'impl', 'order'}
            'impl' : 'numpy' or 'cuda'  (Default: 'numpy')
                The implementation of the data storage arrays
            'order' : 'C' or 'F'  (Default: 'C')
                The axis ordering in the data storage

        Returns
        -------
        l2discr : `DiscreteL2`
            The discretized space
        """
        from odl.discr.default import DiscreteL2

        if not isinstance(grid, TensorGrid):
            raise TypeError('{} is not a `TensorGrid` instance.'.format(grid))
        if not self.domain.contains_set(grid):
            raise ValueError('{} is not contained in the domain {} of the '
                             'space {}'.format(grid, self.domain, self))

        impl = kwargs.pop('impl', 'numpy')
        # TODO: use the consistent inner products instead of the standard ones
        if self.field == RealNumbers():
            if impl == 'numpy':
                dspace_type = Rn
            elif impl == 'cuda':
                if not CUDA_AVAILABLE:
                    raise ValueError('CUDA backend not available.')
                else:
                    dspace_type = CudaRn
        elif self.field == ComplexNumbers():
            if impl == 'numpy':
                dspace_type = Cn
            elif impl == 'cuda':
                if not CUDA_AVAILABLE:
                    raise ValueError('CUDA backend not available.')
                else:
                    raise NotImplementedError
                    # dspace_type = CudaEuclideanCn
        return DiscreteL2(self, grid, dspace_type(grid.ntotal), interp,
                          **kwargs)

    def __str__(self):
        if isinstance(self.field, RealNumbers):
            return 'L2({})'.format(self.domain)
        else:
            return 'L2({}, {})'.format(self.domain, self.field)

    def __repr__(self):
        if isinstance(self.field, RealNumbers):
            return 'L2({!r})'.format(self.domain)
        else:
            return 'L2({!r}, {!r})'.format(self.domain, self.field)

    class Vector(FunctionSpace.Vector):
        """Representation of an `L2` element."""
