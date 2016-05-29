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

"""Operators defined on `DiscreteLp`."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from odl.operator.operator import Operator


__all__ = ('Resampling',)


class Resampling(Operator):

    """An operator that resamples a vector on another grid.

    The operator uses the underlying `DiscretizedSet.sampling` and
    `DiscretizedSet.interpolation` operators to achieve this.

    The spaces need to have the same `DiscretizedSet.uspace` in order
    for this to work. The data space types may be different, although
    performance may vary drastically.
    """

    def __init__(self, domain, range):
        """Initialize a Resampling.

        Parameters
        ----------
        domain : `LinearSpace`
            The space that should be cast from
        range : `LinearSpace`
            The space that should be cast to

        Examples
        --------
        Create two spaces with different number of points and a resampling
        operator.

        >>> import odl
        >>> X = odl.uniform_discr(0, 1, 3)
        >>> Y = odl.uniform_discr(0, 1, 6)
        >>> resampling = Resampling(X, Y)
        """
        if domain.uspace != range.uspace:
            raise ValueError('domain.uspace ({}) does not match range.uspace '
                             '({})'.format(domain.uspace, range.uspace))

        super().__init__(domain=domain, range=range, linear=True)

    def _call(self, x, out=None):
        """Apply resampling operator.

        The vector ``x`` is resampled using the sampling and interpolation
        operators of the underlying spaces.

        Examples
        --------
        Create two spaces with different number of points and create resampling
        operator. Apply operator to vector.

        >>> import odl
        >>> X = odl.uniform_discr(0, 1, 3)
        >>> Y = odl.uniform_discr(0, 1, 6)
        >>> resampling = Resampling(X, Y)
        >>> print(resampling([0, 1, 0]))
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]

        The result depends on the interpolation chosen for the underlying
        spaces.

        >>> Z = odl.uniform_discr(0, 1, 3, interp='linear')
        >>> linear_resampling = Resampling(Z, Y)
        >>> print(linear_resampling([0, 1, 0]))
        [0.0, 0.25, 0.75, 0.75, 0.25, 0.0]
        """
        if out is None:
            return x.interpolation
        else:
            out.sampling(x.interpolation)

    @property
    def inverse(self):
        """Return an (approximate) inverse.

        Returns
        -------
        inverse : Resampling
            The resampling operator defined in the inverse direction.

        See Also
        --------
        adjoint : resampling is unitary, so adjoint is inverse.
        """
        return Resampling(self.range, self.domain)

    @property
    def adjoint(self):
        """Return an (approximate) adjoint.

        The result is only exact if the interpolation and sampling operators
        of the underlying spaces match exactly.

        Returns
        -------
        adjoint : Resampling
            The resampling operator defined in the inverse direction.

        Examples
        --------
        Create resampling operator and inverse

        >>> import odl
        >>> X = odl.uniform_discr(0, 1, 3)
        >>> Y = odl.uniform_discr(0, 1, 6)
        >>> resampling = Resampling(X, Y)
        >>> resampling_inv = resampling.inverse

        The inverse is proper left inverse if the resampling goes from a
        lower sampling to a higher sampling

        >>> x = [0.0, 1.0, 0.0]
        >>> print(resampling_inv(resampling(x)))
        [0.0, 1.0, 0.0]

        But can fail in the other direction

        >>> y = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        >>> print(resampling(resampling_inv(y)))
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        """
        return self.inverse

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
