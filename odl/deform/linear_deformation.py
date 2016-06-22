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

"""
Shape-based reconstruction using optimal information transportation.

The Fisher-Rao metric is used in regularization term. And L2 data matching
term is used in fitting term.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

from odl.operator import Operator
from odl.discr import DiscreteLp, Divergence
from odl.space import ProductSpace

__all__ = ('LinearDeformation', 'MassPreservingLinearDeformation')


class LinearDeformation(Operator):
    """Linearized deformation."""
    def __init__(self, offsets=None):
        """Initialize a linear deformation

        This has the form

        Id + offsets

        Parameters
        ----------
        vector_field : `ProductSpace` of `DiscreteLp` spaces
            The points to calculate the interpolation in.

        Examples
        --------
        Identity deformation:

        >>> import odl
        >>> space = odl.uniform_discr(0, 1, 5)
        >>> pspace = ProductSpace(space, space.ndim)
        >>> vector_field = pspace.element(space.points().T)
        >>> deformation = LinearDeformation(vector_field)
        """

        assert isinstance(offsets.space, ProductSpace)
        assert offsets.space.is_power_space
        assert isinstance(offsets.space[0], DiscreteLp)

        ndim = offsets.space.size
        space = offsets.space[0]

        interp_points = space.points().T
        for i in range(ndim):
            interp_points[i] += offsets[i].ntuple.asarray()

        self.interp_points = interp_points
        self.offsets = offsets

        Operator.__init__(self,
                          domain=space,
                          range=space,
                          linear=True)

    def _call(self, x):
        return x.interpolation(self.interp_points, bounds_check=False)

    @staticmethod
    def identity(space):
        """Create the identity transformation on a space."""
        pspace = ProductSpace(space, space.ndim)
        vector_field = pspace.zero()
        deformation = LinearDeformation(vector_field)
        return deformation


class MassPreservingLinearDeformation(Operator):
    def __init__(self, offsets, vector_field_jacobian, divergence=None):
        """Initialize a linear deformation

        Parameters
        ----------
        vector_field : `ProductSpace` of `DiscreteLp` spaces
            The (small) deformation field.
        vector_field_jacobian : `DiscreteLp`
            The determinant of the jacobian of the deformation.
        divergence : `Operator`
            The divergence operator used in composition.
            Default:
            Divergence(range=vector_field_jacobian.space, method='central')

        Examples
        --------
        Identity deformation:

        >>> import odl
        >>> space = odl.uniform_discr(0, 1, 5)
        >>> pspace = ProductSpace(space, space.ndim)
        >>> vector_field = pspace.element(space.points().T)
        >>> jacobian = space.zero()
        >>> deformation = MassPreservingLinearDeformation(vector_field,
        ...                                               jacobian)
        """

        assert isinstance(offsets.space, ProductSpace)
        assert offsets.space.is_power_space
        assert isinstance(offsets.space[0], DiscreteLp)
        assert vector_field_jacobian in offsets.space[0]

        ndim = offsets.space.size
        space = offsets.space[0]

        interp_points = space.points().T
        for i in range(ndim):
            interp_points[i] += offsets[i].ntuple.asarray()

        self.interp_points = interp_points
        self.offsets = offsets
        self.vector_field_jacobian = vector_field_jacobian

        #offsets.show('offsets')

        if divergence is None:
            self.divergence = Divergence(range=space, method='central')

        Operator.__init__(self,
                          domain=space,
                          range=space,
                          linear=True)

    def _call(self, x):
        result = x.interpolation(self.interp_points, bounds_check=False)
        result = self.range.element(result)
        result *= self.vector_field_jacobian
        return result

    def compose(self, linear_deformation):
        assert isinstance(linear_deformation, LinearDeformation)

        # Find the deformations for the output field
        out_offsets = self.offsets.space.element()

        for i in range(self.interp_points.shape[0]):
            as_element = self.domain.element(self.interp_points[i])
            interpolated = as_element.interpolation(
                linear_deformation.interp_points, bounds_check=False)
            out_offsets[i][:] = interpolated - self.domain.points().T[i]

        # Find the jacobian
        out_jacobian = self.domain.element(
            self.vector_field_jacobian.interpolation(
                linear_deformation.interp_points, bounds_check=False))

        # Weighting
        div = self.divergence(linear_deformation.offsets)
        out_jacobian *= np.exp(div)

        # out_jacobian = self.domain.one()
        return MassPreservingLinearDeformation(out_offsets, out_jacobian)

    @staticmethod
    def identity(space):
        """Create the identity transformation on a space."""
        pspace = ProductSpace(space, space.ndim)
        vector_field = pspace.zero()
        jacobian = space.one()
        deformation = MassPreservingLinearDeformation(vector_field,
                                                      jacobian)
        return deformation

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
