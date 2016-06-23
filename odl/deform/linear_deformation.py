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
import odl

__all__ = ('LinearDeformation', 'MassPreservingLinearDeformation',
           'LinearizedDeformationOperator',
           'MassPreservingDeformationOperator')


class LinearizedDeformationOperator(Operator):
    """
    Linearized deformation operator mapping parameters into a
    deformed template. It is a left action given by pushforward.

    This operator computes the deformed template for a fixed
    template ``I``:

        v --> I(Id + v)

    where ``Id`` is the identity mapping:

        y --> y

    and the vector field ``v`` is a displacement mapping:

        y --> v(y)

    Here, the ``y`` is an element in the domain of the target.
    In diffeomorphic sense,  the domain of ``I`` is the same
    as the target.
    """

    def __init__(self, template):
        """Initialize a new instance.

        Parameters
        ----------
        template : `DiscreteLpVector`
            Fixed template deformed by the vector field.
        """
        self.template = template
        self.domain_space = odl.ProductSpace(self.template.space,
                                             self.template.space.ndim)

        super().__init__(self.domain_space, self.template.space, linear=False)

    def _call(self, displacement):
        """Implementation of ``self(displacement)``.

        Parameters
        ----------
        displacement : `ProductSpaceVector`
            Linearized deformation parameters for image grid points.
        """
        image_pts = self.template.space.grid.points()
        image_pts += np.asarray(displacement).T

        return self.template.interpolation(image_pts.T, bounds_check=False)

    def linear_deform(self, template, displacement):
        """Implementation of ``self(template, displacement)``.

        This function computes the deformed template for a given
        template ``I`` and displacement ``v``:

            (I, v) --> I(Id + v)

        Parameters
        ----------
        template : `DiscreteLpVector`
            Fixed template deformed by the vector field.
        displacement : `ProductSpaceVector`
            Linearized deformation parameters for image grid points.
        """
        image_pts = template.space.grid.points()
        image_pts += np.asarray(displacement).T

        return template.interpolation(image_pts.T, bounds_check=False)


class MassPreservingDeformationOperator(Operator):
    """
    Mass-preserving deformation operator mapping parameters to a
    deformed template. It is a left action given by pushforward.

    This operator computes the deformed template
    for a fixed template ``I``:

        (jacdetinvdef, invdef) --> jacdetinvdef * I(invdef)

    where ``invdef`` is the inverse of deformation mapping:

        y --> invdef(y)

    and ``jacdetinvdef`` is the Jacobian determinant of ``invdef``:

        y --> jacdetinvdef(y)

    Here, ``y`` is an element in the domain of target.
    In diffeomorphic sense,  the domain of ``I`` is the same
    as the target.
    """

    def __init__(self, template):
        """Initialize a new instance.

        Parameters
        ----------
        template : `DiscreteLpVector`
            Fixed template deformed by the vector field.
        """
        self.template = template
        self.domain_space = odl.ProductSpace(self.template.space,
                                             self.template.space.ndim + 1)
        self.invdef_pts = np.empty([self.template.space.ndim,
                                    self.template.size])

        super().__init__(self.domain_space, self.template.space, linear=False)

    def _call(self, jacdetinvdef_invdef):
        """Implementation of ``self(jacdetinvdef_invdef)``.

        Parameters
        ----------
        jacdetinvdef_invdef : 'ProductSpaceVector'

                R^{n} --> R^{1} times R^{n}

            The ``jacdetinvdef_invdef`` is defined on a product
            space ``R^{1} times R^{n}`` for ``jacdetinvdef`` and
            ``invdef``, respectively, where

            jacdetinvdef : 'DiscreteLpVector'

                    R^{n} --> R^{1}

                The Jacobian determinant of the inverse deformation.
                The ``jacdetinvdef`` equals to ``jacdetinvdef_invdef[0]``.
            invdef : `ProductSpaceVector`

                    R^{n} --> R^{n}

                General inverse deformation for image grid points.
                The ``invdef`` equals to ``jacdetinvdef_invdef[1:n]``.
        """
        for i in range(self.template.space.ndim):
            self.invdef_pts[i] = \
                jacdetinvdef_invdef[i+1].ntuple.asarray()
        return jacdetinvdef_invdef[0] * self.template.space.element(
            self.template.interpolation(self.invdef_pts, bounds_check=False))

    def mp_deform(self, template, jacdetinvdef_invdef):
        """Implementation of ``self(template, jacdetinvdef_invdef)``.

        This function computes the deformed template for a given
        template ``I``, and a given inverse deformation ``invdef``
        and its Jacobian determinant ``jacdetinvdef``:

            (I, jacdetinvdef, invdef) --> jacdetinvdef * I(invdef)

        Parameters
        ----------
        template : `DiscreteLpVector`
            Fixed template deformed by the vector field.
        jacdetinvdef_invdef : 'ProductSpaceVector'

                R^{n} --> R^{1} times R^{n}

            The ``jacdetinvdef_invdef`` is defined on a product
            space ``R^{1} times R^{n}`` for ``jacdetinvdef`` and
            ``invdef``, respectively, where

            jacdetinvdef : 'DiscreteLpVector'

                    R^{n} --> R^{1}

                The Jacobian determinant of the inverse deformation.
                The ``jacdetinvdef`` equals to ``jacdetinvdef_invdef[0]``.
            invdef : `ProductSpaceVector`

                    R^{n} --> R^{n}

                General inverse deformation for image grid points.
                The ``invdef`` equals to ``jacdetinvdef_invdef[1:n]``.
        """
        invdef_pts = np.empty([template.space.ndim, template.size])
        for i in range(template.space.ndim):
            invdef_pts[i] = jacdetinvdef_invdef[i+1].ntuple.asarray()
        return jacdetinvdef_invdef[0] * template.space.element(
            template.interpolation(invdef_pts, bounds_check=False))


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

        assert isinstance(offsets.space, odl.ProductSpace)
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
        pspace = odl.ProductSpace(space, space.ndim)
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

        assert isinstance(offsets.space, odl.ProductSpace)
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

        # offsets.show('offsets')

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
        pspace = odl.ProductSpace(space, space.ndim)
        vector_field = pspace.zero()
        jacobian = space.one()
        deformation = MassPreservingLinearDeformation(vector_field,
                                                      jacobian)
        return deformation

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
