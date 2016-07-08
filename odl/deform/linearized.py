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

"""Operators and functions for linearized deformation."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from odl.operator.operator import Operator
from odl import ProductSpace, DiscreteLp, Gradient, Divergence
from odl import PointwiseInner, DiscreteLpVector
import numpy as np


__all__ = ('LinDeformFixedTempl', 'LinDeformFixedDisp')


def _linear_deform(template, displacement, out=None):
    """Linearized deformation of a template with a displacement field.

    The function maps a gived template ``I`` and a given displacement
    field ``v`` to the new function ``x --> I(x + v(x))``.

    Parameters
    ----------
    template : `DiscreteLpVector`
        Template to be deformed by a displacement field.
    displacement : `ProductSpace` element
        The vector field (displacement field) used in the linearized
        deformation.

    Returns
    -------
    deformed_template : `numpy.ndarray`

    Examples
    --------
    >>> import odl
    >>> space = odl.uniform_discr(0, 1, 5)
    >>> disp_field_space = odl.ProductSpace(space, space.ndim)
    >>> template = space.element([0, 0, 1, 0, 0])
    >>> displacement_field = disp_field_space.element([[0, 0, 0, -0.2, 0]])
    >>> _linear_deform(template, displacement_field)
    array([ 0.,  0.,  1.,  1.,  0.])
    """
    image_pts = template.space.points()
    for i, vi in enumerate(displacement):
        image_pts[:, i] += vi.asarray().ravel()
    return template.interpolation(image_pts.T, out=out, bounds_check=False)


class LinDeformFixedTempl(Operator):

    """Operator mapping a displacement field to corresponding deformed template.

    The operator has a fixed template ``I`` and maps a displacement
    field ``v`` to the new function ``x --> I(x + v(x))``.
    """

    def __init__(self, template, domain=None):
        """Initialize a new instance.

        Parameters
        ----------
        template : `DiscreteLpVector` or array-like
            Fixed template that is to be deformed.
            If ``domain`` is not given, ``template`` must
            be a `DiscreteLpVector`, and the domain of this operator
            is inferred from ``template.space``. If ``domain`` is
            given, ``template`` can be anything that is understood
            by the ``domain[0].element()`` method.
        domain : product space of `DiscreteLp`, optional
            Space of displacement fields on which this operator
            acts, i.e. the operator domain. If not given, it is
            inferred from ``template.space``.

        Examples
        --------
        >>> import odl
        >>> space = odl.uniform_discr(0, 1, 5)
        >>> disp_field_space = odl.ProductSpace(space, space.ndim)
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> displacement_field = disp_field_space.element([[0, 0, 0, -0.2, 0]])
        >>> op = LinDeformFixedTempl(template)
        >>> op(displacement_field)
        uniform_discr(0.0, 1.0, 5).element([0.0, 0.0, 1.0, 1.0, 0.0])
        """
        if domain is None:
            if not isinstance(template, DiscreteLpVector):
                raise TypeError('`template` must be a `DiscreteLpVector`'
                                'instance if `domain` is None, got {!r}'
                                ''.format(template))

            domain = ProductSpace(template.space, template.space.ndim)
        else:
            template = domain[0].element(template)

        Operator.__init__(self, domain, domain[0], linear=False)

        self._template = template

    def _call(self, displacement, out=None):
        """Implementation of ``self(displacement)``.

        Parameters
        ----------
        displacement : `domain` element
            The displacement field used in the linearized deformation.
        """
        return _linear_deform(self._template, displacement, out)

    def derivative(self, displacement):
        """Derivative of the operator in ``displacement``.

        Parameters
        ----------
        displacement : `domain` element-like
            Point that the derivative need to be computed at.

        Returns
        -------
        derivative : `PointwiseInner`
            The derivative evaluated at ``displacement``.
        """
        displacement = self.domain.element(displacement)

        # TODO allow users to select what method to use here.
        grad = Gradient(self.range, method='central',
                        padding_method='symmetric')
        grad_templ = grad(self._template)
        self.def_grad = displacement.space.element(
            [_linear_deform(gf, displacement) for gf in grad_templ])

        inner_op = PointwiseInner(displacement.space, self.def_grad)

        return inner_op


class LinDeformFixedDisp(Operator):

    """Deformation operator with fixed displacement acting on template.

    The operator has a fixed displacement field ``v`` and
    maps a template ``I`` to the new function ``x --> I(x + v(x))``.
    """

    def __init__(self, displacement, domain=None):
        """Initialize a new instance.

        Parameters
        ----------
        displacement : `ProductSpace` element or array-like
            Fixed displacement field used in the linearized deformation.
            If ``domain`` is not given, ``displacement`` must
            be a `ProductSpace` element, and the domain of this operator
            is inferred from ``displacement[0].space``. If ``domain`` is
            given, ``displacement`` can be anything that is understood
            by the ``ProductSpace(domain, domain.ndim).element()`` method.
        domain : `DiscreteLp`, optional
            Space of templates on which this operator acts, i.e. the operator
            domain. If not given, ``displacement[0].space`` is used as domain.

        Examples
        --------
        >>> import odl
        >>> space = odl.uniform_discr(0, 1, 5)
        >>> disp_field_space = odl.ProductSpace(space, space.ndim)
        >>> displacement_field = disp_field_space.element([[0, 0, 0, -0.2, 0]])
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> op = LinDeformFixedDisp(displacement_field)
        >>> op(template)
        uniform_discr(0.0, 1.0, 5).element([0.0, 0.0, 1.0, 1.0, 0.0])
        """
        if domain is None:
            if not isinstance(displacement.space[0], DiscreteLp):
                raise TypeError('`displacement[0]` {!r} not an element of'
                                '`DiscreteLp`'.format(displacement[0]))
            if not displacement.space.is_power_space:
                raise TypeError('`displacement.space` {!r} not a product'
                                'space'.format(displacement.space))

            domain = displacement[0].space
        else:
            displacement = ProductSpace(domain, domain.ndim).element(
                displacement)

        Operator.__init__(self, domain, domain, linear=True)

        self._displacement = displacement

    def _call(self, template, out=None):
        """Implementation of ``self(template)``.

        Parameters
        ----------
        template : `domain` element
            Given template that is to be deformed by the fixed
            displacement field.
        """
        return _linear_deform(template, self._displacement, out)

    @property
    def adjoint(self):
        """Adjoint of the operator.

        Returns
        -------
        adjoint: `Operator`
            The adjoint of the operator.
        """

        # TODO allow users to select what method to use here.
        div_op = Divergence(range=self.domain, method='forward',
                            padding_method='symmetric')
        jacobian_det = np.exp(-div_op(self._displacement))
        return jacobian_det * LinDeformFixedDisp(-self._displacement)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
