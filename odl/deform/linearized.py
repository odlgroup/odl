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
        template : `DiscreteLpVector` or element-like
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
        Create a template and deform it with a given deformation field.

        Where the deformation field is zero we expect to get the same output
        as the input. In the 4:th point, the deformation is non-zero and hence
        we expect to get the value of the point 0.2 to the left, that is 1.0.

        >>> import odl
        >>> space = odl.uniform_discr(0, 1, 5, interp='nearest')
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> op = LinDeformFixedTempl(template)
        >>> disp_field = [[0, 0, 0, -0.2, 0]]
        >>> print(op(disp_field))
        [0.0, 0.0, 1.0, 1.0, 0.0]

        The result depends on the chosen interpolation. If we chose 'linear'
        interpolation and offset the point half the distance between two
        points, 0.1, we expect to get the mean of the values.

        >>> space = odl.uniform_discr(0, 1, 5, interp='linear')
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> op = LinDeformFixedTempl(template)
        >>> disp_field = [[0, 0, 0, -0.1, 0]]
        >>> print(op(disp_field))
        [0.0, 0.0, 1.0, 0.5, 0.0]

        See Also
        --------
        LinDeformFixedDisp : Deformation with a fixed displacement.
        """
        if domain is None:
            if not isinstance(template, DiscreteLpVector):
                raise TypeError('`template` must be a `DiscreteLpVector`'
                                'instance if `domain` is None, got {!r}'
                                ''.format(template))

            domain = template.space.tangent_space
        else:
            if not isinstance(domain, ProductSpace):
                raise TypeError('`domain` {!r} not a `ProductSpace`'
                                ''.format(domain))
            if not domain.is_power_space:
                raise TypeError('`domain` {!r} not a product'
                                'space'.format(domain))
            if not isinstance(domain[0], DiscreteLp):
                raise TypeError('`domain[0]` {!r} not a `DiscreteLp`'
                                ''.format(domain))
            if not domain[0].is_rn:
                raise TypeError('`domain[0]` {!r} not a real space'
                                ''.format(domain[0]))

            template = domain[0].element(template)

        Operator.__init__(self, domain, template.space, linear=False)

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
        # To implement the complex case we need to be able to embed the real
        # tangent space into the range of the gradient. Issue #59.
        if not self.range.is_rn:
            raise NotImplementedError('derivative not implemented for complex '
                                      'spaces.')

        displacement = self.domain.element(displacement)

        # TODO allow users to select what method to use here.
        grad = Gradient(domain=self.range, method='central',
                        padding_method='symmetric')
        grad_templ = grad(self._template)
        def_grad = self.domain.element(
            [_linear_deform(gf, displacement) for gf in grad_templ])

        return PointwiseInner(self.domain, def_grad)

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.domain == self._template.space.tangent_space:
            domain_repr = ''
        else:
            domain_repr = ', domain={!r}'.format(self.domain)

        return '{}({!r}{})'.format(self.__class__.__name__,
                                   self._template,
                                   domain_repr)


class LinDeformFixedDisp(Operator):

    """Deformation operator with fixed displacement acting on template.

    The operator has a fixed displacement field ``v`` and
    maps a template ``I`` to the new function ``x --> I(x + v(x))``.
    """

    def __init__(self, displacement, domain=None):
        """Initialize a new instance.

        Parameters
        ----------
        displacement : `ProductSpace` element-like
            Fixed displacement field used in the linearized deformation.
            If ``domain`` is not given, ``displacement`` must
            be a `ProductSpace` element, and the domain of this operator
            is inferred from ``displacement[0].space``. If ``domain`` is
            given, ``displacement`` can be anything that is understood
            by the ``domain.tangent_space.element()`` method.
        domain : `DiscreteLp`, optional
            Space of templates on which this operator acts, i.e. the operator
            domain. If not given, ``displacement[0].space`` is used as domain.

        Examples
        --------
        Create a given deformation and use it to deform a function.

        Where the deformation field is zero we expect to get the same output
        as the input. In the 4:th point, the deformation is non-zero and hence
        we expect to get the value of the point 0.2 to the left, that is 1.0.

        >>> import odl
        >>> space = odl.uniform_discr(0, 1, 5)
        >>> disp_field = space.tangent_space.element([[0, 0, 0, -0.2, 0]])
        >>> op = LinDeformFixedDisp(disp_field)
        >>> template = [0, 0, 1, 0, 0]
        >>> print(op([0, 0, 1, 0, 0]))
        [0.0, 0.0, 1.0, 1.0, 0.0]

        The result depends on the chosen interpolation. If we chose 'linear'
        interpolation and offset the point half the distance between two
        points, 0.1, we expect to get the mean of the values.

        >>> space = odl.uniform_discr(0, 1, 5, interp='linear')
        >>> disp_field = space.tangent_space.element([[0, 0, 0, -0.1, 0]])
        >>> op = LinDeformFixedDisp(disp_field)
        >>> template = [0, 0, 1, 0, 0]
        >>> print(op(template))
        [0.0, 0.0, 1.0, 0.5, 0.0]

        See Also
        --------
        LinDeformFixedTempl : Deformation with a fixed template.
        """
        if domain is None:
            if not isinstance(displacement.space, ProductSpace):
                raise TypeError('`displacement.space` {!r} not a '
                                '`ProductSpace`'.format(displacement.space))
            if not displacement.space.is_power_space:
                raise TypeError('`displacement.space` {!r} not a power'
                                'space'.format(displacement.space))
            if not isinstance(displacement[0].space, DiscreteLp):
                raise TypeError('`displacement[0].space` {!r} not an '
                                '`DiscreteLp`'.format(displacement[0]))

            domain = displacement[0].space
        else:
            if not isinstance(domain, DiscreteLp):
                raise TypeError('`displacement[0]` {!r} not an `DiscreteLp`'
                                ''.format(displacement[0]))

            displacement = domain.tangent_space.element(displacement)

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
        div_op = Divergence(domain=self._displacement.space, method='forward',
                            padding_method='symmetric')
        jacobian_det = self.domain.element(np.exp(-div_op(self._displacement)))
        deformation = LinDeformFixedDisp(-self._displacement,
                                         domain=self.domain)
        return jacobian_det * deformation

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.domain == self._displacement.space[0]:
            domain_repr = ''
        else:
            domain_repr = ', domain={!r}'.format(self.domain)

        return '{}({!r}{})'.format(self.__class__.__name__,
                                   self._displacement,
                                   domain_repr)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
