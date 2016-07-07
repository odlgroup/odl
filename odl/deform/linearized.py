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


__all__ = ('LinDeformFixedTempl', 'LinDeformFixedTemplDeriv',
           'LinDeformFixedTemplDerivAdj', 'LinDeformFixedDisp',
           'LinDeformFixedDispAdj')


def linear_deform(template, displacement, out=None):
    """Linearized deformation of a template with a displacement field.

    The template ``I`` is deformed by a displacement field ``v`` as:

        (I, v) --> I(Id + v)

    where ``Id`` is the identity mapping:

        y --> y

    and the vector field ``v`` is a displacement field:

        y --> v(y)

    Here, the ``y`` is an element in the domain.

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
    >>> linear_deform(template, displacement_field)
    array([ 0.,  0.,  1.,  1.,  0.])
    """
    image_pts = template.space.points()
    for i, vi in enumerate(displacement):
        image_pts[:, i] += vi.asarray().ravel()
    return template.interpolation(image_pts.T, out=out, bounds_check=False)


class LinDeformFixedTempl(Operator):

    """Operator mapping a displacement field to corresponding deformed template.

    The operator maps a displacement field to the corresponding
    deformed fixed template ``I``:

        v --> I(Id + v)

    where ``Id`` is the identity mapping:

        y --> y

    and the vector field ``v`` is a displacement field:

        y --> v(y)

    Here, the ``y`` is an element in the domain.
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

        self.template = template

    def _call(self, displacement):
        """Implementation of ``self(displacement)``.

        Parameters
        ----------
        displacement : `domain` element
            The displacement field used in the linearized deformation.
        """
        return linear_deform(self.template, displacement)

    def derivative(self, displacement):
        """Derivative of the operator in ``displacement``.

        Parameters
        ----------
        displacement : `domain` element
            Point that the derivative need to be computed at.

        Returns
        -------
        derivative : `LinDeforFixedTempDeriv`
            The derivative evaluated at ``displacement``.
        """
        return LinDeformFixedTemplDeriv(self.template, displacement)


class LinDeformFixedTemplDeriv(Operator):

    """Derivative of the fixed template linearized deformation operator.

    This operator computes the derivative of the fixed template
    linearized deformation operator for the fixed
    template ``I`` at given displacement ``v``:

        u --> grad(I)(Id + v).T u

    where the ``u`` is a given vector field and
    the ``Id`` is the identity mapping:

        y --> y

    and the vector field ``v`` is a displacement field:

        y --> v(y)

    Here, the ``y`` is an element in the domain.
    """

    def __init__(self, template, displacement):
        """Initialize a new instance.

        Parameters
        ----------
        template : `DiscreteLp` element
            Fixed template deformed by the vector field.
        displacement: `ProductSpace` element
            Point at which the derivative is taken.

        Examples
        --------
        >>> import odl
        >>> space = odl.uniform_discr(0, 1, 5)
        >>> disp_field_space = odl.ProductSpace(space, space.ndim)
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> disp_field = disp_field_space.element([[0, 0, 0, -0.2, 0]])
        >>> vector_field = disp_field_space.element([[1, 1, 1, 2, 1]])
        >>> op = LinDeformFixedTemplDeriv(template, disp_field)
        >>> op(vector_field)
        uniform_discr(0.0, 1.0, 5).element([0.0, 5.0, -5.0, -10.0, 0.0])
        """
        if displacement[0] not in template.space:
            raise TypeError('`displacement[0]` {!r} not an element of'
                            '`template.space`'.format(displacement[0],
                                                      template.space))

        Operator.__init__(self, displacement.space, template.space,
                          linear=True)

        self.template = template
        self.displacement = displacement

        grad = Gradient(self.range, method='forward',
                        padding_method='symmetric')
        grad_template = grad(self.template)

        self.def_grad = self.displacement.space.element(
            [linear_deform(gf, self.displacement) for gf in grad_template])

    def _call(self, vector_field):
        """Implementation of ``self(vector_field)``.

        Parameters
        ----------
        vector_field: `domain` element
            The evaluation point, i.e. a displacement field, for the
            deformed gradient of the template.
        """
        inner_op = PointwiseInner(self.displacement.space, self.def_grad)

        return inner_op(vector_field)

    @property
    def adjoint(self):
        """Adjoint of the derivative operator.

        Returns
        -------
        adjoint : `LinDeformFixedTemplDerivAdj`
            The adjoint of the operator.
        """
        return LinDeformFixedTemplDerivAdj(self.template, self.displacement,
                                           self.def_grad)


class LinDeformFixedTemplDerivAdj(Operator):

    """Adjoint operator of the derivative operator.

    This operator computes the adjoint of the derivative of the fixed
    template linearized deformation operator for the fixed template ``I``
    at given displacement ``v``:

        J --> grad(I)(Id + v) J

    where the ``J`` is the given template and
    the ``Id`` is the identity mapping:

        y --> y

    and the vector field ``v`` is a displacement mapping:

        y --> v(y)

    Here, the ``y`` is an element in the domain.
    """

    def __init__(self, template, displacement, def_grad=None):
        """Initialize a new instance.

        Parameters
        ----------
        template : `DiscreteLp` element
            Fixed template deformed by the vector field.
        displacement: `ProductSpace` element
            Point that the adjoint of the derivative needs
            to be computed at.

        Examples
        --------
        >>> import odl
        >>> space = odl.uniform_discr(0, 1, 5)
        >>> disp_field_space = odl.ProductSpace(space, space.ndim)
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> given_template = space.element([1, 2, 1, 2, 1])
        >>> disp_field = disp_field_space.element([[0, 0, 0, -0.2, 0]])
        >>> op = LinDeformFixedTemplDerivAdj(template, disp_field)
        >>> op(given_template)
        ProductSpace(uniform_discr(0.0, 1.0, 5), 1).element([
            [0.0, 10.0, -5.0, -10.0, 0.0]
        ])
        """
        if displacement[0] not in template.space:
            raise TypeError('`displacement[0]` {!r} not an element of'
                            '`template.space`'.format(displacement[0],
                                                      template.space))

        Operator.__init__(self, template.space, displacement.space,
                          linear=True)

        if def_grad is None:
            self.template = template
            self.displacement = displacement

            grad = Gradient(self.domain, method='forward',
                            padding_method='symmetric')
            template_grad = grad(self.template)

            self.def_grad = self.displacement.space.element(
                [linear_deform(gf, self.displacement) for gf in template_grad])
        else:
            self.def_grad = def_grad

    def _call(self, func):
        """Implement ``self(func)```.

        Parameters
        ----------
        func : `domain` element
            The evaluation point, i.e. an element in the template space,
            for the deformed gradient of the template.
        """
        return [gf * func for gf in self.def_grad]

    @property
    def adjoint(self):
        """Adjoint of the ajoint operator.

        Returns
        -------
        adjoint : `LinDeformFixedTemplDeriv`
            The adjoint of the operator.
        """
        return LinDeformFixedTemplDeriv(self.template, self.displacement)


class LinDeformFixedDisp(Operator):

    """Deformation operator with fixed displacement acting on template.

    This linear operator maps a template ``I`` to the corresponding
    deformed template using a fixed displacement field ``v``:

        I --> I(Id + v)

    where the ``I`` is a given template, ``Id`` is the identity mapping:

        y --> y

    and the vector field ``v`` is a displacement mapping:

        y --> v(y)

    Here, the ``y`` is an element in the domain.
    """

    def __init__(self, displacement, domain=None):
        """Initialize a new instance.

        Parameters
        ----------
        displacement : `ProductSpace` element
            Fixed displacement field used in the linearized deformation.

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
            if not isinstance(displacement.space, ProductSpace):
                raise TypeError('`displacement` {!r} not an element'
                                'of `ProductSpace`'.format(displacement))
            if not isinstance(displacement.space[0], DiscreteLp):
                raise TypeError('`displacement[0]` {!r} not an element of'
                                '`DiscreteLp`'.format(displacement[0]))

            domain = displacement[0].space
        else:
            displacement = ProductSpace(domain, domain.ndim).element(
                displacement)

        Operator.__init__(self, domain, domain, linear=True)

        self.displacement = displacement

    def _call(self, template):
        """Implementation of ``self(template)``.

        Parameters
        ----------
        template : `domain` element
            Given template that is to be deformed by the fixed
            displacement field.
        """
        return linear_deform(template, self.displacement)

    @property
    def adjoint(self):
        """Adjoint of the operator.

        Returns
        -------
        adjoint: `LinDeformFixedTemplDerivAdj`
            The adjoint of the operator.
        """
        return LinDeformFixedDispAdj(self.displacement)


class LinDeformFixedDispAdj(Operator):

    """Adjoint of the fixed displacement linearized deformation operator.

    This operator computes the adjoint of the linear operator that
    map a template ``I`` to its deformation using a fixed displacement ``v``:

        I --> exp(-div(v)) * I(Id - v)

    Here, the ``I`` is the given template, ``Id`` is the identity mapping:

        y --> y

    the vector field ``v`` is a displacement field:

        y --> v(y)

    and the ``y`` is an element in the domain.

    Here, ``exp(-div(v))`` is an approximation of the determinant of the
    Jacobian of ``(Id + v)^{-1}``, which is valid if the magnitude
    of``v`` is close to ``0``.
    """

    def __init__(self, displacement):
        """Initialize a new instance.

        Parameters
        ----------
        displacement : `ProductSpace` element
            Fixed displacement field used in the linearized deformation.

        Examples
        --------
        >>> import odl
        >>> space = odl.uniform_discr(0, 1, 5)
        >>> disp_field_space = odl.ProductSpace(space, space.ndim)
        >>> disp_field = disp_field_space.element([[0.2, 0.2, 0.2, 0.2, 0.2]])
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> op = LinDeformFixedDispAdj(disp_field)
        >>> op(template)
        uniform_discr(0.0, 1.0, 5).element([0.0, 0.0, 0.0, 1.0, 0.0])
        """
        if not isinstance(displacement.space, ProductSpace):
            raise TypeError('`displacement` {!r} not an element'
                            'of `ProductSpace`'.format(displacement))
        if not isinstance(displacement.space[0], DiscreteLp):
            raise TypeError('`displacement[0]` {!r} not an element of'
                            '`DiscreteLp`'.format(displacement[0]))

        domain = displacement[0].space

        Operator.__init__(self, domain, domain, linear=True)

        self.displacement = displacement

    def _call(self, template):
        """Implement ``self(template)```.

        Parameters
        ----------
        template : `domain` element
            Given template that is to be deformed by the fixed
            displacement field.
        """
        div_op = Divergence(range=template.space, method='forward',
                            padding_method='symmetric')
        jacobian_det = np.exp(-div_op(self.displacement))
        return jacobian_det * template.space.element(
            linear_deform(template, -self.displacement))

    @property
    def adjoint(self):
        """Adjoint of the operator.

        Returns
        -------
        adjoint: `LinDeformFixedDisp`
            The adjoint of the operator.
        """
        return LinDeformFixedDisp(self.displacement)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
