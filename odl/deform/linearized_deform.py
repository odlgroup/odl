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
Operators of linearized deformations.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from odl.operator.operator import Operator
import odl
import numpy as np


__all__ = ('LinDeformFixedTempl', 'LinDeformFixedTemplDeriv',
           'LinDeformFixedTemplDerivAdj', 'LinDeformFixedDisp',
           'LinDeformFixedDispAdj', 'linear_deform', 'deform_grad')


def linear_deform(template, displacement):
    """Linearized deformation of a template with a displacement field.

    The template ``I`` is deformed by a displacement field ``v`` as:

        (I, v) --> I(Id + v)

    where ``Id`` is the identity mapping:

        y --> y

    and the vector field ``v`` is a displacement field:

        y --> v(y)

    Parameters
    ----------
    template : `DiscreteLpVector`
        Template to be deformed by a displacement field.
    displacement : product space of `ProductSpaceVector`
        The vector field (displacement field) used in the linearized
        deformation.

    Returns
    -------
    deformed_template : `numpy.ndarray`
        Deformed template as an numpy array.

    Examples
    --------
    Deform a 1D function ``template`` that has value ``1`` at 3:rd sample
    point using a displacement field ``displacement_field`` that has
    value ``-0.2`` in the 4:th sample point. The outcome should be an
    array with values ``1`` at 3:rd and 4:th positions.

    >>> import odl
    >>> space = odl.uniform_discr(0, 1, 5)
    >>> disp_field_space = odl.ProductSpace(space, space.ndim)
    >>> template = space.element([0, 0, 1, 0, 0])
    >>> displacement_field = disp_field_space.element([[0, 0, 0, -0.2, 0]])
    >>> linear_deform(template, displacement_field)
    array([ 0.,  0.,  1.,  1.,  0.])
    """
    image_pts = template.space.grid.points()
    image_pts += np.asarray(displacement).T
    return template.interpolation(image_pts.T, bounds_check=False)


def deform_grad(grad_f, displacement):
    """Compute the deformation of the template gradient.

    The template gradient ``grad(I)`` after deformation with displacement
    field ``v``: grad(I)(Id + v)

    Parameters
    ----------
    grad_f: `ProductSpaceVector`
        Gradient of the template, i.e. a vector field on the
        image domain
    displacement : `ProductSpaceVector`
        The vector field (displacement field) used in the linearized
        deformation.
    Examples
    --------
    >>> import odl

    Define ``gradOp`` as the gradient operator, let it act on ``template``.

    >>> space = odl.uniform_discr(0, 1, 5)
    >>> template = space.element([0, 0, 1, 0, 0])
    >>> gradOp = odl.Gradient(template.space, method='forward')
    >>> template_grad = gradOp(template)
    >>> template_grad
    ProductSpace(uniform_discr(0.0, 1.0, 5), 1).element([
        [0.0, 5.0, -5.0, 0.0, 0.0]
    ])

    Define a displacement field ``displacement_field``.

    >>> disp_field_space = gradOp.range
    >>> displacement_field = disp_field_space.element([[0, 0, 0, -0.2, 0]])

    Calculate linearized deformation of the ``template_grad`` (gradient of
    the template) using the displacement field ``displacement_field``.

    >>> deform_grad(template_grad, displacement_field)
    ProductSpace(uniform_discr(0.0, 1.0, 5), 1).element([
        [0.0, 5.0, -5.0, -5.0, 0.0]
    ])
    """
    return displacement.space.element(
        [linear_deform(gf, displacement) for gf in grad_f])


class LinDeformFixedTempl(Operator):

    """Operator mapping a displacement field to corresponding deformed template.

    The operator maps a displacement field to the corresponding
    deformed fixed template ``I``. The linearized deformation is given by
    the left action defined through the pushforward, so

        v --> I(Id + v)

    where ``Id`` is the identity mapping:

        y --> y

    and the vector field ``v`` is a displacement field:

        y --> v(y)

    Here, the ``y`` is an element in the domain of the target.
    """

    def __init__(self, template):
        """Initialize a new instance.

        Parameters
        ----------
        template : `DiscreteLpVector`
            Fixed template that is to be deformed by the displacement field.

        Examples
        --------
        Deform a 1D function ``template`` that has value ``1`` at 3:rd sample
        point using a displacement field ``displacement_field`` that has
        value ``-0.2`` in the 4:th sample point. The outcome should be a
        1D function that has values ``1`` at 3:rd and 4:th sample points.

        >>> import odl
        >>> space = odl.uniform_discr(0, 1, 5)
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> op = LinDeformFixedTempl(template)
        >>> disp_field_space = odl.ProductSpace(space, space.ndim)
        >>> displacement_field = disp_field_space.element([[0, 0, 0, -0.2, 0]])
        >>> op(displacement_field)
        uniform_discr(0.0, 1.0, 5).element([0.0, 0.0, 1.0, 1.0, 0.0])
        """
        if not isinstance(template.space, odl.DiscreteLp):
            raise TypeError('`template.space` {!r} not an element of'
                            '`DiscreteLp`'.format(template.space))

        domain_space = odl.ProductSpace(template.space, template.space.ndim)
        range_space = template.space

        Operator.__init__(self, domain_space, range_space, linear=False)

        self.template = template

    def _call(self, displacement):
        """Implementation of ``self(displacement)``.

        Parameters
        ----------
        displacement : `domain` element
            The displacement field used in the linearized deformation.
        """
        if displacement[0] not in self.template.space:
            raise TypeError('`displacement[0]` {!r} not an element of'
                            '`template.space`'.format(displacement[0],
                                                      self.template.space))

        return linear_deform(self.template, displacement)

    def derivative(self, displacement):
        """Gateaux derivative of the operator in ``displacement``.

        Parameters
        ----------
        displacement : `domain` element
            The point that the Gateaux derivative need to be computed at.

        Returns
        -------
        derivative : `LinDeforFixedTempDeriv`
            The derivative evaluated at ``displacement``
        """
        return LinDeformFixedTemplDeriv(self.template, displacement)


class LinDeformFixedTemplDeriv(Operator):

    """Derivative of the fixed template linearized deformation operator.

    This operator computes the Gateaux derivative of the fixed
    template linearized deformation operator for the fixed
    template ``I`` at given displacement ``v``:

        u --> grad(I)(Id + v).T u

    where the ``u`` is a given vector field and
    the ``Id`` is the identity mapping:

        y --> y

    and the vector field ``v`` is a displacement field:

        y --> v(y)

    Here, the ``y`` is an element in the domain of the target.
    In diffeomorphic sense, the domain of ``I`` is the same
    as the target.
    """

    def __init__(self, template, displacement):
        """Initialize a new instance.

        Parameters
        ----------
        template : `DiscreteLpVector`
            Fixed template deformed by the vector field.
        displacement: `ProductSpaceVector`
            The point that the Gateaux derivative needs to be computed at.

        Examples
        --------
        >>> import odl
        >>> space = odl.uniform_discr(0, 1, 5)
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> disp_field_space = odl.ProductSpace(space, space.ndim)
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

    def _call(self, vector_field):
        """Implementation of ``self(vector_field)``.

        Parameters
        ----------
        vector_field: `domain` element
            The evaluation point, i.e. a displacement field, for the
            deformed gradient of the template.
        """
        if vector_field not in self.displacement.space:
            raise TypeError('`vector_field` {!r} not an element of'
                            '`displacement.space`'.format(
                                vector_field, self.displacement.space))

        grad = odl.Gradient(self.range)
        grad_template = grad(self.template)
        def_grad = deform_grad(grad_template, self.displacement)

        innerProdOp = odl.PointwiseInner(self.displacement.space, def_grad)

        return innerProdOp(vector_field)

    @property
    def adjoint(self):
        """Adjoint of the derivative operator.

        Returns
        -------
        adj_op : `LinDeformFixedTemplDerivAdj`
            The adjoint of the operator.
        """
        return LinDeformFixedTemplDerivAdj(self.template, self.displacement)


class LinDeformFixedTemplDerivAdj(Operator):

    """Adjoint operator of the derivative operator.

    This operator computes the adjoint of the Gateaux derivative of the fixed
    template linearized deformation operator for the fixed template ``I``
    at given displacement ``v``:

        J --> grad(I)(Id + v) J

    where the ``J`` is the given template and
    the ``Id`` is the identity mapping:

        y --> y

    and the vector field ``v`` is a displacement mapping:

        y --> v(y)

    Here, the ``y`` is an element in the domain of the target.
    In diffeomorphic sense, the domain of ``I`` is the same
    as the target.
    """

    def __init__(self, template, displacement):
        """Initialize a new instance.

        Parameters
        ----------
        template : `DiscreteLpVector`
            Fixed template deformed by the vector field.
        displacement: `ProductSpaceVector`
            The point that the adjoint of the Gateaux derivative needs
            to be computed at.

        Examples
        --------
        >>> import odl
        >>> space = odl.uniform_discr(0, 1, 5)
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> disp_field_space = odl.ProductSpace(space, space.ndim)
        >>> disp_field = disp_field_space.element([[0, 0, 0, -0.2, 0]])
        >>> op = LinDeformFixedTemplDerivAdj(template, disp_field)
        >>> given_template = space.element([1, 2, 1, 2, 1])
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

        self.template = template
        self.displacement = displacement

    def _call(self, func):
        """Implement ``self(func)```.

        Parameters
        ----------
        func : `DiscreteLpVector`
            The evaluation point, i.e. an element in the template space,
            for the deformed gradient of the template.
        """
        grad = odl.Gradient(self.domain)
        template_grad = grad(self.template)

        def_grad = deform_grad(template_grad, self.displacement)

        return [gf * func for gf in def_grad]


class LinDeformFixedDisp(Operator):

    """Operator using fixed displacement to map template to deformed template.

    This linear operator maps a template ``I`` to the corresponding
    deformed template using a fixed displacement field ``v``.
    The linearized deformation is given by the left action defined
    through the pushforward, so

        I --> I(Id + v)

    where the ``I`` is a given template, ``Id`` is the identity mapping:

        y --> y

    and the vector field ``v`` is a displacement mapping:

        y --> v(y)

    Here, the ``y`` is an element in the domain of the target.
    In diffeomorphic sense, the domain of ``I`` is the same
    as the target.
    """

    def __init__(self, displacement):
        """Initialize a new instance.

        Parameters
        ----------
        displacement : `ProductSpaceVector`
            Fixed displacement field used in the linearized deformation.

        Examples
        --------
        >>> import odl
        >>> space = odl.uniform_discr(0, 1, 5)
        >>> disp_field_space = odl.ProductSpace(space, space.ndim)
        >>> displacement_field = disp_field_space.element([[0, 0, 0, -0.2, 0]])
        >>> op = LinDeformFixedDisp(displacement_field)
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> op(template)
        uniform_discr(0.0, 1.0, 5).element([0.0, 0.0, 1.0, 1.0, 0.0])
        """
        if not isinstance(displacement.space, odl.ProductSpace):
            raise TypeError('`displacement` {!r} not an element'
                            'of `ProductSpace`'.format(displacement))
        if not isinstance(displacement.space[0], odl.DiscreteLp):
            raise TypeError('`displacement[0]` {!r} not an element of'
                            '`DiscreteLp`'.format(displacement[0]))

        domain_space = displacement[0].space

        Operator.__init__(self, domain_space, domain_space, linear=True)

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
        adj_op : `LinDeformFixedDispAdj`
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

    and, ``exp(-div(v))`` is an approximation of the determinant of the
    Jacobian of ``(Id + v)^{-1}``, which is accurate in some sense if the
    magnitude of``v`` is close to ``0``.
    """

    def __init__(self, displacement):
        """Initialize a new instance.

        Parameters
        ----------
        displacement : `ProductSpaceVector`
            Fixed displacement field used in the linearized deformation.

        Examples
        --------
        >>> import odl
        >>> space = odl.uniform_discr(0, 1, 5)
        >>> disp_field_space = odl.ProductSpace(space, space.ndim)
        >>> displacement_field = disp_field_space.element([[0, 0, 0, 0.02, 0]])
        >>> op = LinDeformFixedDispAdj(displacement_field)
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> op(template)
        uniform_discr(0.0, 1.0, 5).element([0.0, 0.0, 0.95122942450071402,
            0.0, 0.0])
        """
        if not isinstance(displacement.space, odl.ProductSpace):
            raise TypeError('`displacement` {!r} not an element'
                            'of `ProductSpace`'.format(displacement))

        domain_space = displacement[0].space
        range_space = domain_space

        Operator.__init__(self, domain_space, range_space, linear=True)

        self.displacement = displacement

    def _call(self, template):
        """Implement ``self(template)```.

        Parameters
        ----------
        template : `DiscreteLpVector`
            Given template that is to be deformed by the fixed
            displacement field.
        """
        div_op = odl.Divergence(range=template.space, method='central',
                                padding_method='symmetric')
        jacobian_det = np.exp(-div_op(self.displacement))
        return jacobian_det * template.space.element(
            linear_deform(template, -self.displacement))


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
