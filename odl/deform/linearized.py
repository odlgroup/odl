# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Operators and functions for linearized deformation."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.discr import DiscreteLp, Gradient, Divergence
from odl.operator import Operator, PointwiseInner
from odl.space import ProductSpace


__all__ = ('LinDeformFixedTempl', 'LinDeformFixedDisp')


def _linear_deform(template, displacement, out=None):
    """Linearized deformation of a template with a displacement field.

    The function maps a given template ``I`` and a given displacement
    field ``v`` to the new function ``x --> I(x + v(x))``.

    Parameters
    ----------
    template : `DiscreteLpElement`
        Template to be deformed by a displacement field.
    displacement : element of power space of ``template.space``
        Vector field (displacement field) used to deform the
        template.
    out : `numpy.ndarray`, optional
        Array to which the function values of the deformed template
        are written. It must have the same shape as ``template`` and
        a data type compatible with ``template.dtype``.

    Returns
    -------
    deformed_template : `numpy.ndarray`
        Function values of the deformed template. If ``out`` was given,
        the returned object is a reference to it.

    Examples
    --------
    Create a simple 1D template to initialize the operator and
    apply it to a displacement field. Where the displacement is zero,
    the output value is the same as the input value.
    In the 4-th point, the value is taken from 0.2 (one cell) to the
    left, i.e. 1.0.

    >>> space = odl.uniform_discr(0, 1, 5)
    >>> disp_field_space = space.tangent_bundle
    >>> template = space.element([0, 0, 1, 0, 0])
    >>> displacement_field = disp_field_space.element([[0, 0, 0, -0.2, 0]])
    >>> _linear_deform(template, displacement_field)
    array([ 0.,  0.,  1.,  1.,  0.])

    The result depends on the chosen interpolation. With 'linear'
    interpolation and an offset equal to half the distance between two
    points, 0.1, one gets the mean of the values.

    >>> space = odl.uniform_discr(0, 1, 5, interp='linear')
    >>> disp_field_space = space.tangent_bundle
    >>> template = space.element([0, 0, 1, 0, 0])
    >>> displacement_field = disp_field_space.element([[0, 0, 0, -0.1, 0]])
    >>> _linear_deform(template, displacement_field)
    array([ 0. ,  0. ,  1. ,  0.5,  0. ])
    """
    image_pts = template.space.points()
    for i, vi in enumerate(displacement):
        image_pts[:, i] += vi.asarray().ravel(order=vi.space.view_order)
    values = template.interpolation(image_pts.T, out=out, bounds_check=False)
    return values.reshape(displacement.space[0].shape,
                          order=displacement.space[0].view_order)


class LinDeformFixedTempl(Operator):

    """Deformation operator with fixed template acting on displacement fields.

    The operator has a fixed template ``I`` and maps a displacement
    field ``v`` to the new function ``x --> I(x + v(x))``.

    See Also
    --------
    LinDeformFixedDisp : Deformation with a fixed displacement.

    Notes
    -----
    For :math:`\Omega \subset \mathbb{R}^d`, we take :math:`X = L^p(\Omega)`
    to be the template space, i.e. :math:`I \\in X`. Then the vector field
    space is identified with :math:`V := X^d`. Hence the deformation operator
    with fixed template maps :math:`V` into :math:`X`:

    .. math::
        W_I : V \\to X, \quad W_I(v) := I(\cdot + v(\cdot)),

    i.e., :math:`W_I(v)(x) = I(x + v(x))`.

    Note that this operator is non-linear. Its derivative at :math:`v` is
    an operator that maps :math:`V` into :math:`X`:

    .. math::
        W_I'(v) : V \\to X, \quad W_I'(v)(u) =
        \\big< \\nabla I(\cdot + v(\cdot)), u \\big>_{\mathbb{R}^d},

    i.e., :math:`W_I'(v)(u)(x) = \\nabla I(x + v(x))^T u(x)`,

    which is to be understood as a point-wise inner product, resulting
    in a function in :math:`X`. And the adjoint of the preceding derivative
    is also an operator that maps :math:`X` into :math:`V`:

    .. math::
        W_I'(v)^* : X \\to V, \quad W_I'(v)^*(J) =
        J \, \\nabla I(\cdot + v(\cdot)),

    i.e., :math:`W_I'(v)^*(J)(x) = J(x) \, \\nabla I(x + v(x))`.
    """

    def __init__(self, template, domain=None):
        """Initialize a new instance.

        Parameters
        ----------
        template : `DiscreteLpElement`
            Fixed template that is to be deformed.
        domain : power space of `DiscreteLp`, optional
            The space of all allowed coordinates in the deformation.
            A `ProductSpace` of ``template.ndim`` copies of a function-space.
            It must fulfill
            ``domain[0].partition == template.space.partition``, so
            this option is useful mainly when using different interpolations
            in displacement and template.
            Default: ``template.space.real_space.tangent_bundle``

        Examples
        --------
        Create a simple 1D template to initialize the operator and
        apply it to a displacement field. Where the displacement is zero,
        the output value is the same as the input value.
        In the 4-th point, the value is taken from 0.2 (one cell) to the
        left, i.e. 1.0.

        >>> space = odl.uniform_discr(0, 1, 5, interp='nearest')
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> op = LinDeformFixedTempl(template)
        >>> disp_field = [[0, 0, 0, -0.2, 0]]
        >>> print(op(disp_field))
        [0.0, 0.0, 1.0, 1.0, 0.0]

        The result depends on the chosen interpolation. With 'linear'
        interpolation and an offset equal to half the distance between two
        points, 0.1, one gets the mean of the values.

        >>> space = odl.uniform_discr(0, 1, 5, interp='linear')
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> op = LinDeformFixedTempl(template)
        >>> disp_field = [[0, 0, 0, -0.1, 0]]
        >>> print(op(disp_field))
        [0.0, 0.0, 1.0, 0.5, 0.0]
        """
        space = getattr(template, 'space', None)
        if not isinstance(space, DiscreteLp):
            raise TypeError('`template.space` must be a `DiscreteLp`'
                            'instance, got {!r}'.format(space))
        self.__template = template

        if domain is None:
            domain = self.template.space.real_space.tangent_bundle
        else:
            if not isinstance(domain, ProductSpace):
                raise TypeError('`domain` must be a `ProductSpace` '
                                'instance, got {!r}'.format(domain))
            if not domain.is_power_space:
                raise TypeError('`domain` must be a power space, '
                                'got {!r}'.format(domain))
            if not isinstance(domain[0], DiscreteLp):
                raise TypeError('`domain[0]` must be a `DiscreteLp` '
                                'instance, got {!r}'.format(domain[0]))

            if template.space.partition != domain[0].partition:
                raise ValueError(
                    '`template.space.partition` not equal to `coord_space`s '
                    'partiton ({!r} != {!r})'
                    ''.format(template.space.partition, domain[0].partition))

        super().__init__(domain=domain,
                         range=self.template.space,
                         linear=False)

    @property
    def template(self):
        """Fixed template of this deformation operator."""
        return self.__template

    def _call(self, displacement, out=None):
        """Implementation of ``self(displacement[, out])``."""
        return _linear_deform(self.template, displacement, out)

    def derivative(self, displacement):
        """Derivative of the operator at ``displacement``.

        Parameters
        ----------
        displacement : `domain` `element-like`
            Point at which the derivative is computed.

        Returns
        -------
        derivative : `PointwiseInner`
            The derivative evaluated at ``displacement``.
        """
        # To implement the complex case we need to be able to embed the real
        # vector field space into the range of the gradient. Issue #59.
        if not self.range.is_real_space:
            raise NotImplementedError('derivative not implemented for complex '
                                      'spaces.')

        displacement = self.domain.element(displacement)

        # TODO: allow users to select what method to use here.
        grad = Gradient(domain=self.range, method='central',
                        pad_mode='symmetric')
        grad_templ = grad(self.template)
        def_grad = self.domain.element(
            [_linear_deform(gf, displacement) for gf in grad_templ])

        return PointwiseInner(self.domain, def_grad)

    def __repr__(self):
        """Return ``repr(self)``."""
        arg_reprs = [repr(self.template)]
        if self.domain != self.__displacement.space[0]:
            arg_reprs.append('domain={!r}'.format(self.domain))
        arg_str = ', '.join(arg_reprs)

        return '{}({})'.format(self.__class__.__name__, arg_str)


class LinDeformFixedDisp(Operator):

    """Deformation operator with fixed displacement acting on templates.

    The operator has a fixed displacement field ``v`` and maps a template
    ``I`` to the new function ``x --> I(x + v(x))``.

    See Also
    --------
    LinDeformFixedTempl : Deformation with a fixed template.

    Notes
    -----
    For :math:`\Omega \subset \mathbb{R}^d`, we take :math:`V := X^d`
    to be the space of displacement fields, where :math:`X = L^p(\Omega)`
    is the template space. Hence the deformation operator with the fixed
    displacement field :math:`v \\in V` maps :math:`X` into :math:`X`:

    .. math::
        W_v : X \\to X, \quad W_v(I) := I(\cdot + v(\cdot)),

    i.e., :math:`W_v(I)(x) = I(x + v(x))`.

    This operator is linear, so its derivative is itself, but it may not be
    bounded and may thus not have a formal adjoint. For "small" :math:`v`,
    though, one can approximate the adjoint by

    .. math::
        W_v^*(I) \\approx \exp(-\mathrm{div}\, v) \, I(\cdot - v(\cdot)),

    i.e., :math:`W_v^*(I)(x) \\approx \exp(-\mathrm{div}\,v(x))\, I(x - v(x))`.
    """

    def __init__(self, displacement, templ_space=None):
        """Initialize a new instance.

        Parameters
        ----------
        displacement : element of a power space of `DiscreteLp`
            Fixed displacement field used in the deformation.
        templ_space : `DiscreteLp`, optional
            Template space on which this operator is applied, i.e. the
            operator domain and range. It must fulfill
            ``templ_space[0].partition == displacement.space.partition``, so
            this option is useful mainly for support of complex spaces and if
            different interpolations should be used for displacement and
            template.
            Default: ``displacement.space[0]``

        Examples
        --------
        Create a simple 1D template to initialize the operator and
        apply it to a displacement field. Where the displacement is zero,
        the output value is the same as the input value.
        In the 4-th point, the value is taken from 0.2 (one cell) to the
        left, i.e. 1.0.

        >>> space = odl.uniform_discr(0, 1, 5)
        >>> disp_field = space.tangent_bundle.element([[0, 0, 0, -0.2, 0]])
        >>> op = LinDeformFixedDisp(disp_field)
        >>> template = [0, 0, 1, 0, 0]
        >>> print(op([0, 0, 1, 0, 0]))
        [0.0, 0.0, 1.0, 1.0, 0.0]

        The result depends on the chosen interpolation. With 'linear'
        interpolation and an offset equal to half the distance between two
        points, 0.1, one gets the mean of the values.

        >>> space = odl.uniform_discr(0, 1, 5, interp='linear')
        >>> disp_field = space.tangent_bundle.element([[0, 0, 0, -0.1, 0]])
        >>> op = LinDeformFixedDisp(disp_field)
        >>> template = [0, 0, 1, 0, 0]
        >>> print(op(template))
        [0.0, 0.0, 1.0, 0.5, 0.0]
        """
        space = getattr(displacement, 'space', None)
        if not isinstance(space, ProductSpace):
            raise TypeError('`displacement.space` must be a `ProductSpace` '
                            'instance, got {!r}'.format(space))
        if not space.is_power_space:
            raise TypeError('`displacement.space` must be a power space, '
                            'got {!r}'.format(space))
        if not isinstance(space[0], DiscreteLp):
            raise TypeError('`displacement.space[0]` must be a `DiscreteLp` '
                            'instance, got {!r}'.format(space[0]))

        if templ_space is None:
            templ_space = displacement.space[0]
        else:
            if not isinstance(templ_space, DiscreteLp):
                raise TypeError('`templ_space` must be a `DiscreteLp` '
                                'instance, got {!r}'.format(templ_space))
            if templ_space.partition != space[0].partition:
                raise ValueError(
                    '`templ_space.partition` not equal to `displacement`s '
                    'partiton ({!r} != {!r})'
                    ''.format(templ_space.partition, space[0].partition))

        self.__displacement = displacement
        super().__init__(domain=templ_space, range=templ_space, linear=True)

    @property
    def displacement(self):
        """Fixed displacement field of this deformation operator."""
        return self.__displacement

    def _call(self, template, out=None):
        """Implementation of ``self(template[, out])``."""
        return _linear_deform(template, self.displacement, out)

    @property
    def inverse(self):
        """Inverse deformation using ``-v`` as displacement.

        Note that this implementation uses an approximation that is only
        valid for small displacements.
        """
        return LinDeformFixedDisp(-self.displacement, templ_space=self.domain)

    @property
    def adjoint(self):
        """Adjoint of the linear operator.

        Note that this implementation uses an approximation that is only
        valid for small displacements.
        """
        # TODO allow users to select what method to use here.
        div_op = Divergence(domain=self.displacement.space, method='forward',
                            pad_mode='symmetric')
        jacobian_det = self.domain.element(
            np.exp(-div_op(self.displacement)))

        return jacobian_det * self.inverse

    def __repr__(self):
        """Return ``repr(self)``."""
        arg_reprs = [repr(self.displacement)]
        if self.domain != self.__displacement.space[0]:
            arg_reprs.append('templ_space={!r}'.format(self.domain))
        arg_str = ', '.join(arg_reprs)

        return '{}({})'.format(self.__class__.__name__, arg_str)

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
