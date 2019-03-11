# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Operators and functions for linearized deformation."""

from __future__ import absolute_import, division, print_function

import numpy as np

from odl.discr import DiscretizedSpace, Divergence, Gradient
from odl.discr.discr_utils import _normalize_interp, per_axis_interpolator
from odl.operator import Operator, PointwiseInner
from odl.util import repr_string, signature_string_parts

__all__ = ('LinDeformFixedTempl', 'LinDeformFixedDisp', 'linear_deform')


def linear_deform(space, template, displacement, interp='linear', out=None):
    """Linearized deformation of a template with a displacement field.

    The function maps a given template ``I`` and a given displacement
    field ``v`` to the new function ``x --> I(x + v(x))``.

    Parameters
    ----------
    space : `DiscretizedSpace`
        Function space in which the deformation should be performed.
    template : `array-like` or callable
        Template to be deformed by a displacement field. Must be castable to
        an element of ``space``.
    displacement : `array-like` or callable
        Vector field (displacement field) used to deform the template.
        Must be castable to an element of ``space ** space.ndim``.
    interp : str or sequence of str, optional
        Interpolation type that should be used to sample the template on
        the deformed grid. A single value applies to all axes, and a
        sequence gives the interpolation scheme per axis.

        Supported values: ``'nearest'``, ``'linear'``

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
    A simple displacement of one point can be achieved with a displacement
    field that is everywhere zero except in that point. For instance, take
    the value of the 4th point from ``0.2`` (one cell) to the left, i.e.,
    ``1.0``:

    >>> space = odl.uniform_discr(0, 1, 5)
    >>> disp_field_space = space.tangent_bundle
    >>> template = [0, 0, 1, 0, 0]
    >>> displacement_field = [[0, 0, 0, -0.2, 0]]
    >>> linear_deform(space, template, displacement_field, interp='nearest')
    array([ 0.,  0.,  1.,  1.,  0.])

    The result depends on the chosen interpolation. With ``'linear'``
    interpolation and an offset of half the distance between two points,
    ``0.1``, one gets the mean of the values:

    >>> displacement_field = [[0, 0, 0, -0.1, 0]]
    >>> linear_deform(space, template, displacement_field, interp='linear')
    array([ 0. ,  0. ,  1. ,  0.5,  0. ])

    We can also use callables directly, both as template and as deformation
    field. For instance, we can flip a function by using the displacement
    ``v(x) = -x + (1 - x)``:

    >>> space.element(lambda x: x)
    array([ 0.1,  0.3,  0.5,  0.7,  0.9])
    >>> linear_deform(space, lambda x: x, [lambda x: 1 - 2 * x])
    array([ 0.9,  0.7,  0.5,  0.3,  0.1])
    """
    if not isinstance(space, DiscretizedSpace):
        raise TypeError(
            '`space` must be a `DiscretizedSpace`, got {!r}'.format(space)
        )
    template = space.element(template)
    displacement = space.tangent_bundle.element(displacement)
    points = space.points()
    for i, vi in enumerate(displacement):
        points[:, i] += vi.ravel()
    templ_interpolator = per_axis_interpolator(
        template, coord_vecs=space.grid.coord_vectors, interp=interp
    )
    values = templ_interpolator(points.T, out=out)
    return values.reshape(space.shape)


class LinDeformFixedTempl(Operator):

    r"""Deformation operator with fixed template acting on displacement fields.

    The operator has a fixed template ``I`` and maps a displacement
    field ``v`` to the new function ``x --> I(x + v(x))``.

    See Also
    --------
    LinDeformFixedDisp : Deformation with a fixed displacement.

    Notes
    -----
    For :math:`\Omega \subset \mathbb{R}^d`, we take :math:`X = L^p(\Omega)`
    to be the template space, i.e. :math:`I \in X`. Then the vector field
    space is identified with :math:`V := X^d`. Hence the deformation operator
    with fixed template maps :math:`V` into :math:`X`:

    .. math::
        W_I : V \to X, \quad W_I(v) := I(\cdot + v(\cdot)),

    i.e., :math:`W_I(v)(x) = I(x + v(x))`.

    Note that this operator is non-linear. Its derivative at :math:`v` is
    an operator that maps :math:`V` into :math:`X`:

    .. math::
        W_I'(v) : V \to X, \quad W_I'(v)(u) =
        \big< \nabla I(\cdot + v(\cdot)), u \big>_{\mathbb{R}^d},

    i.e., :math:`W_I'(v)(u)(x) = \nabla I(x + v(x))^T u(x)`,

    which is to be understood as a point-wise inner product, resulting
    in a function in :math:`X`. And the adjoint of the preceding derivative
    is also an operator that maps :math:`X` into :math:`V`:

    .. math::
        W_I'(v)^* : X \to V, \quad W_I'(v)^*(J) =
        J \, \nabla I(\cdot + v(\cdot)),

    i.e., :math:`W_I'(v)^*(J)(x) = J(x) \, \nabla I(x + v(x))`.
    """

    def __init__(self, range, template, interp='linear'):
        """Initialize a new instance.

        Parameters
        ----------
        range : `DiscretizedSpace`
            Template space to which the operator maps.
        template : `array-like` or callable
            Fixed template that is to be deformed. Must be castable to an
            element of ``range``.
        interp : str or sequence of str
            Interpolation type that should be used to sample the template on
            the deformed grid. A single value applies to all axes, and a
            sequence gives the interpolation scheme per axis.

            Supported values: ``'nearest'``, ``'linear'``

            .. warning::
                Choosing ``'nearest'`` interpolation results in a formally
                non-differentiable operator since the gradient of the
                template is not well-defined. If the operator derivative
                is to be used, a differentiable interpolation scheme (e.g.,
                ``'linear'``) should be chosen.

        Examples
        --------
        A simple displacement of one point can be achieved with a displacement
        field that is everywhere zero except in that point. For instance, take
        the value of the 4th point from ``0.2`` (one cell) to the left, i.e.,
        ``1.0``:

        >>> space = odl.uniform_discr(0, 1, 5)
        >>> template = [0, 0, 1, 0, 0]
        >>> op = odl.deform.LinDeformFixedTempl(
        ...     space, template, interp='nearest'
        ... )
        >>> disp_field = [[0, 0, 0, -0.2, 0]]
        >>> op(disp_field)
        array([ 0.,  0.,  1.,  1.,  0.])

        The result depends on the chosen interpolation. With ``'linear'``
        interpolation and an offset of half the distance between two points,
        ``0.1``, one gets the mean of the values:

        >>> op = odl.deform.LinDeformFixedTempl(
        ...     space, template, interp='linear'
        ... )
        >>> disp_field = [[0, 0, 0, -0.1, 0]]
        >>> op(disp_field)
        array([ 0. ,  0. ,  1. ,  0.5,  0. ])
        """
        if not isinstance(range, DiscretizedSpace):
            raise TypeError(
                '`range` must be a `DiscretizedSpace`, got {!r}'.format(range)
            )

        super(LinDeformFixedTempl, self).__init__(
            domain=range.tangent_bundle, range=range, linear=False
        )

        self.__template = range.element(template)
        self.__interp_byaxis = _normalize_interp(interp, range.ndim)

    @property
    def template(self):
        """Fixed template of this deformation operator."""
        return self.__template

    @property
    def interp_byaxis(self):
        """Tuple of per-axis interpolation schemes."""
        return self.__interp_byaxis

    @property
    def interp(self):
        """Interpolation scheme or tuple of per-axis interpolation schemes."""
        if (
            len(self.interp_byaxis) != 0
            and all(s == self.interp_byaxis[0] for s in self.interp_byaxis[1:])
        ):
            return self.interp_byaxis[0]
        else:
            return self.interp_byaxis

    def _call(self, displacement, out=None):
        """Implementation of ``self(displacement[, out])``."""
        return linear_deform(
            self.range, self.template, displacement, self.interp, out
        )

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
        if not self.range.is_real:
            raise NotImplementedError(
                'derivative not implemented for complex spaces.'
            )

        displ = self.domain.element(displacement)

        # TODO: allow users to select what method to use here.
        grad = Gradient(
            domain=self.range, method='central', pad_mode='symmetric'
        )
        grad_templ = grad(self.template)
        def_grad = [
            linear_deform(self.range, gf, displ, self.interp)
            for gf in grad_templ
        ]

        return PointwiseInner(self.domain, def_grad)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.range, self.template]
        optargs = [('interp', self.interp, 'linear')]
        inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(
            self.__class__.__name__, inner_parts, allow_mixed_seps=False
        )


class LinDeformFixedDisp(Operator):

    r"""Deformation operator with fixed displacement acting on templates.

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
    displacement field :math:`v \in V` maps :math:`X` into :math:`X`:

    .. math::
        W_v : X \to X, \quad W_v(I) := I(\cdot + v(\cdot)),

    i.e., :math:`W_v(I)(x) = I(x + v(x))`.

    This operator is linear, so its derivative is itself, but it may not be
    bounded and may thus not have a formal adjoint. For "small" :math:`v`,
    though, one can approximate the adjoint by

    .. math::
        W_v^*(I) \approx \exp(-\mathrm{div}\, v) \, I(\cdot - v(\cdot)),

    i.e., :math:`W_v^*(I)(x) \approx \exp(-\mathrm{div}\,v(x))\, I(x - v(x))`.
    """

    def __init__(self, domain, displacement, interp='linear'):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscretizedSpace`
            Template space from which the operator takes inputs.
        displacement : `array-like` or callable
            Fixed displacement field used in the deformation. Must be castable
            to an element of ``domain.tangent_bundle``.
        interp : str or sequence of str
            Interpolation type that should be used to sample the template on
            the deformed grid. A single value applies to all axes, and a
            sequence gives the interpolation scheme per axis.

            Supported values: ``'nearest'``, ``'linear'``

        Examples
        --------
        A simple displacement of one point can be achieved with a displacement
        field that is everywhere zero except in that point. For instance, take
        the value of the 4th point from ``0.2`` (one cell) to the left, i.e.,
        ``1.0``:

        >>> space = odl.uniform_discr(0, 1, 5)
        >>> disp_field = [[0, 0, 0, -0.2, 0]]
        >>> op = odl.deform.LinDeformFixedDisp(
        ...     space, disp_field, interp='nearest'
        ... )
        >>> template = [0, 0, 1, 0, 0]
        >>> op(template)
        array([ 0.,  0.,  1.,  1.,  0.])

        The result depends on the chosen interpolation. With ``'linear'``
        interpolation and an offset of half the distance between two points,
        ``0.1``, one gets the mean of the values:

        >>> space = odl.uniform_discr(0, 1, 5)
        >>> disp_field = [[0, 0, 0, -0.1, 0]]
        >>> op = odl.deform.LinDeformFixedDisp(
        ...     space, disp_field, interp='linear'
        ... )
        >>> template = [0, 0, 1, 0, 0]
        >>> op(template)
        array([ 0. ,  0. ,  1. ,  0.5,  0. ])
        """
        if not isinstance(domain, DiscretizedSpace):
            raise TypeError(
                '`domain` must be a `DiscretizedSpace`, got {!r}'.format(domain)
            )

        super(LinDeformFixedDisp, self).__init__(
            domain=domain, range=domain, linear=True
        )

        try:
            self.__displacement = (self.domain.tangent_bundle).element(
                displacement
            )
        except (ValueError, TypeError):
            self.__displacement = (self.domain.tangent_bundle).element(
                [displacement]
            )
        self.__interp_byaxis = _normalize_interp(interp, domain.ndim)

    @property
    def interp_byaxis(self):
        """Tuple of per-axis interpolation schemes."""
        return self.__interp_byaxis

    @property
    def interp(self):
        """Interpolation scheme or tuple of per-axis interpolation schemes."""
        if (
            len(self.interp_byaxis) != 0
            and all(s == self.interp_byaxis[0] for s in self.interp_byaxis[1:])
        ):
            return self.interp_byaxis[0]
        else:
            return self.interp_byaxis

    @property
    def displacement(self):
        """Fixed displacement field of this deformation operator."""
        return self.__displacement

    def _call(self, template, out=None):
        """Implementation of ``self(template[, out])``."""
        return linear_deform(
            self.domain, template, self.displacement, self.interp, out
        )

    @property
    def inverse(self):
        """Inverse deformation using ``-v`` as displacement.

        Note that this implementation uses an approximation that is only
        valid for small displacements.
        """
        return LinDeformFixedDisp(
            self.domain, -self.displacement, interp=self.interp
        )

    @property
    def adjoint(self):
        """Adjoint of the linear operator.

        Note that this implementation uses an approximation that is only
        valid for small displacements.
        """
        # TODO allow users to select what method to use here.
        div_op = Divergence(
            domain=self.domain.tangent_bundle,
            method='forward',
            pad_mode='symmetric',
        )
        jacobian_det = self.domain.element(np.exp(-div_op(self.displacement)))

        return jacobian_det * self.inverse

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.domain, self.displacement]
        optargs = [('interp', self.interp, 'linear')]
        inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(
            self.__class__.__name__, inner_parts, allow_mixed_seps=False
        )


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
