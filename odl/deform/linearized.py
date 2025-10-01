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
from odl.discr.discr_space import DiscretizedSpaceElement
from odl.discr.discr_utils import _normalize_interp, per_axis_interpolator
from odl.operator import Operator, PointwiseInner
from odl.space import ProductSpace
from odl.space.pspace import ProductSpaceElement
from odl.util import indent, signature_string
from odl.array_API_support import exp, lookup_array_backend

__all__ = ('LinDeformFixedTempl', 'LinDeformFixedDisp', 'linear_deform')


def linear_deform(template, displacement, interp='linear', out=None):
    """Linearized deformation of a template with a displacement field.

    The function maps a given template ``I`` and a given displacement
    field ``v`` to the new function ``x --> I(x + v(x))``.

    Parameters
    ----------
    template : `DiscretizedSpaceElement`
        Template to be deformed by a displacement field.
    displacement : element of power space of ``template.space``
        Vector field (displacement field) used to deform the
        template.
    interp : str or sequence of str
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
    Create a simple 1D template to initialize the operator and
    apply it to a displacement field. Where the displacement is zero,
    the output value is the same as the input value.
    In the 4-th point, the value is taken from 0.2 (one cell) to the
    left, i.e. 1.0.

    >>> space = odl.uniform_discr(0, 1, 5)
    >>> disp_field_space = space.tangent_bundle
    >>> template = space.element([0, 0, 1, 0, 0])
    >>> displacement_field = disp_field_space.element([[0, 0, 0, -0.2, 0]])
    >>> linear_deform(template, displacement_field, interp='nearest')
    array([ 0.,  0.,  1.,  1.,  0.])

    The result depends on the chosen interpolation. With 'linear'
    interpolation and an offset of half the distance between two
    points, 0.1, one gets the mean of the values.

    >>> displacement_field = disp_field_space.element([[0, 0, 0, -0.1, 0]])
    >>> linear_deform(template, displacement_field, interp='linear')
    array([ 0. ,  0. ,  1. ,  0.5,  0. ])
    """
    points = template.space.points()
    if isinstance(displacement, ProductSpaceElement):
        impl, device = displacement[0].impl, displacement[0].device
        backend = lookup_array_backend(impl)
    else:
        raise ValueError(f'{type(displacement)}')
    
    points = backend.array_constructor(points, device=device)
    
    for i, vi in enumerate(displacement):
        points[:, i] += vi.asarray().ravel()
    templ_interpolator = per_axis_interpolator(
        template, coord_vecs=template.space.grid.coord_vectors, interp=interp
    )

    values = templ_interpolator(points.T, out=out)
    return values.reshape(template.space.shape)


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

    def __init__(self, template, domain=None, interp='linear'):
        """Initialize a new instance.

        Parameters
        ----------
        template : `DiscretizedSpaceElement`
            Fixed template that is to be deformed.
        domain : power space of `DiscretizedSpace`, optional
            The space of all allowed coordinates in the deformation.
            A `ProductSpace` of ``template.ndim`` copies of a function-space.
            It must fulfill
            ``domain[0].partition == template.space.partition``, so
            this option is useful mainly when using different interpolations
            in displacement and template.

            Default: ``template.space.real_space.tangent_bundle``

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
        Create a simple 1D template to initialize the operator and
        apply it to a displacement field. Where the displacement is zero,
        the output value is the same as the input value.
        In the 4-th point, the value is taken from 0.2 (one cell) to the
        left, i.e. 1.0.

        >>> space = odl.uniform_discr(0, 1, 5)
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> op = LinDeformFixedTempl(template, interp='nearest')
        >>> disp_field = [[0, 0, 0, -0.2, 0]]
        >>> print(op(disp_field))
        [ 0.,  0.,  1.,  1.,  0.]

        The result depends on the chosen interpolation. With 'linear'
        interpolation and an offset of half the distance between two
        points, 0.1, one gets the mean of the values.

        >>> op = LinDeformFixedTempl(template, interp='linear')
        >>> disp_field = [[0, 0, 0, -0.1, 0]]
        >>> print(op(disp_field))
        [ 0. ,  0. ,  1. ,  0.5,  0. ]
        """
        if not isinstance(template, DiscretizedSpaceElement):
            raise TypeError(
                '`template` must be a `DiscretizedSpaceElement, got {!r}`'
                ''.format(template)
            )
        self.__template = template

        if domain is None:
            domain = self.template.space.real_space.tangent_bundle
        else:
            if not isinstance(domain, ProductSpace):
                # TODO: allow non-product spaces in the 1D case
                raise TypeError('`domain` must be a `ProductSpace` '
                                'instance, got {!r}'.format(domain))
            if not domain.is_power_space:
                raise TypeError('`domain` must be a power space, '
                                'got {!r}'.format(domain))
            if not isinstance(domain[0], DiscretizedSpace):
                raise TypeError('`domain[0]` must be a `DiscretizedSpace` '
                                'instance, got {!r}'.format(domain[0]))

            if template.space.partition != domain[0].partition:
                raise ValueError(
                    '`template.space.partition` not equal to `coord_space`s '
                    'partiton ({!r} != {!r})'
                    ''.format(template.space.partition, domain[0].partition))

        super(LinDeformFixedTempl, self).__init__(
            domain=domain, range=template.space, linear=False)

        self.__interp_byaxis = _normalize_interp(interp, template.space.ndim)

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
        return linear_deform(self.template, displacement, self.interp, out)

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
            raise NotImplementedError('derivative not implemented for complex '
                                      'spaces.')

        displacement = self.domain.element(displacement)

        # TODO: allow users to select what method to use here.
        grad = Gradient(domain=self.range, method='central',
                        pad_mode='symmetric')
        grad_templ = grad(self.template)
        def_grad = self.domain.element(
            [linear_deform(gf, displacement, self.interp) for gf in grad_templ]
        )

        return PointwiseInner(self.domain, def_grad)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.template]
        optargs = [
            ('domain', self.domain, self.template.space.tangent_bundle),
            ('interp', self.interp, 'linear'),
        ]
        inner_str = signature_string(posargs, optargs, mod='!r', sep=',\n')
        return '{}(\n{}\n)'.format(self.__class__.__name__, indent(inner_str))


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

    def __init__(self, displacement, templ_space=None, interp='linear'):
        """Initialize a new instance.

        Parameters
        ----------
        displacement : element of a power space of `DiscretizedSpace`
            Fixed displacement field used in the deformation.
        templ_space : `DiscretizedSpace`, optional
            Template space on which this operator is applied, i.e. the
            operator domain and range. It must fulfill
            ``templ_space[0].partition == displacement.space.partition``, so
            this option is useful mainly for support of complex spaces and if
            different interpolations should be used for displacement and
            template.

            Default: ``displacement.space[0]``

        interp : str or sequence of str
            Interpolation type that should be used to sample the template on
            the deformed grid. A single value applies to all axes, and a
            sequence gives the interpolation scheme per axis.

            Supported values: ``'nearest'``, ``'linear'``

        Examples
        --------
        Create a simple 1D template to initialize the operator and
        apply it to a displacement field. Where the displacement is zero,
        the output value is the same as the input value.
        In the 4-th point, the value is taken from 0.2 (one cell) to the
        left, i.e. 1.0.

        >>> space = odl.uniform_discr(0, 1, 5)
        >>> disp_field = space.tangent_bundle.element([[0, 0, 0, -0.2, 0]])
        >>> op = odl.deform.LinDeformFixedDisp(disp_field, interp='nearest')
        >>> template = [0, 0, 1, 0, 0]
        >>> print(op([0, 0, 1, 0, 0]))
        [ 0.,  0.,  1.,  1.,  0.]

        The result depends on the chosen interpolation. With 'linear'
        interpolation and an offset of half the distance between two
        points, 0.1, one gets the mean of the values.

        >>> disp_field = space.tangent_bundle.element([[0, 0, 0, -0.1, 0]])
        >>> op = odl.deform.LinDeformFixedDisp(disp_field, interp='linear')
        >>> template = [0, 0, 1, 0, 0]
        >>> print(op(template))
        [ 0. ,  0. ,  1. ,  0.5,  0. ]
        """
        if not isinstance(displacement, ProductSpaceElement):
            raise TypeError(
                '`displacement` must be a `ProductSpaceElement`, got {!r}'
                ''.format(displacement)
            )

        if not displacement.space.is_power_space:
            raise ValueError(
                '`displacement.space` must be a power space, got {!r}'
                ''.format(displacement.space)
            )
        if not isinstance(displacement.space[0], DiscretizedSpace):
            raise ValueError(
                '`displacement.space[0]` must be a `DiscretizedSpace`, '
                'got {!r}'.format(displacement.space[0]))

        self.__displacement = displacement

        if templ_space is None:
            templ_space = displacement.space[0]
        else:
            if not isinstance(templ_space, DiscretizedSpace):
                raise TypeError('`templ_space` must be a `DiscretizedSpace` '
                                'instance, got {!r}'.format(templ_space))
            if templ_space.partition != displacement.space[0].partition:
                raise ValueError(
                    '`templ_space.partition` not equal to `displacement`s '
                    'partiton ({!r} != {!r})'
                    ''.format(templ_space.partition,
                              displacement.space[0].partition)
                )

        super(LinDeformFixedDisp, self).__init__(
            domain=templ_space, range=templ_space, linear=True)

        self.__interp_byaxis = _normalize_interp(interp, templ_space.ndim)

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
        return linear_deform(template, self.displacement, self.interp, out)

    @property
    def inverse(self):
        """Inverse deformation using ``-v`` as displacement.

        Note that this implementation uses an approximation that is only
        valid for small displacements.
        """
        return LinDeformFixedDisp(
            -self.displacement, templ_space=self.domain, interp=self.interp
        )

    @property
    def adjoint(self):
        """Adjoint of the linear operator.

        Note that this implementation uses an approximation that is only
        valid for small displacements.
        """
        # TODO allow users to select what method to use here.
        div_op = Divergence(domain=self.displacement.space, method='forward',
                            pad_mode='symmetric')
        jacobian_det = self.domain.element(exp(-div_op(self.displacement)))

        return jacobian_det * self.inverse

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.displacement]
        optargs = [
            ('templ_space', self.domain, self.displacement.space[0]),
            ('interp', self.interp, 'linear'),
        ]
        inner_str = signature_string(posargs, optargs, mod='!r', sep=',\n')
        return '{}(\n{}\n)'.format(self.__class__.__name__, indent(inner_str))


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
