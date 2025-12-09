# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Operators defined for tensor fields."""


import numpy as np
from math import prod

from odl.core.discr.discr_space import DiscretizedSpace
from odl.core.operator.tensor_ops import PointwiseTensorFieldOperator
from odl.core.space import ProductSpace
from odl.core.util import indent, signature_string, writable_array
from odl.core.array_API_support import asarray, get_array_and_backend

__all__ = ('PartialDerivative', 'Gradient', 'Divergence', 'Laplacian')

_SUPPORTED_DIFF_METHODS = ('central', 'forward', 'backward')
_SUPPORTED_PAD_MODES = ('constant',
                        'symmetric', 'symmetric_adjoint',
                        'periodic',
                        'order0', 'order0_adjoint',
                        'order1', 'order1_adjoint',
                        'order2', 'order2_adjoint')

_ADJ_METHOD = {'central': 'central',
               'forward': 'backward',
               'backward': 'forward'}

_ADJ_PADDING = {'constant': 'constant',
                'symmetric': 'symmetric_adjoint',
                'symmetric_adjoint': 'symmetric',
                'periodic': 'periodic',
                'order0': 'order0_adjoint',
                'order0_adjoint': 'order0',
                'order1': 'order1_adjoint',
                'order1_adjoint': 'order1',
                'order2': 'order2_adjoint',
                'order2_adjoint': 'order2'}


class PartialDerivative(PointwiseTensorFieldOperator):

    """Calculate the discrete partial derivative along a given axis.

    Calls helper function `finite_diff` to calculate finite difference.
    Preserves the shape of the underlying grid.
    """

    def __init__(self, domain, axis, range=None, method='forward',
                 pad_mode='constant', pad_const=0):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscretizedSpace`
            Space of elements on which the operator can act.
        axis : int
            Axis along which the partial derivative is evaluated.
        range : `DiscretizedSpace`, optional
            Space of elements to which the operator maps, must have the
            same shape as ``domain``.
            For the default ``None``, the range is the same as ``domain``.
        method : {'forward', 'backward', 'central'}, optional
            Finite difference method which is used in the interior of the
            domain of ``f``.
        pad_mode : string, optional
            The padding mode to use outside the domain.

            ``'constant'``: Fill with ``pad_const``.

            ``'symmetric'``: Reflect at the boundaries, not doubling the
            outmost values.

            ``'periodic'``: Fill in values from the other side, keeping
            the order.

            ``'order0'``: Extend constantly with the outmost values
            (ensures continuity).

            ``'order1'``: Extend with constant slope (ensures continuity of
            the first derivative). This requires at least 2 values along
            each axis where padding is applied.

            ``'order2'``: Extend with second order accuracy (ensures continuity
            of the second derivative). This requires at least 3 values along
            the ``axis`` where padding is applied.

        pad_const : float, optional
            For ``pad_mode == 'constant'``, ``f`` assumes
            ``pad_const`` for indices outside the domain of ``f``

        Examples
        --------
        >>> f = np.array([[ 0.,  1.,  2.,  3.,  4.],
        ...               [ 0.,  2.,  4.,  6.,  8.]])
        >>> discr = odl.uniform_discr([0, 0], [2, 1], f.shape)
        >>> par_deriv = PartialDerivative(discr, axis=0, pad_mode='order1')
        >>> par_deriv(f)
        uniform_discr([ 0.,  0.], [ 2.,  1.], (2, 5)).element(
            [[ 0.,  1.,  2.,  3.,  4.],
             [ 0.,  1.,  2.,  3.,  4.]]
        )
        """
        if not isinstance(domain, DiscretizedSpace):
            raise TypeError(f"`domain` {domain} is not a DiscretizedSpace instance")

        if range is None:
            range = domain

        # Method is affine if nonzero padding is given.
        linear = not (pad_mode == "constant" and pad_const != 0)
        super().__init__(domain, range, base_space=domain, linear=linear)
        self.axis = int(axis)
        self.dx = self.domain.cell_sides[axis]

        self.method, method_in = str(method).lower(), method
        if method not in _SUPPORTED_DIFF_METHODS:
            raise ValueError(f"`method` {method_in} not understood")

        self.pad_mode, pad_mode_in = str(pad_mode).lower(), pad_mode
        if pad_mode not in _SUPPORTED_PAD_MODES:
            raise ValueError(f"`pad_mode` {pad_mode_in} not understood")

        self.pad_const = self.domain.field.element(pad_const)

    def _call(self, x, out=None):
        """Calculate partial derivative of ``x``."""
        if out is None:
            out = self.range.element()

        # TODO: this pipes CUDA arrays through NumPy. Write native operator.
        with writable_array(out) as out_arr:
            finite_diff(x.asarray(), axis=self.axis, dx=self.dx,
                        method=self.method, pad_mode=self.pad_mode,
                        pad_const=self.pad_const, out=out_arr)
        return out

    def derivative(self, point=None):
        """Return the derivative operator.

        The partial derivative is usually linear, but in case the 'constant'
        ``pad_mode`` is used with nonzero ``pad_const``, the
        derivative is given by the derivative with 0 ``pad_const``.

        Parameters
        ----------
        point : `domain` `element-like`, optional
            The point to take the derivative in. Does not change the result
            since the operator is affine.
        """
        if self.pad_mode == 'constant' and self.pad_const != 0:
            return PartialDerivative(self.domain, self.axis, self.range,
                                     self.method, self.pad_mode, 0)
        else:
            return self

    @property
    def adjoint(self):
        """Return the adjoint operator."""
        if not self.is_linear:
            raise ValueError(
                f"operator with nonzero pad_const ({self.pad_const}) is not linear and has no adjoint"
            )

        return -PartialDerivative(self.range, self.axis, self.domain,
                                  _ADJ_METHOD[self.method],
                                  _ADJ_PADDING[self.pad_mode],
                                  self.pad_const)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.domain]
        optargs = [('axis', self.axis, None),
                   ('range', self.range, self.domain),
                   ('method', self.method, 'forward'),
                   ('pad_mode', self.pad_mode, 'constant'),
                   ('pad_const', self.pad_const, 0)]
        inner_str = signature_string(posargs, optargs,
                                     sep=',\n', mod=['!r', ''])
        return f"{self.__class__.__name__}(\n{indent(inner_str)}\n)"

    def __str__(self):
        """Return ``str(self)``."""
        dom_ran_str = "\n-->\n".join([repr(self.domain), repr(self.range)])
        return f"{self.__class__.__name__}:\n{indent(dom_ran_str)}"


class Gradient(PointwiseTensorFieldOperator):
    """Spatial gradient operator for `DiscretizedSpace` spaces.

    Calls helper function `finite_diff` to calculate each component of the
    resulting product space element. For the adjoint of the `Gradient`
    operator, zero padding is assumed to match the negative `Divergence`
    operator
    """

    def __init__(self, domain=None, range=None, method='forward',
                 pad_mode='constant', pad_const=0):
        """Initialize a new instance.

        Zero padding is assumed for the adjoint of the `Gradient`
        operator to match negative `Divergence` operator.

        Parameters
        ----------
        domain : `DiscretizedSpace`, optional
            Space of elements which the operator acts on.
            This is required if ``range`` is not given.
        range : power space of `DiscretizedSpace`, optional
            Space of elements to which the operator maps.
            This is required if ``domain`` is not given.
        method : {'forward', 'backward', 'central'}, optional
            Finite difference method to be used.
        pad_mode : string, optional
            The padding mode to use outside the domain.

            ``'constant'``: Fill with ``pad_const``.

            ``'symmetric'``: Reflect at the boundaries, not doubling the
            outmost values.

            ``'periodic'``: Fill in values from the other side, keeping
            the order.

            ``'order0'``: Extend constantly with the outmost values
            (ensures continuity).

            ``'order1'``: Extend with constant slope (ensures continuity of
            the first derivative). This requires at least 2 values along
            each axis where padding is applied.

            ``'order2'``: Extend with second order accuracy (ensures continuity
            of the second derivative). This requires at least 3 values along
            each axis.

        pad_const : float, optional
            For ``pad_mode == 'constant'``, ``f`` assumes
            ``pad_const`` for indices outside the domain of ``f``

        Examples
        --------
        Creating a Gradient operator:

        >>> dom = odl.uniform_discr([0, 0], [1, 1], (10, 20))
        >>> ran = odl.ProductSpace(dom, dom.ndim)  # 2-dimensional
        >>> grad_op = Gradient(dom)
        >>> grad_op.range == ran
        True
        >>> grad_op2 = Gradient(range=ran)
        >>> grad_op2.domain == dom
        True
        >>> grad_op3 = Gradient(domain=dom, range=ran)
        >>> grad_op3.domain == dom
        True
        >>> grad_op3.range == ran
        True

        Calling the operator:

        >>> data = np.array([[ 0., 1., 2., 3., 4.],
        ...                  [ 0., 2., 4., 6., 8.]])
        >>> discr = odl.uniform_discr([0, 0], [2, 5], data.shape)
        >>> f = discr.element(data)
        >>> grad = Gradient(discr)
        >>> grad_f = grad(f)
        >>> grad_f[0]
        uniform_discr([ 0.,  0.], [ 2.,  5.], (2, 5)).element(
            [[ 0.,  1.,  2.,  3.,  4.],
             [ 0., -2., -4., -6., -8.]]
        )
        >>> grad_f[1]
        uniform_discr([ 0.,  0.], [ 2.,  5.], (2, 5)).element(
            [[ 1.,  1.,  1.,  1., -4.],
             [ 2.,  2.,  2.,  2., -8.]]
        )

        Verify adjoint:

        >>> g = grad.range.element((data, data ** 2))
        >>> adj_g = grad.adjoint(g)
        >>> adj_g
        uniform_discr([ 0.,  0.], [ 2.,  5.], (2, 5)).element(
            [[ -0.,  -2.,  -5.,  -8., -11.],
             [ -0.,  -5., -14., -23., -32.]]
        )
        >>> g.inner(grad_f) / f.inner(adj_g)
        1.0
        """
        if domain is None and range is None:
            raise ValueError("either `domain` or `range` must be specified")

        if domain is None:
            try:
                domain = range[0]
            except TypeError:
                pass

        if range is None:
            range = ProductSpace(domain, domain.ndim)

        # Check range first since `domain` may end up to be `None` in
        # the case filtered out here (see above)
        if not isinstance(range, ProductSpace):
            raise TypeError(f"`range` {range} is not a `ProductSpace` instance")
        if not range.is_power_space:
            raise ValueError(f"`range` {range} is not a power space")

        if not isinstance(domain, DiscretizedSpace):
            raise TypeError(f"`domain` {domain} is not a `DiscretizedSpace` instance")

        if len(range) != domain.ndim:
            raise ValueError(
                f"`range` must be a power space of length n = {domain.ndim},with `n == domain.ndim`, got n = {len(range)} instead"
            )

        linear = not (pad_mode == "constant" and pad_const != 0)
        super().__init__(domain, range, base_space=domain, linear=linear)

        self.method, method_in = str(method).lower(), method
        if method not in _SUPPORTED_DIFF_METHODS:
            raise ValueError(f"`method` {method_in} not understood")

        self.pad_mode, pad_mode_in = str(pad_mode).lower(), pad_mode
        if pad_mode not in _SUPPORTED_PAD_MODES:
            raise ValueError(f"`pad_mode` {pad_mode_in} not understood")

        self.pad_const = domain.field.element(pad_const)

    def _call(self, x, out=None):
        """Calculate the spatial gradient of ``x``."""
        x_arr = x.asarray()
        ndim = self.domain.ndim
        dx = self.domain.cell_sides

        if out is None:
            return self.range.element([
                     finite_diff(x_arr, axis=axis, dx=dx[axis], method=self.method,
                            pad_mode=self.pad_mode,
                            pad_const=self.pad_const,
                            )
                     for axis in range(ndim)])
        else:
            for axis in range(ndim):
                with writable_array(out[axis]) as out_arr:
                    finite_diff(x_arr, axis=axis, dx=dx[axis], method=self.method,
                                pad_mode=self.pad_mode,
                                pad_const=self.pad_const,
                                out=out_arr)
            return out

    def derivative(self, point=None):
        """Return the derivative operator.

        The gradient is usually linear, but in case the 'constant'
        ``pad_mode`` is used with nonzero ``pad_const``, the
        derivative is given by the Gradient with ``pad_const=0``.

        Parameters
        ----------
        point : `domain` element, optional
            The point to take the derivative in. Does not change the result
            since the operator is affine.
        """
        if self.pad_mode == 'constant' and self.pad_const != 0:
            return Gradient(self.domain, self.range, self.method,
                            pad_mode=self.pad_mode,
                            pad_const=0)
        else:
            return self

    @property
    def adjoint(self):
        """Adjoint of this operator.

        The adjoint is given by the negative `Divergence` with corrections for
        the method and padding.

        The Divergence is constructed from a ``space`` as a product space
        operator ``space^n --> space``, hence we need to provide the domain of
        this operator.
        """
        if not self.is_linear:
            raise ValueError(
                f"operator with nonzero pad_const ({self.pad_const}) is not linear and has no adjoint"
            )

        return - Divergence(domain=self.range, range=self.domain,
                            method=_ADJ_METHOD[self.method],
                            pad_mode=_ADJ_PADDING[self.pad_mode],
                            pad_const=self.pad_const)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.domain]
        optargs = [('range', self.range, self.domain ** self.domain.ndim),
                   ('method', self.method, 'forward'),
                   ('pad_mode', self.pad_mode, 'constant'),
                   ('pad_const', self.pad_const, 0)]
        inner_str = signature_string(posargs, optargs,
                                     sep=[',\n', ', ', ',\n'],
                                     mod=['!r', ''])
        return f"{self.__class__.__name__}(\n{indent(inner_str)}\n)"

    def __str__(self):
        """Return ``str(self)``."""
        dom_ran_str = "\n-->\n".join([repr(self.domain), repr(self.range)])
        return f"{self.__class__.__name__}:\n{indent(dom_ran_str)}"


class Divergence(PointwiseTensorFieldOperator):
    """Divergence operator for `DiscretizedSpace` spaces.

    Calls helper function `finite_diff` for each component of the input
    product space vector. For the adjoint of the `Divergence` operator to
    match the negative `Gradient` operator implicit zero is assumed.
    """

    def __init__(self, domain=None, range=None, method='forward',
                 pad_mode='constant', pad_const=0):
        """Initialize a new instance.

        Zero padding is assumed for the adjoint of the `Divergence`
        operator to match the negative `Gradient` operator.

        Parameters
        ----------
        domain : power space of `DiscretizedSpace`, optional
            Space of elements which the operator acts on.
            This is required if ``range`` is not given.
        range : `DiscretizedSpace`, optional
            Space of elements to which the operator maps.
            This is required if ``domain`` is not given.
        method : {'forward', 'backward', 'central'}, optional
            Finite difference method to be used
        pad_mode : string, optional
            The padding mode to use outside the domain.

            ``'constant'``: Fill with ``pad_const``.

            ``'symmetric'``: Reflect at the boundaries, not doubling the

            ``'periodic'``: Fill in values from the other side, keeping
            the order.

            ``'order0'``: Extend constantly with the outmost values
            (ensures continuity).

            ``'order1'``: Extend with constant slope (ensures continuity of
            the first derivative). This requires at least 2 values along
            each axis.

            ``'order2'``: Extend with second order accuracy (ensures continuity
            of the second derivative). This requires at least 3 values along
            each axis.

        pad_const : float, optional
            For ``pad_mode == 'constant'``, ``f`` assumes
            ``pad_const`` for indices outside the domain of ``f``

        Examples
        --------
        Initialize a Divergence opeator:

        >>> ran = odl.uniform_discr([0, 0], [3, 5], (3, 5))
        >>> dom = odl.ProductSpace(ran, ran.ndim)  # 2-dimensional
        >>> div = Divergence(dom)
        >>> div.range == ran
        True
        >>> div2 = Divergence(range=ran)
        >>> div2.domain == dom
        True
        >>> div3 = Divergence(domain=dom, range=ran)
        >>> div3.domain == dom
        True
        >>> div3.range == ran
        True

        Call the operator:

        >>> data = np.array([[0., 1., 2., 3., 4.],
        ...                  [1., 2., 3., 4., 5.],
        ...                  [2., 3., 4., 5., 6.]])
        >>> f = div.domain.element([data, data])
        >>> div_f = div(f)
        >>> print(div_f)
        [[  2.,   2.,   2.,   2.,  -3.],
         [  2.,   2.,   2.,   2.,  -4.],
         [ -1.,  -2.,  -3.,  -4., -12.]]

        Verify adjoint:

        >>> g = div.range.element(data ** 2)
        >>> adj_div_g = div.adjoint(g)
        >>> g.inner(div_f) / f.inner(adj_div_g)
        1.0
        """
        if domain is None and range is None:
            raise ValueError("either `domain` or `range` must be specified")

        if domain is None:
            domain = ProductSpace(range, range.ndim)

        if range is None:
            try:
                range = domain[0]
            except TypeError:
                pass

        # Check `domain` first since `range` may end up to be `None` in
        # the case filtered out here (see above)
        if not isinstance(domain, ProductSpace):
            raise TypeError(f"`domain` {domain} is not a `ProductSpace` instance")
        if not domain.is_power_space:
            raise ValueError(f"`domain` {domain} is not a power space")

        if not isinstance(range, DiscretizedSpace):
            raise TypeError(f"`range` {range} is not a `DiscretizedSpace` " "instance")

        if len(domain) != range.ndim:
            raise ValueError(
                f"`domain` must be a power space of length n = {range.ndim},with `n == range.ndim`, got n = {len(domain)} instead"
            )

        linear = not (pad_mode == "constant" and pad_const != 0)
        super().__init__(domain, range, base_space=range, linear=linear)

        self.method, method_in = str(method).lower(), method
        if method not in _SUPPORTED_DIFF_METHODS:
            raise ValueError(f"`method` {method_in} not understood")

        self.pad_mode, pad_mode_in = str(pad_mode).lower(), pad_mode
        if pad_mode not in _SUPPORTED_PAD_MODES:
            raise ValueError(f"`pad_mode` {pad_mode_in} not understood")

        self.pad_const = range.field.element(pad_const)

    def _call(self, x, out=None):
        """Calculate the divergence of ``x``."""

        ndim = self.range.ndim
        dx = self.range.cell_sides

        backend = self.range.array_backend

        def directional_derivative(axis, out=None):
            return finite_diff( x[axis], axis=axis, dx=dx[axis]
                              , method=self.method, pad_mode=self.pad_mode
                              , pad_const=self.pad_const
                              , out=out )

        if out is None:
            result = directional_derivative(0)
            for axis in range(1,len(x)):
                result += directional_derivative(axis)

            return self.range.element(result)

        else:
            assert(backend.impl != 'pytorch')

            tmp = self.range.element().asarray()
            with writable_array(out) as out_arr:
                for axis in range(ndim):
                    directional_derivative(axis, out=tmp)
                    if axis == 0:
                        out_arr[:] = tmp
                    else:
                        out_arr += tmp
 
            return out

    def derivative(self, point=None):
        """Return the derivative operator.

        The Divergence is usually linear, but in case the 'constant'
        ``pad_mode`` is used with nonzero ``pad_const``, the
        derivative is given by the Divergence with ``pad_const=0``.

        Parameters
        ----------
        point : `domain` element, optional
            The point to take the derivative in. Does not change the result
            since the operator is affine.
        """
        if self.pad_mode == 'constant' and self.pad_const != 0:
            return Divergence(self.domain, self.range, self.method,
                              pad_mode=self.pad_mode, pad_const=0)
        else:
            return self

    @property
    def adjoint(self):
        """Adjoint of this operator.

        The adjoint is given by the negative `Gradient` with corrections for
        the method and padding.
        """
        if not self.is_linear:
            raise ValueError(
                f"operator with nonzero pad_const ({self.pad_const}) is not linear and has no adjoint"
            )

        return - Gradient(self.range, self.domain,
                          method=_ADJ_METHOD[self.method],
                          pad_mode=_ADJ_PADDING[self.pad_mode])

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.domain]
        optargs = [('range', self.range, self.domain[0]),
                   ('method', self.method, 'forward'),
                   ('pad_mode', self.pad_mode, 'constant'),
                   ('pad_const', self.pad_const, 0)]
        inner_str = signature_string(posargs, optargs,
                                     sep=[',\n', ', ', ',\n'],
                                     mod=['!r', ''])
        return f"{self.__class__.__name__}(\n{indent(inner_str)}\n)"

    def __str__(self):
        """Return ``str(self)``."""
        dom_ran_str = "\n-->\n".join([repr(self.domain), repr(self.range)])
        return f"{self.__class__.__name__}:\n{indent(dom_ran_str)}"


class Laplacian(PointwiseTensorFieldOperator):
    """Spatial Laplacian operator for `DiscretizedSpace` spaces.

    Calls helper function `finite_diff` to calculate each component of the
    resulting product space vector.

    Outside the domain zero padding is assumed.
    """

    def __init__(self, domain, range=None, pad_mode='constant', pad_const=0):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscretizedSpace`
            Space of elements which the operator is acting on.
        pad_mode : string, optional
            The padding mode to use outside the domain.

            ``'constant'``: Fill with ``pad_const``.

            ``'symmetric'``: Reflect at the boundaries, not doubling the
            outmost values.

            ``'periodic'``: Fill in values from the other side, keeping
            the order.

            ``'order0'``: Extend constantly with the outmost values
            (ensures continuity).

            ``'order1'``: Extend with constant slope (ensures continuity of
            the first derivative). This requires at least 2 values along
            each axis.

            ``'order2'``: Extend with second order accuracy (ensures continuity
            of the second derivative). This requires at least 3 values along
            each axis where padding is applied.

        pad_const : float, optional
            For ``pad_mode == 'constant'``, ``f`` assumes
            ``pad_const`` for indices outside the domain of ``f``

        Examples
        --------
        >>> data = np.array([[ 0., 0., 0.],
        ...                  [ 0., 1., 0.],
        ...                  [ 0., 0., 0.]])
        >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
        >>> f = space.element(data)
        >>> lap = Laplacian(space)
        >>> lap(f)
        uniform_discr([ 0.,  0.], [ 3.,  3.], (3, 3)).element(
            [[ 0.,  1.,  0.],
             [ 1., -4.,  1.],
             [ 0.,  1.,  0.]]
        )
        """
        if not isinstance(domain, DiscretizedSpace):
            raise TypeError(f"`domain` {domain} is not a DiscretizedSpace instance")

        if range is None:
            range = domain

        super().__init__(domain, range, base_space=domain, linear=True)

        self.pad_mode, pad_mode_in = str(pad_mode).lower(), pad_mode
        if pad_mode not in _SUPPORTED_PAD_MODES:
            raise ValueError(f"`pad_mode` {pad_mode_in} not understood")
        if pad_mode in ('order1', 'order1_adjoint',
                        'order2', 'order2_adjoint'):
            # TODO: Add these pad modes
            raise ValueError(f"`pad_mode` {pad_mode_in} not implemented for Laplacian.")

        self.pad_const = self.domain.field.element(pad_const)

    def _call(self, x, out=None):
        """Calculate the spatial Laplacian of ``x``."""
        if out is None:
            out = self.range.zero()
        else:
            out.set_zero()

        out_arr = out.asarray()

        x_arr, backend = get_array_and_backend(x)
        tmp = backend.array_namespace.empty(
            shape=out.shape, dtype=out.dtype,device=x.device
            )

        ndim = self.domain.ndim
        dx = self.domain.cell_sides

        with writable_array(out) as out_arr:
            for axis in range(ndim):
                # TODO: this can be optimized
                finite_diff(x_arr, axis=axis, dx=dx[axis] ** 2,
                            method='forward',
                            pad_mode=self.pad_mode,
                            pad_const=self.pad_const, out=tmp)

                out_arr += tmp

                finite_diff(x_arr, axis=axis, dx=dx[axis] ** 2,
                            method='backward',
                            pad_mode=self.pad_mode,
                            pad_const=self.pad_const, out=tmp)

                out_arr -= tmp

        return out

    def derivative(self, point=None):
        """Return the derivative operator.

        The Laplacian is usually linear, but in case the 'constant'
        ``pad_mode`` is used with nonzero ``pad_const``, the
        derivative is given by the derivative with 0 ``pad_const``.

        Parameters
        ----------
        point : ``domain`` element, optional
            The point to take the derivative in. Does not change the result
            since the operator is affine.
        """
        if self.pad_mode == 'constant' and self.pad_const != 0:
            return Laplacian(self.domain, self.range,
                             pad_mode=self.pad_mode, pad_const=0)
        else:
            return self

    @property
    def adjoint(self):
        """Return the adjoint operator.

        The laplacian is self-adjoint, so this returns ``self``.
        """
        return Laplacian(self.range, self.domain,
                         pad_mode=self.pad_mode, pad_const=0)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.domain]
        optargs = [('range', self.range, self.domain ** self.domain.ndim),
                   ('pad_mode', self.pad_mode, 'constant'),
                   ('pad_const', self.pad_const, 0)]
        inner_str = signature_string(posargs, optargs,
                                     sep=[',\n', ', ', ',\n'],
                                     mod=['!r', ''])
        return f"{self.__class__.__name__}(\n{indent(inner_str)}\n)"

    def __str__(self):
        """Return ``str(self)``."""
        dom_ran_str = "\n-->\n".join([repr(self.domain), repr(self.range)])
        return f"{self.__class__.__name__}:\n{indent(dom_ran_str)}"


def _finite_diff_numpy(f_arr, axis, dx=1.0, method='forward', out=None,
                pad_mode='constant', pad_const=0):
    """ NumPy-specific version of `finite_diff`. """

    ndim = f_arr.ndim

    if f_arr.shape[axis] < 2:
        raise ValueError(
            f"in axis {axis}: at least two elements required, got {f_arr.shape[axis]}"
        )

    if axis < 0:
        axis += ndim
    if not (0 <= axis < ndim):
        raise IndexError(f"`axis` {axis} outside the valid range 0 ... {ndim - 1}")

    dx, dx_in = float(dx), dx
    if dx <= 0 or not np.isfinite(dx):
        raise ValueError(f"`dx` must be positive, got {dx_in}")

    method, method_in = str(method).lower(), method
    if method not in _SUPPORTED_DIFF_METHODS:
        raise ValueError(f"`method` {method_in} was not understood")

    if pad_mode not in _SUPPORTED_PAD_MODES:
        raise ValueError(f"`pad_mode` {pad_mode} not understood")

    f_arr, backend = get_array_and_backend(f_arr)
    namespace = backend.array_namespace
    device = f_arr.device
    pad_const = backend.array_constructor([pad_const], dtype=f_arr.dtype, device=device)

    if out is None:
        out = namespace.empty_like(f_arr, dtype=f_arr.dtype, device=device)
    else:
        if out.shape != f_arr.shape:
            raise ValueError(f"expected output shape {f_arr.shape}, got {out.shape}")
    orig_shape = f_arr.shape

    if orig_shape[axis] < 2 and pad_mode == "order1":
        raise ValueError(
            f"size of array to small to use 'order1', needs at least 2 elements along axis {axis}."
        )
    if orig_shape[axis] < 3 and pad_mode == "order2":
        raise ValueError(
            f"size of array to small to use 'order2', needs at least 3 elements along axis {axis}."
        )

    # Swap axes so that the axis of interest is first. In NumPy (but not PyTorch),
    # this is a O(1) operation and is done to simplify the code below.
    out, out_in = namespace.swapaxes(out, 0, axis), out
    f_arr = namespace.swapaxes(f_arr, 0, axis)

    def fd_subtraction(a, b):
        namespace.subtract(a, b, out=out[1:-1])

    # Interior of the domain of f
    if method == 'central':
        # 1D equivalent: out[1:-1] = (f[2:] - f[:-2])/2.0
        fd_subtraction(f_arr[2:], f_arr[:-2])
        out[1:-1] /= 2.0

    elif method == 'forward':
        # 1D equivalent: out[1:-1] = (f[2:] - f[1:-1])
        fd_subtraction(f_arr[2:], f_arr[1:-1])

    elif method == 'backward':
        # 1D equivalent: out[1:-1] = (f[1:-1] - f[:-2])
        fd_subtraction(f_arr[1:-1], f_arr[:-2])

    # Boundaries
    if pad_mode == 'constant':
        # Assume constant value c for indices outside the domain of ``f``

        # With padding the method used on endpoints is the same as in the
        # interior of the domain of f

        if method == 'central':
            out[0] = (f_arr[1] - pad_const) / 2.0
            out[-1] = (pad_const - f_arr[-2]) / 2.0

        elif method == 'forward':
            out[0] = f_arr[1] - f_arr[0]
            out[-1] = pad_const - f_arr[-1]

        elif method == 'backward':
            out[0] = f_arr[0] - pad_const
            out[-1] = f_arr[-1] - f_arr[-2]

    elif pad_mode == 'symmetric':
        # Values of f for indices outside the domain of f are replicates of
        # the edge values

        # With padding the method used on endpoints is the same as in the
        # interior of the domain of f

        if method == 'central':
            out[0] = (f_arr[1] - f_arr[0]) / 2.0
            out[-1] = (f_arr[-1] - f_arr[-2]) / 2.0

        elif method == 'forward':
            out[0] = f_arr[1] - f_arr[0]
            out[-1] = 0

        elif method == 'backward':
            out[0] = 0
            out[-1] = f_arr[-1] - f_arr[-2]

    elif pad_mode == 'symmetric_adjoint':
        # The adjoint case of symmetric

        if method == 'central':
            out[0] = (f_arr[1] + f_arr[0]) / 2.0
            out[-1] = (-f_arr[-1] - f_arr[-2]) / 2.0

        elif method == 'forward':
            out[0] = f_arr[1]
            out[-1] = -f_arr[-1]

        elif method == 'backward':
            out[0] = f_arr[0]
            out[-1] = -f_arr[-2]

    elif pad_mode == 'periodic':
        # Values of f for indices outside the domain of f are replicates of
        # the edge values on the other side

        if method == 'central':
            out[0] = (f_arr[1] - f_arr[-1]) / 2.0
            out[-1] = (f_arr[0] - f_arr[-2]) / 2.0

        elif method == 'forward':
            out[0] = f_arr[1] - f_arr[0]
            out[-1] = f_arr[0] - f_arr[-1]

        elif method == 'backward':
            out[0] = f_arr[0] - f_arr[-1]
            out[-1] = f_arr[-1] - f_arr[-2]

    elif pad_mode == 'order0':
        # Values of f for indices outside the domain of f are replicates of
        # the edge value.

        if method == 'central':
            out[0] = (f_arr[1] - f_arr[0]) / 2.0
            out[-1] = (f_arr[-1] - f_arr[-2]) / 2.0

        elif method == 'forward':
            out[0] = f_arr[1] - f_arr[0]
            out[-1] = 0

        elif method == 'backward':
            out[0] = 0
            out[-1] = f_arr[-1] - f_arr[-2]

    elif pad_mode == 'order0_adjoint':
        # Values of f for indices outside the domain of f are replicates of
        # the edge value.

        if method == 'central':
            out[0] = (f_arr[0] + f_arr[1]) / 2.0
            out[-1] = -(f_arr[-1] + f_arr[-2]) / 2.0

        elif method == 'forward':
            out[0] = f_arr[1]
            out[-1] = -f_arr[-1]

        elif method == 'backward':
            out[0] = f_arr[0]
            out[-1] = -f_arr[-2]

    elif pad_mode == 'order1':
        # Values of f for indices outside the domain of f are linearly
        # extrapolated from the inside.

        # independent of ``method``

        out[0] = f_arr[1] - f_arr[0]
        out[-1] = f_arr[-1] - f_arr[-2]

    elif pad_mode == 'order1_adjoint':
        # Values of f for indices outside the domain of f are linearly
        # extrapolated from the inside.

        if method == 'central':
            out[0] = f_arr[0] + f_arr[1] / 2.0
            out[-1] = -f_arr[-1] - f_arr[-2] / 2.0

            # Increment in case array is very short and we get aliasing
            out[1] -= f_arr[0] / 2.0
            out[-2] += f_arr[-1] / 2.0

        elif method == 'forward':
            out[0] = f_arr[0] + f_arr[1]
            out[-1] = -f_arr[-1]

            # Increment in case array is very short and we get aliasing
            out[1] -= f_arr[0]

        elif method == 'backward':
            out[0] = f_arr[0]
            out[-1] = -f_arr[-1] - f_arr[-2]

            # Increment in case array is very short and we get aliasing
            out[-2] += f_arr[-1]

    elif pad_mode == 'order2':
        # 2nd order edges

        out[0] = -(3.0 * f_arr[0] - 4.0 * f_arr[1] + f_arr[2]) / 2.0
        out[-1] = (3.0 * f_arr[-1] - 4.0 * f_arr[-2] + f_arr[-3]) / 2.0

    elif pad_mode == 'order2_adjoint':
        # Values of f for indices outside the domain of f are quadratically
        # extrapolated from the inside.

        if method == 'central':
            out[0] = 1.5 * f_arr[0] + 0.5 * f_arr[1]
            out[-1] = -1.5 * f_arr[-1] - 0.5 * f_arr[-2]

            # Increment in case array is very short and we get aliasing
            out[1] -= 1.5 * f_arr[0]
            out[2] += 0.5 * f_arr[0]
            out[-3] -= 0.5 * f_arr[-1]
            out[-2] += 1.5 * f_arr[-1]

        elif method == 'forward':
            out[0] = 1.5 * f_arr[0] + 1.0 * f_arr[1]
            out[-1] = -1.5 * f_arr[-1]

            # Increment in case array is very short and we get aliasing
            out[1] -= 2.0 * f_arr[0]
            out[2] += 0.5 * f_arr[0]
            out[-3] -= 0.5 * f_arr[-1]
            out[-2] += 1.0 * f_arr[-1]

        elif method == 'backward':
            out[0] = 1.5 * f_arr[0]
            out[-1] = -1.0 * f_arr[-2] - 1.5 * f_arr[-1]

            # Increment in case array is very short and we get aliasing
            out[1] -= 1.0 * f_arr[0]
            out[2] += 0.5 * f_arr[0]
            out[-3] -= 0.5 * f_arr[-1]
            out[-2] += 2.0 * f_arr[-1]
    else:
        raise NotImplementedError('unknown pad_mode')

    # divide by step size
    out /= dx

    return out_in

def _finite_diff_pytorch(f_arr, axis, dx=1.0, method='forward',
                pad_mode='constant', pad_const=0):
    """ PyTorch-specific version of `finite_diff`. Notice that this has no output argument. """

    f_arr, _ = get_array_and_backend(f_arr)
    import torch

    ndim = f_arr.ndim

    if f_arr.shape[axis] < 2:
        raise ValueError(
            f"in axis {axis}: at least two elements required, got {f_arr.shape[axis]}"
        )

    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        raise IndexError(f"`axis` {axis} outside the valid range 0 ... {ndim - 1}")

    dx, dx_in = float(dx), dx
    if dx <= 0 or not np.isfinite(dx):
        raise ValueError(f"`dx` must be positive, got {dx_in}")

    method, method_in = str(method).lower(), method
    if method not in _SUPPORTED_DIFF_METHODS:
        raise ValueError(f"`method` {method_in} was not understood")

    if pad_mode not in _SUPPORTED_PAD_MODES:
        raise ValueError(f"`pad_mode` {pad_mode} not understood")

    orig_shape = f_arr.shape

    if orig_shape[axis] < 2 and pad_mode == "order1":
        raise ValueError(
            f"size of array to small to use 'order1', needs at least 2 elements along axis {axis}."
        )
    if orig_shape[axis] < 3 and pad_mode == "order2":
        raise ValueError(
            f"size of array to small to use 'order2', needs at least 3 elements along axis {axis}."
        )

    # Reshape (in O(1)), so the axis of interest is the pænultimate, all previous
    # axes are flattened into the batch dimension, and all subsequent axes flattened
    # into the final dimension. This allows a batched 2D convolution of final size 1
    # to perform the differentiation in only the axis of interest.
    f_arr = f_arr.reshape([ prod(orig_shape[:axis])
                          , 1
                          , orig_shape[axis]
                          , prod(orig_shape[axis+1:])
                          ])

    dtype = f_arr.dtype

    # Kernel for convolution that expresses the finite-difference operator on, at least,
    # the interior of the domain of f
    def as_kernel(mat):
        return torch.tensor(mat, dtype=dtype, device=f_arr.device)
    
    if method == 'central':
        fd_kernel = as_kernel([[[[-1],[0],[1]]]]) / (2*dx)
    elif method == 'forward':
        fd_kernel = as_kernel([[[[0],[-1],[1]]]]) / dx
    elif method == 'backward':
        fd_kernel = as_kernel([[[[-1],[1],[0]]]]) / dx

    if pad_mode == 'constant':
        if pad_const==0:
            result = torch.conv2d(f_arr, fd_kernel, padding='same')
        else:
            padding_arr = torch.ones_like(f_arr[:,:,0:1,:]) * pad_const
            result = torch.conv2d( 
                torch.cat([padding_arr, f_arr, padding_arr], dim=-2), fd_kernel, padding='valid' 
                )

    else:
        raise NotImplementedError(f"{pad_mode=} not implemented for PyTorch")

    return result.reshape(orig_shape)


def finite_diff(f, axis, dx=1.0, method='forward', out=None,
                pad_mode='constant', pad_const=0):
    """Calculate the partial derivative of ``f`` along a given ``axis``.

    In the interior of the domain of f, the partial derivative is computed
    using first-order accurate forward or backward difference or
    second-order accurate central differences.

    With padding the same method and thus accuracy is used on endpoints as
    in the interior i.e. forward and backward differences use first-order
    accuracy on edges while central differences use second-order accuracy at
    edges.

    Without padding one-sided forward or backward differences are used at
    the boundaries. The accuracy at the endpoints can then also be
    triggered by the edge order.

    The returned array has the same shape as the input array ``f``.

    Per default forward difference with dx=1 and no padding is used.

    Parameters
    ----------
    f : `array-like`
         An N-dimensional array.
    axis : int
        The axis along which the partial derivative is evaluated.
    dx : float, optional
        Scalar specifying the distance between sampling points along ``axis``.
    method : {'central', 'forward', 'backward'}, optional
        Finite difference method which is used in the interior of the domain
         of ``f``.
    out : `numpy.ndarray`, optional
         An N-dimensional array to which the output is written. Has to have
         the same shape as the input array ``f``.
    pad_mode : string, optional
        The padding mode to use outside the domain.

        ``'constant'``: Fill with ``pad_const``.

        ``'symmetric'``: Reflect at the boundaries, not doubling the
        outmost values.

        ``'periodic'``: Fill in values from the other side, keeping
        the order.

        ``'order0'``: Extend constantly with the outmost values
        (ensures continuity).

        ``'order1'``: Extend with constant slope (ensures continuity of
        the first derivative). This requires at least 2 values along
        each axis where padding is applied.

        ``'order2'``: Extend with second order accuracy (ensures continuity
        of the second derivative). This requires at least 3 values along
        each axis where padding is applied.

    pad_const : float, optional
        For ``pad_mode == 'constant'``, ``f`` assumes ``pad_const`` for
        indices outside the domain of ``f``

    Returns
    -------
    out : `numpy.ndarray`
        N-dimensional array of the same shape as ``f``. If ``out`` was
        provided, the returned object is a reference to it.

    Examples
    --------
    >>> f = np.array([ 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

    >>> finite_diff(f, axis=0)
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  -9.])

    Without arguments the above defaults to:

    >>> finite_diff(f, axis=0, dx=1.0, method='forward', pad_mode='constant')
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  -9.])

    Parameters can be changed one by one:

    >>> finite_diff(f, axis=0, dx=0.5)
    array([ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  -18.])
    >>> finite_diff(f, axis=0, pad_mode='order1')
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 1.])

    Central differences and different edge orders:

    >>> finite_diff(0.5 * f ** 2, axis=0, method='central', pad_mode='order1')
    array([ 0.5,  1. ,  2. ,  3. ,  4. ,  5. ,  6. ,  7. ,  8. ,  8.5])
    >>> finite_diff(0.5 * f ** 2, axis=0, method='central', pad_mode='order2')
    array([-0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])

    In-place evaluation:

    >>> out = f.copy()
    >>> out is finite_diff(f, axis=0, out=out)
    True
    """
    _, backend = get_array_and_backend(f)
    if pad_mode == 'constant' and backend.impl=='pytorch':
        if out is None:
            return _finite_diff_pytorch(
                f, axis, dx=dx, method=method, pad_mode=pad_mode, pad_const=pad_const
                )
        assert isinstance(out, backend.array_type), f"{type(out)=}"
        if out.shape != f.shape:
            raise ValueError('expected output shape {}, got {}'
                             ''.format(f.shape, out.shape))
        out[:] = _finite_diff_pytorch(
            f, axis, dx=dx, method=method, pad_mode=pad_mode, pad_const=pad_const
            )
        return out
    else:
        return _finite_diff_numpy(
            f, axis, dx=dx, method=method, out=out, pad_mode=pad_mode, pad_const=pad_const)



if __name__ == '__main__':
    from odl.core.util.testutils import run_doctests
    run_doctests()
