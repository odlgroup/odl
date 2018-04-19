# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for computing the gradient and Hessian of functionals."""

from __future__ import absolute_import, division, print_function

import numpy as np

from odl.operator import Operator
from odl.solvers.functional.functional import Functional
from odl.space.base_tensors import TensorSpace
from odl.util import (
    is_floating_dtype, dtype_repr, signature_string_parts, repr_string)

__all__ = ('NumericalDerivative', 'NumericalGradient',)


class NumericalDerivative(Operator):

    """The derivative of an operator by finite differences.

    See Also
    --------
    NumericalGradient : Compute the gradient of a functional
    """

    def __init__(self, operator, point, method='forward', step=None):
        r"""Initialize a new instance.

        Parameters
        ----------
        operator : `Operator`
            The operator whose derivative should be computed numerically. Its
            domain and range must be `TensorSpace` spaces.
        point : ``operator.domain`` `element-like`
            The point to compute the derivative in.
        method : {'backward', 'forward', 'central'}, optional
            The method to use to compute the derivative.
        step : float, optional
            The step length used in the derivative computation.
            Default: selects the step according to the dtype of the space.

        Examples
        --------
        Compute a numerical estimate of the derivative (Hessian) of the squared
        L2 norm:

        >>> space = odl.rn(3)
        >>> func = odl.solvers.L2NormSquared(space)
        >>> hess = odl.solvers.NumericalDerivative(func.gradient, [1, 1, 1])
        >>> hess([0, 0, 1])
        rn(3).element([ 0.,  0.,  2.])

        Find the Hessian matrix:

        >>> hess_matrix = odl.matrix_representation(hess)
        >>> np.allclose(hess_matrix, 2 * np.eye(3))
        True

        Notes
        -----
        If the operator is :math:`A` and step size :math:`h` is used, the
        derivative in the point :math:`x` and direction :math:`dx` is computed
        as follows.

        ``method='backward'``:

        .. math::
            \partial A(x)(dx) =
            (A(x) - A(x - dx \cdot h / \| dx \|))
            \cdot \frac{\| dx \|}{h}

        ``method='forward'``:

        .. math::
            \partial A(x)(dx) =
            (A(x + dx \cdot h / \| dx \|) - A(x))
            \cdot \frac{\| dx \|}{h}

        ``method='central'``:

        .. math::
            \partial A(x)(dx) =
            (A(x + dx \cdot h / (2 \| dx \|)) -
             A(x - dx \cdot h / (2 \| dx \|))
            \cdot \frac{\| dx \|}{h}

        The number of operator evaluations is 2 in all cases.
        """
        if not isinstance(operator, Operator):
            raise TypeError('`operator` has to be an `Operator` instance')

        if not isinstance(operator.domain, TensorSpace):
            raise TypeError('`operator.domain` must be a `TensorSpace` '
                            'instance')
        if not isinstance(operator.range, TensorSpace):
            raise TypeError('`operator.range` must be a `TensorSpace` '
                            'instance')
        if not is_floating_dtype(operator.domain.dtype):
            raise ValueError('`operator.domain.dtype` must be a floating '
                             'point type, got {}'
                             ''.format(dtype_repr(operator.domain.dtype)))

        self.operator = operator
        self.point = operator.domain.element(point)

        if step is None:
            self.step = self._default_step(operator)
        else:
            self.step = float(step)

        self.method, method_in = str(method).lower(), method
        if self.method not in ('backward', 'forward', 'central'):
            raise ValueError("`method` '{}' not understood").format(method_in)

        super(NumericalDerivative, self).__init__(
            operator.domain, operator.range, linear=True)

    @staticmethod
    def _default_step(op):
        """Return the default step size for an operator.

        We use half of the number of digits of the machine epsilon for the
        operator domain dtype since this "usually" gives a good balance
        between precision and numerical stability.
        """
        return np.sqrt(np.finfo(op.domain.dtype).eps)

    def _call(self, dx):
        """Return ``self(x)``."""
        x = self.point

        dx_norm = dx.norm()
        if dx_norm == 0:
            return 0

        scaled_dx = dx * (self.step / dx_norm)

        if self.method == 'backward':
            dAdx = self.operator(x) - self.operator(x - scaled_dx)
        elif self.method == 'forward':
            dAdx = self.operator(x + scaled_dx) - self.operator(x)
        elif self.method == 'central':
            dAdx = (self.operator(x + scaled_dx / 2) -
                    self.operator(x - scaled_dx / 2))
        else:
            raise RuntimeError('unknown method')

        return dAdx * (dx_norm / self.step)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> space = odl.uniform_discr(0, 1, 4)
        >>> op = odl.ufunc_ops.square(space)
        >>> num_deriv = odl.solvers.NumericalDerivative(op, space.one())
        >>> num_deriv
        NumericalDerivative(
            square(uniform_discr(0.0, 1.0, 4)),
            point=uniform_discr(0.0, 1.0, 4).element([ 1.,  1.,  1.,  1.])
        )
        """
        posargs = [self.operator]
        optargs = [('point', self.point, None),
                   ('method', self.method, 'forward')]
        if not np.isclose(self.step, self._default_step(self.operator)):
            optargs.append(('step', self.step, None))

        inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class NumericalGradient(Operator):

    """The gradient of a `Functional` computed by finite differences.

    See Also
    --------
    NumericalDerivative : Compute directional derivative
    """

    def __init__(self, functional, method='forward', step=None):
        r"""Initialize a new instance.

        Parameters
        ----------
        functional : `Functional`
            The functional whose gradient should be computed. Its domain must
            be a `TensorSpace`.
        method : {'backward', 'forward', 'central'}, optional
            The method to use to compute the gradient.
        step : float, optional
            The step length used in the derivative computation.
            Default: selects the step according to the dtype of the space.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> func = odl.solvers.L2NormSquared(space)
        >>> grad = odl.solvers.NumericalGradient(func)
        >>> grad([1, 1, 1])
        rn(3).element([ 2.,  2.,  2.])

        The gradient gives the correct value with sufficiently small step size:

        >>> grad([1, 1, 1]) == func.gradient([1, 1, 1])
        True

        If the step is too large the result is not correct:

        >>> grad = odl.solvers.NumericalGradient(func, step=0.5)
        >>> grad([1, 1, 1])
        rn(3).element([ 2.5,  2.5,  2.5])

        But it can be improved by using the more accurate ``method='central'``:

        >>> grad = odl.solvers.NumericalGradient(func, method='central',
        ...                                      step=0.5)
        >>> grad([1, 1, 1])
        rn(3).element([ 2.,  2.,  2.])

        Notes
        -----
        If the functional is :math:`f` and step size :math:`h` is used, the
        gradient is computed as follows.

        ``method='backward'``:

        .. math::
            (\nabla f(x))_i = \frac{f(x) - f(x - h e_i)}{h}

        ``method='forward'``:

        .. math::
            (\nabla f(x))_i = \frac{f(x + h e_i) - f(x)}{h}

        ``method='central'``:

        .. math::
            (\nabla f(x))_i = \frac{f(x + (h/2) e_i) - f(x - (h/2) e_i)}{h}

        The number of function evaluations is ``functional.domain.size + 1`` if
        ``'backward'`` or ``'forward'`` is used and
        ``2 * functional.domain.size`` if ``'central'`` is used.
        On large domains this will be computationally infeasible.
        """
        if not isinstance(functional, Functional):
            raise TypeError('`functional` has to be a `Functional` instance')

        if not isinstance(functional.domain, TensorSpace):
            raise TypeError('`functional.domain` must be a `TensorSpace` '
                            'instance')
        if not is_floating_dtype(functional.domain.dtype):
            raise ValueError('`functional.domain.dtype` must be a floating '
                             'point type, got {}'
                             ''.format(dtype_repr(functional.domain.dtype)))

        self.functional = functional
        if step is None:
            self.step = self._default_step(functional)
        else:
            self.step = float(step)

        self.method, method_in = str(method).lower(), method
        if self.method not in ('backward', 'forward', 'central'):
            raise ValueError("`method` '{}' not understood").format(method_in)

        super(NumericalGradient, self).__init__(
            functional.domain, functional.domain, linear=functional.is_linear)

    @staticmethod
    def _default_step(func):
        """Return the default step size for a functional.

        We use half of the number of digits of the machine epsilon for the
        functional domain dtype since this "usually" gives a good balance
        between precision and numerical stability.
        """
        return np.sqrt(np.finfo(func.domain.dtype).eps)

    def _call(self, x):
        """Return ``self(x)``."""
        # The algorithm takes finite differences in one dimension at a time
        # reusing the dx vector to improve efficiency.
        dfdx = self.domain.zero()
        dx = self.domain.zero()

        if self.method == 'backward':
            fx = self.functional(x)
            for i in range(self.domain.size):
                dx[i - 1] = 0  # reset step from last iteration
                dx[i] = self.step
                dfdx[i] = fx - self.functional(x - dx)
        elif self.method == 'forward':
            fx = self.functional(x)
            for i in range(self.domain.size):
                dx[i - 1] = 0  # reset step from last iteration
                dx[i] = self.step
                dfdx[i] = self.functional(x + dx) - fx
        elif self.method == 'central':
            for i in range(self.domain.size):
                dx[i - 1] = 0  # reset step from last iteration
                dx[i] = self.step / 2
                dfdx[i] = self.functional(x + dx) - self.functional(x - dx)
        else:
            raise RuntimeError('unknown method')

        dfdx /= self.step
        return dfdx

    def derivative(self, point):
        """Return the derivative in ``point``.

        The derivative of the gradient is often called the Hessian.

        Parameters
        ----------
        point : `domain` `element-like`
            The point that the derivative should be taken in.

        Returns
        -------
        derivative : `NumericalDerivative`
            Numerical estimate of the derivative. Uses the same method as this
            operator does, but with half the number of significant digits in
            the step size in order to preserve numerical stability.

        Examples
        --------
        Compute a numerical estimate of the derivative of the squared L2 norm:

        >>> space = odl.rn(3)
        >>> func = odl.solvers.L2NormSquared(space)
        >>> grad = odl.solvers.NumericalGradient(func)
        >>> hess = grad.derivative([1, 1, 1])
        >>> hess([1, 0, 0])
        rn(3).element([ 2.,  0.,  0.])

        Find the Hessian matrix:

        >>> hess_matrix = odl.matrix_representation(hess)
        >>> np.allclose(hess_matrix, 2 * np.eye(3))
        True
        """
        return NumericalDerivative(self, point,
                                   method=self.method, step=np.sqrt(self.step))

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> func = odl.solvers.L2NormSquared(space)
        >>> num_grad = odl.solvers.NumericalGradient(func)
        >>> num_grad
        NumericalGradient(L2NormSquared(rn(3)))
        """
        posargs = [self.functional]
        optargs = [('method', self.method, 'forward')]
        if not np.isclose(self.step, self._default_step(self.functional)):
            optargs.append(('step', self.step, None))

        inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
