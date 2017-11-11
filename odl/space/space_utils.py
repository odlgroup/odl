# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utility functions for space implementations."""

from __future__ import print_function, division, absolute_import
import numpy as np

from odl.set import RealNumbers, ComplexNumbers
from odl.space.entry_points import tensor_space_impl
from odl.space.pspace import ProductSpaceElement
from odl.space.weighting import ArrayWeighting, ConstWeighting
from odl.util import OptionalArgDecorator


__all__ = ('vector', 'tensor_space', 'cn', 'rn')


def vector(array, dtype=None, order=None, impl='numpy'):
    """Create a vector from an array-like object.

    Parameters
    ----------
    array : `array-like`
        Array from which to create the vector. Scalars become
        one-dimensional vectors.
    dtype : optional
        Set the data type of the vector manually with this option.
        By default, the space type is inferred from the input data.
    order : {None, 'C', 'F'}, optional
        Axis ordering of the data storage. For the default ``None``,
        no contiguousness is enforced, avoiding a copy if possible.
    impl : str, optional
        Impmlementation back-end for the space. See
        `odl.space.entry_points.tensor_space_impl_names` for available
        options.

    Returns
    -------
    vector : `Tensor`
        Vector created from the input array. Its concrete type depends
        on the provided arguments.

    Notes
    -----
    This is a convenience function and not intended for use in
    speed-critical algorithms.

    Examples
    --------
    Create one-dimensional vectors:

    >>> odl.vector([1, 2, 3])  # No automatic cast to float
    tensor_space(3, dtype=int).element([1, 2, 3])
    >>> odl.vector([1, 2, 3], dtype=float)
    rn(3).element([ 1.,  2.,  3.])
    >>> odl.vector([1, 2 - 1j, 3])
    cn(3).element([ 1.+0.j,  2.-1.j,  3.+0.j])

    Non-scalar types are also supported:

    >>> odl.vector([True, True, False])
    tensor_space(3, dtype=bool).element([ True,  True, False])

    The function also supports multi-dimensional input:

    >>> odl.vector([[1, 2, 3],
    ...             [4, 5, 6]])
    tensor_space((2, 3), dtype=int).element(
        [[1, 2, 3],
         [4, 5, 6]]
    )
    """
    # Sanitize input
    arr = np.array(array, copy=False, order=order, ndmin=1)
    if arr.dtype is object:
        raise ValueError('invalid input data resulting in `dtype==object`')

    # Set dtype
    if dtype is not None:
        space_dtype = dtype
    else:
        space_dtype = arr.dtype

    space = tensor_space(arr.shape, dtype=space_dtype, impl=impl)
    return space.element(arr)


def tensor_space(shape, dtype=None, impl='numpy', **kwargs):
    """Return a tensor space with arbitrary scalar data type.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis for elements in this space. A
        single integer results in a space with 1 axis.
    dtype : optional
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string.
        For ``None``, the `TensorSpace.default_dtype` of the
        created space is used.
    impl : str, optional
        Impmlementation back-end for the space. See
        `odl.space.entry_points.tensor_space_impl_names` for available
        options.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    space : `TensorSpace`

    Examples
    --------
    Space of 3-tuples with ``int64`` entries (although not strictly a
    vector space):

    >>> odl.tensor_space(3, dtype='int64')
    tensor_space(3, dtype=int)

    2x3 tensors with same data type:

    >>> odl.tensor_space((2, 3), dtype='int64')
    tensor_space((2, 3), dtype=int)

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'float64'``:

    >>> ts = odl.tensor_space((2, 3))
    >>> ts
    rn((2, 3))
    >>> ts.dtype
    dtype('float64')

    See Also
    --------
    rn, cn : Constructors for real and complex spaces
    """
    tspace_cls = tensor_space_impl(impl)

    if dtype is None:
        dtype = tspace_cls.default_dtype()

    # Use args by keyword since the constructor may take other arguments
    # by position
    return tspace_cls(shape=shape, dtype=dtype, **kwargs)


def cn(shape, dtype=None, impl='numpy', **kwargs):
    """Return a space of complex tensors.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis for elements in this space. A
        single integer results in a space with 1 axis.
    dtype : optional
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string. Only complex floating-point data types are allowed.
        For ``None``, the `TensorSpace.default_dtype` of the
        created space is used in the form
        ``default_dtype(ComplexNumbers())``.
    impl : str, optional
        Impmlementation back-end for the space. See
        `odl.space.entry_points.tensor_space_impl_names` for available
        options.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    cn : `TensorSpace`

    Examples
    --------
    Space of complex 3-tuples with ``complex64`` entries:

    >>> odl.cn(3, dtype='complex64')
    cn(3, dtype='complex64')

    Complex 2x3 tensors with ``complex64`` entries:

    >>> odl.cn((2, 3), dtype='complex64')
    cn((2, 3), dtype='complex64')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'complex128'``:

    >>> space = odl.cn((2, 3))
    >>> space
    cn((2, 3))
    >>> space.dtype
    dtype('complex128')

    See Also
    --------
    tensor_space : Space of tensors with arbitrary scalar data type.
    rn : Real tensor space.
    """
    cn_cls = tensor_space_impl(impl)

    if dtype is None:
        dtype = cn_cls.default_dtype(ComplexNumbers())

    # Use args by keyword since the constructor may take other arguments
    # by position
    cn = cn_cls(shape=shape, dtype=dtype, **kwargs)
    if not cn.is_complex:
        raise ValueError('data type {!r} not a complex floating-point type.'
                         ''.format(dtype))
    return cn


def rn(shape, dtype=None, impl='numpy', **kwargs):
    """Return a space of real tensors.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis for elements in this space. A
        single integer results in a space with 1 axis.
    dtype : optional
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string. Only real floating-point data types are allowed.
        For ``None``, the `TensorSpace.default_dtype` of the
        created space is used in the form
        ``default_dtype(RealNumbers())``.
    impl : str, optional
        Impmlementation back-end for the space. See
        `odl.space.entry_points.tensor_space_impl_names` for available
        options.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    real_space : `TensorSpace`

    Examples
    --------
    Space of real 3-tuples with ``float32`` entries:

    >>> odl.rn(3, dtype='float32')
    rn(3, dtype='float32')

    Real 2x3 tensors with ``float32`` entries:

    >>> odl.rn((2, 3), dtype='float32')
    rn((2, 3), dtype='float32')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'float64'``:

    >>> ts = odl.rn((2, 3))
    >>> ts
    rn((2, 3))
    >>> ts.dtype
    dtype('float64')

    See Also
    --------
    tensor_space : Space of tensors with arbitrary scalar data type.
    cn : Complex tensor space.
    """
    rn_cls = tensor_space_impl(impl)

    if dtype is None:
        dtype = rn_cls.default_dtype(RealNumbers())

    # Use args by keyword since the constructor may take other arguments
    # by position
    rn = rn_cls(shape=shape, dtype=dtype, **kwargs)
    if not rn.is_real:
        raise ValueError('data type {!r} not a real floating-point type.'
                         ''.format(dtype))
    return rn


class auto_weighting(OptionalArgDecorator):

    """Make an unweighted adjoint automatically account for weightings.

    Depending on the weightings, the correction is achieved by composing
    the unweighted operator with either `ScalingOperator` or
    `ConstantOperator`. The following rules are applied for the domain
    weighting ``w``, the range weighting ``v`` and the provided unweighted
    adjoint ``B^*``:

    - If both ``w`` and ``v`` are arrays, return ::

        (1 / w) * (B^*) * v

    - If ``w`` is an array and ``v`` a constant, return ::

        (v / w) * (B^*)

    - If ``w`` is a constant and ``v`` an array, return ::

        (B^*) * (w / v)

    - If both ``w`` and ``v`` are constants, return ::

        (B^*) * (v / w)

      if ``B.range.size < B.domain.size``, otherwise ::

        (v / w) * (B^*)

    - Ignore constants 1.0.

    To avoid the inconvenience of dealing with `OperatorComp` objects,
    the given operator is monkey-patched instead of composed.

    Parameters
    ----------
    unweighted_adjoint : `Operator`
        Unweighted variant of the adjoint. It will be patched with a
        new ``_call()`` method.
        The weightings of ``domain`` and ``range`` of the operator
        must be `ArrayWeighting` or `ConstWeighting`.
    optimize : bool, optional
        If ``True``, merge and move around constant weightings for
        highest expected efficiency.

    Notes
    -----
    Consider a linear operator :math:`A: X_w \\to Y_v` between spaces with
    weights :math:`w` and :math:`v`, respectively, along with the same
    operator :math:`B: X \\to Y` defined between the unweighted variants of
    the spaces. (This means that :math:`B f = A f` for all
    :math:`f \\in X \cong X_w`).

    Then, the adjoint of :math:`A` is related to the adjoint of :math:`B`
    as follows:

    .. math::
        \\langle Af, g \\rangle_{Y_v} =
        \\langle Bf, v \cdot g \\rangle_Y =
        \\langle f, B^*(v \cdot g) \\rangle_X =
        \\langle f, w^{-1}\, B^*(v \cdot g) \\rangle_{X_w}.

    Thus, from the existing unweighted adjoint :math:`B^*` one can compute
    the weighted one as :math:`A^* = w^{-1}\, B^*(v\, \cdot)`.
    Depending on the types of weighting, this expression can be simplified
    further, e.g., a constant weight can be absorbed into the other weight.
    """

    @staticmethod
    def _wrapper(unweighted_adjoint, optimize=True):
        """Return the weighted variant of the unweighted adjoint."""
        # Support decorating the `adjoint` property directly
        import inspect
        from functools import wraps
        from odl.operator.operator import Operator

        if (inspect.isfunction(unweighted_adjoint) and
                unweighted_adjoint.__name__ == 'adjoint'):
            # We need this level of indirection since `self` needs to
            # be filled in with the instance, but we decorate at class
            # level.
            @wraps(unweighted_adjoint)
            def weighted_adjoint(self):
                adj = unweighted_adjoint(self)
                if not isinstance(adj, Operator):
                    raise TypeError('`adjoint` did not return an `Operator`')
                if adj is self:
                    raise TypeError(
                        'returning `self` in an `adjoint` property using '
                        '`auto_weighting` is not allowed')

                # This is for cached adjoints: don't double-wrap
                if hasattr(adj, '_call_unweighted'):
                    return adj
                else:
                    return auto_weighting._instance_wrapper(adj, optimize)

            return weighted_adjoint

        else:
            raise TypeError(
                "`auto_weighting` can only be applied to 'adjoint' methods "
                '(@auto_weighting decorator)')

    @staticmethod
    def _instance_wrapper(unweighted_adjoint, optimize=True):
        """Wrapper for `Operator` instances."""
        # Use notions of the original operator, not the adjoint
        dom_weighting = unweighted_adjoint.range.weighting
        ran_weighting = unweighted_adjoint.domain.weighting

        if isinstance(dom_weighting, ArrayWeighting):
            dom_w_type = 'array'
            dom_w = dom_weighting.array
        elif isinstance(dom_weighting, ConstWeighting):
            dom_w_type = 'const'
            dom_w = dom_weighting.const
        else:
            raise TypeError(
                'weighting of `unweighted_adjoint.range` must be of '
                'type `ArrayWeighting` or `ConstWeighting`, got {}'
                ''.format(type(dom_weighting)))

        if isinstance(ran_weighting, ArrayWeighting):
            ran_w_type = 'array'
            ran_w = ran_weighting.array
        elif isinstance(ran_weighting, ConstWeighting):
            ran_w_type = 'const'
            ran_w = ran_weighting.const
        else:
            raise TypeError(
                'weighting of `unweighted_adjoint.domain` must be of '
                'type `ArrayWeighting` or `ConstWeighting`, got {}'
                ''.format(type(ran_weighting)))

        # Compute the effective weights and mark constants 1.0 as to be
        # skipped
        if not optimize:
            new_dom_w, new_ran_w = dom_w, ran_w
            skip_dom = dom_w_type == 'const' and dom_w == 1.0
            skip_ran = ran_w_type == 'const' and ran_w == 1.0
        elif dom_w_type == 'array' and ran_w_type == 'array':
            new_dom_w, new_ran_w = dom_w, ran_w
            skip_dom = skip_ran = False
        elif dom_w_type == 'array' and ran_w_type == 'const':
            new_dom_w = dom_w / ran_w
            new_ran_w = 1.0
            skip_dom = False
            skip_ran = True
        elif dom_w_type == 'const' and ran_w_type == 'array':
            new_dom_w = 1.0
            new_ran_w = ran_w / dom_w
            skip_dom = True
            skip_ran = False
        elif dom_w_type == 'const' and ran_w_type == 'const':
            if unweighted_adjoint.domain.size < unweighted_adjoint.range.size:
                new_dom_w = 1.0
                new_ran_w = ran_w / dom_w
                skip_dom = True
                skip_ran = False
            else:
                new_dom_w = dom_w / ran_w
                new_ran_w = 1.0
                skip_dom = False
                skip_ran = True

        # Define the new `_call` depending on original signature
        self = unweighted_adjoint

        def mul_weight(x, w):
            """Multiplication with weights that works in product spaces."""
            if not isinstance(x, ProductSpaceElement) or np.isscalar(w):
                return w * x
            else:
                # Product space, array weight
                return x.space.element([wi * xi for xi, wi in zip(x, w)])

        def idiv_weight(x, w):
            """In-place division by weights that works in product spaces."""
            if not isinstance(x, ProductSpaceElement) or np.isscalar(w):
                x /= w
            else:
                # Product space, array weight
                for xi, wi in zip(x, w):
                    xi /= wi

        # Monkey-patching starts here
        if self._call_has_out and self._call_out_optional:
            def _call(x, out=None):
                if not skip_ran:
                    x = mul_weight(x, new_ran_w)
                out = self._call_unweighted(x, out=out)
                if not skip_dom:
                    idiv_weight(out, new_dom_w)
                return out

            self._call_unweighted = self._call_in_place
            self._call_in_place = self._call_out_of_place = _call

        elif self._call_has_out and not self._call_out_optional:
            def _call(x, out):
                if not skip_ran:
                    x = mul_weight(x, new_ran_w)
                self._call_unweighted(x, out=out)
                if not skip_dom:
                    idiv_weight(out, new_dom_w)
                return out

            self._call_unweighted = self._call_in_place
            self._call_in_place = _call

        else:
            def _call(x):
                if not skip_ran:
                    x = mul_weight(x, new_ran_w)
                out = self._call_unweighted(x)
                if not skip_dom:
                    idiv_weight(out, new_dom_w)
                return out

            self._call_unweighted = self._call_out_of_place
            self._call_out_of_place = _call

        return self


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
