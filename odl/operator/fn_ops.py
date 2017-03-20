# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Default operators defined on fn (F^n where F is some field)."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from numbers import Integral
import numpy as np

from odl.operator.operator import Operator
from odl.space import tensor_space
from odl.space.base_tensors import TensorSpace
from odl.util import signature_string, indent_rows


__all__ = ('SamplingOperator', 'WeightedSumSamplingOperator',
           'FlatteningOperator')


def _normalize_sampling_points(sampling_points, ndim):
    """Normalize points to an ndim-long list of linear index arrays."""
    sampling_points_in = sampling_points
    if ndim == 0:
        sampling_points = [np.array(sampling_points, dtype=int, copy=False)]
        if sampling_points[0].size != 0:
            raise ValueError('`sampling_points` must be empty for '
                             '0-dim. `domain`')
    elif ndim == 1:
        if isinstance(sampling_points, Integral):
            sampling_points = (sampling_points,)
        sampling_points = [np.array(sampling_points, dtype=int, copy=False,
                                    ndmin=1)]
        if sampling_points[0].ndim > 1:
            raise ValueError('expected 1D index (array), got {}'
                             ''.format(sampling_points_in))
    else:
        try:
            iter(sampling_points)
        except TypeError:
            raise TypeError('`sampling_points` must be a sequence '
                            'for domain with ndim > 1')
        else:
            if np.ndim(sampling_points) == 1:
                sampling_points = [np.array(p, dtype=int)
                                   for p in sampling_points]
            else:
                sampling_points = [
                    np.array(pts, dtype=int, copy=False, ndmin=1)
                    for pts in sampling_points]
                if any(pts.ndim != 1 for pts in sampling_points):
                    raise ValueError(
                        'index arrays in `sampling_points` must be 1D, '
                        'got {!r}'.format(sampling_points_in))

    return sampling_points


class SamplingOperator(Operator):

    """Operator that samples coefficients.

    The operator is defined by ::

        SamplingOperator(f) == c * f[indices]

    with the weight c being determined by the variant. By choosing
    c = 1, this operator approximates point evaluations or inner products
    with dirac deltas, see option ``'point_eval'``. By choosing
    c = cell_volume it approximates the integration of f over the cell by
    multiplying its function valume with the cell volume, see option
    ``'integrate'``.
    """

    def __init__(self, domain, sampling_points, variant='point_eval'):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `TensorSpace`
            Set of elements on which this operator acts.
        sampling_points : 1D `array-like` or sequence of 1D array-likes
            Indices that determine the sampling points.
            In n dimensions, it should be a sequence of n arrays, where
            each member array is of equal length N. The indexed positions
            are ``(arr1[i], arr2[i], ..., arrn[i])``, in total N
            points.
            If ``domain`` is one-dimensional, a single array-like can be
            used. Likewise, a single point can be given as integer in 1D,
            and as a array-like sequence in nD.
        variant : {'point_eval', 'integrate'}, optional
            For ``'point_eval'`` this operator performs the sampling by
            evaluation the function at the sampling points. The
            ``'integrate'`` variant approximates integration by
            multiplying point evaluation with the cell volume.

        Examples
        --------
        Sampling in 1d can be done with a single index (an int) or a
        sequence of such:

        >>> space = odl.uniform_discr(0, 1, 4)
        >>> op = odl.SamplingOperator(space, sampling_points=1)
        >>> x = space.element([1, 2, 3, 4])
        >>> op(x)
        rn(1).element([2.0])
        >>> op = odl.SamplingOperator(space, sampling_points=[1, 2, 1])
        >>> op(x)
        rn(3).element([2.0, 3.0, 2.0])

        There are two variants ``'point_eval'`` (default) and
        ``'integrate'``, where the latter scales values by the cell
        volume to approximate the integral over the cells of the points:

        >>> op = odl.SamplingOperator(space, sampling_points=[1, 2, 1],
        ...                           variant='integrate')
        >>> space.cell_volume  # the scaling constant
        0.25
        >>> op(x)
        rn(3).element([0.5, 0.75, 0.5])

        In higher dimensions, a sequence of index array-likes must be
        given, or a single sequence for a single point:

        >>> space = odl.uniform_discr([0, 0], [1, 1], (2, 3))
        >>> # Sample at the index (0, 2)
        >>> op = odl.SamplingOperator(space, sampling_points=[0, 2])
        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> op(x)
        rn(1).element([3.0])
        >>> sampling_points = [[0, 1],  # indices (0, 2) and (1, 1)
        ...                    [2, 1]]
        >>> op = odl.SamplingOperator(space, sampling_points)
        >>> op(x)
        rn(2).element([3.0, 5.0])
        """
        if not isinstance(domain, TensorSpace):
            raise TypeError('`domain` must be a `TensorSpace` instance, got '
                            '{!r}'.format(domain))

        self.__sampling_points = _normalize_sampling_points(sampling_points,
                                                            domain.ndim)
        self.__variant = str(variant).lower()
        if self.variant not in ('point_eval', 'integrate'):
            raise ValueError('`variant` {!r} not understood'.format(variant))

        ran = tensor_space(self.sampling_points[0].size, dtype=domain.dtype)
        super().__init__(domain, ran, linear=True)

    @property
    def variant(self):
        """Weighting scheme for the sampling operator."""
        return self.__variant

    @property
    def sampling_points(self):
        """Indices where to sample the function."""
        return self.__sampling_points

    def _call(self, x, out=None):
        """Collect indices weighted with the cell volume."""
        if out is None:
            out = x[self.sampling_points]
        else:
            out[:] = x[self.sampling_points]

        if self.variant == 'point_eval':
            weights = 1.0
        elif self.variant == 'integrate':
            weights = getattr(self.domain, 'cell_volume', 1.0)
        else:
            raise RuntimeError('bad variant {!r}'.format(self.variant))

        if weights != 1.0:
            out *= weights

        return out

    @property
    def adjoint(self):
        """Adjoint of the sampling operator, a `WeightedSumSamplingOperator`.

        If each sampling point occurs only once, the adjoint consists
        in inserting the given values into the output at the sampling
        points. Duplicate sampling points are weighted with their
        multiplicity.

        Examples
        --------
        >>> space = odl.uniform_discr([-1, -1], [1, 1], shape=(2, 3))
        >>> # Point (0, 0) occurs twice
        >>> sampling_points = [[0, 1, 1, 0],
        ...                    [0, 1, 2, 0]]
        >>> op = odl.SamplingOperator(space, sampling_points)
        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> op.adjoint(op(x)).inner(x) - op(x).inner(op(x)) < 1e-10
        True
        >>> op = odl.SamplingOperator(space, sampling_points,
        ...                           variant='integrate')
        >>> # Put ones at the indices in sampling_points, using double
        >>> # weight at (0, 0) since it occurs twice
        >>> op.adjoint(op.range.one())
        uniform_discr([-1.0, -1.0], [1.0, 1.0], (2, 3)).element(
            [[2.0, 0.0, 0.0],
             [0.0, 1.0, 1.0]]
        )
        >>> op.adjoint(op(x)).inner(x) - op(x).inner(op(x)) < 1e-10
        True
        """
        if self.variant == 'point_eval':
            variant = 'dirac'
        elif self.variant == 'integrate':
            variant = 'char_fun'
        else:
            raise RuntimeError('bad variant {!r}'.format(self.variant))

        return WeightedSumSamplingOperator(self.domain, self.sampling_points,
                                           variant)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.domain, self.sampling_points]
        optargs = [('variant', self.variant, 'point_eval')]
        sig_str = signature_string(posargs, optargs, mod=['!r', ''],
                                   sep=[',\n', '', ',\n'])
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(sig_str))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class WeightedSumSamplingOperator(Operator):

    """Operator computing the sum of coefficients at sampling locations.

    This operator is the adjoint of `SamplingOperator`.

    Notes
    -----
    The weighted sum sampling operator for a sequence
    :math:`I = (i_n)_{n=1}^N`
    of indices (possibly with duplicates) is given by

    .. math::
        W_I(g)(x) = \sum_{i \\in I} d_i(x) g_i,

    where :math:`g \\in \mathbb{F}^N` is the value vector, and
    :math:`d_i` is either a Dirac delta or a characteristic function of
    the cell centered around the point indexed by :math:`i`.
    """

    def __init__(self, range, sampling_points, variant='char_fun'):
        """Initialize a new instance.

        Parameters
        ----------
        range : `TensorSpace`
            Set of elements into which this operator maps.
        sampling_points : 1D `array-like` or sequence of 1D array-likes
            Indices that determine the sampling points.
            In n dimensions, it should be a sequence of n arrays, where
            each member array is of equal length N. The indexed positions
            are ``(arr1[i], arr2[i], ..., arrn[i])``, in total N
            points.
            If ``range`` is one-dimensional, a single array-like can be
            used. Likewise, a single point can be given as integer in 1D,
            and as a array-like sequence in nD.
        variant : {'char_fun', 'dirac'}, optional
            This option determines which function to sum over.

        Examples
        --------
        In 1d, a single index (an int) or a sequence of such can be used
        for indexing.

        >>> space = odl.uniform_discr(0, 1, 4)
        >>> op = odl.WeightedSumSamplingOperator(space, sampling_points=1)
        >>> op.domain
        rn(1)
        >>> x = op.domain.element([1])
        >>> # Put value 1 at index 1
        >>> op(x)
        uniform_discr(0.0, 1.0, 4).element([0.0, 1.0, 0.0, 0.0])
        >>> op = odl.WeightedSumSamplingOperator(space,
        ...                                      sampling_points=[1, 2, 1])
        >>> op.domain
        rn(3)
        >>> x = op.domain.element([1, 0.5, 0.25])
        >>> # Index 1 occurs twice and gets two contributions (1 and 0.25)
        >>> op(x)
        uniform_discr(0.0, 1.0, 4).element([0.0, 1.25, 0.5, 0.0])

        The ``'dirac'`` variant scales the values by the reciprocal
        cell volume of the operator range:

        >>> op = odl.WeightedSumSamplingOperator(
        ...     space, sampling_points=[1, 2, 1], variant='dirac')
        >>> x = op.domain.element([1, 0.5, 0.25])
        >>> 1 / op.range.cell_volume  # the scaling constant
        4.0
        >>> op(x)
        uniform_discr(0.0, 1.0, 4).element([0.0, 5.0, 2.0, 0.0])

        In higher dimensions, a sequence of index array-likes must be
        given, or a single sequence for a single point:

        >>> space = odl.uniform_discr([0, 0], [1, 1], (2, 3))
        >>> # Sample at the index (0, 2)
        >>> op = odl.WeightedSumSamplingOperator(space,
        ...                                      sampling_points=[0, 2])
        >>> x = op.domain.element([1])
        >>> # Insert the value 1 at index (0, 2)
        >>> op(x)
        uniform_discr([0.0, 0.0], [1.0, 1.0], (2, 3)).element(
            [[0.0, 0.0, 1.0],
             [0.0, 0.0, 0.0]]
        )
        >>> sampling_points = [[0, 1],  # indices (0, 2) and (1, 1)
        ...                    [2, 1]]
        >>> op = odl.WeightedSumSamplingOperator(space, sampling_points)
        >>> x = op.domain.element([1, 2])
        >>> op(x)
        uniform_discr([0.0, 0.0], [1.0, 1.0], (2, 3)).element(
            [[0.0, 0.0, 1.0],
             [0.0, 2.0, 0.0]]
        )
        """
        if not isinstance(range, TensorSpace):
            raise TypeError('`range` must be a `TensorSpace` instance, got '
                            '{!r}'.format(range))
        self.__sampling_points = _normalize_sampling_points(sampling_points,
                                                            range.ndim)
        # Convert a list of index arrays to linear index array
        indices_flat = np.ravel_multi_index(self.sampling_points,
                                            dims=range.shape)
        if np.isscalar(indices_flat):
            self._indices_flat = np.array([indices_flat], dtype=int)
        else:
            self._indices_flat = indices_flat

        self.__variant = str(variant).lower()
        if self.variant not in ('dirac', 'char_fun'):
            raise ValueError('`variant` {!r} not understood'.format(variant))

        domain = tensor_space(self.sampling_points[0].size, dtype=range.dtype)
        super().__init__(domain, range, linear=True)

    @property
    def variant(self):
        """Weighting scheme for the operator."""
        return self.__variant

    @property
    def sampling_points(self):
        """Indices where to sample the function."""
        return self.__sampling_points

    def _call(self, x, out=None):
        """Sum all values if indices are given multiple times."""
        y = np.bincount(self._indices_flat, weights=x,
                        minlength=self.range.size)

        if out is None:
            out = y.reshape(self.range.shape)
        else:
            out[:] = y.reshape(self.range.shape)

        if self.variant == 'dirac':
            weights = getattr(self.range, 'cell_volume', 1.0)
        elif self.variant == 'char_fun':
            weights = 1.0
        else:
            raise RuntimeError('The variant "{!r}" is not yet supported'
                               ''.format(self.variant))

        if weights != 1.0:
            out /= weights

        return out

    @property
    def adjoint(self):
        """Adjoint of this operator, a `SamplingOperator`.

        The ``'char_fun'`` variant of this operator corresponds to the
        ``'integrate'`` sampling operator, and ``'dirac'`` corresponds to
        ``'point_eval'``.

        Examples
        --------
        >>> space = odl.uniform_discr([-1, -1], [1, 1], shape=(2, 3))
        >>> # Point (0, 0) occurs twice
        >>> sampling_points = [[0, 1, 1, 0],
        ...                    [0, 1, 2, 0]]
        >>> op = odl.WeightedSumSamplingOperator(space, sampling_points,
        ...                                      variant='dirac')
        >>> y = op.range.element([[1, 2, 3],
        ...                       [4, 5, 6]])
        >>> op.adjoint(y)
        rn(4).element([1.0, 5.0, 6.0, 1.0])
        >>> x = op.domain.element([1, 2, 3, 4])
        >>> op.adjoint(op(x)).inner(x) - op(x).inner(op(x)) < 1e-10
        True
        >>> op = odl.WeightedSumSamplingOperator(space, sampling_points,
        ...                                      variant='char_fun')
        >>> op.adjoint(op(x)).inner(x) - op(x).inner(op(x)) < 1e-10
        True
        """
        if self.variant == 'dirac':
            variant = 'point_eval'
        elif self.variant == 'char_fun':
            variant = 'integrate'
        else:
            raise RuntimeError('The variant "{!r}" is not yet supported'
                               ''.format(self.variant))

        return SamplingOperator(self.range, self.sampling_points, variant)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.range, self.sampling_points]
        optargs = [('variant', self.variant, 'char_fun')]
        sig_str = signature_string(posargs, optargs, mod=['!r', ''],
                                   sep=[',\n', '', ',\n'])
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(sig_str))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class FlatteningOperator(Operator):

    """Operator that reshapes the object as a column vector.

    The operation performed by this operator is ::

        FlatteningOperator(x) == ravel(x)

    The range of this operator is always a `TensorSpace`, i.e., even if
    the domain is a discrete function space.
    """

    def __init__(self, domain, order=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `TensorSpace`
            Set of elements on which this operator acts.
        order : {None, 'C', 'F'}, optional
            If provided, flattening is performed in this order. ``'C'``
            means that that the last index is changing fastest, while in
            ``'F'`` ordering, the first index changes fastest.
            By default, ``domain.order`` is used.

        Examples
        --------
        >>> space = odl.uniform_discr([-1, -1], [1, 1], shape=(2, 3))
        >>> op = odl.FlatteningOperator(space)
        >>> op.range
        rn(6)
        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> op(x)
        rn(6).element([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> op = odl.FlatteningOperator(space, order='F')
        >>> op(x)
        rn(6).element([1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        """
        if not isinstance(domain, TensorSpace):
            raise TypeError('`domain` must be a `TensorSpace` instance, got '
                            '{!r}'.format(domain))

        if order is None:
            self.__order = domain.order
        else:
            self.__order = str(order).upper()
            if self.order not in ('C', 'F'):
                raise ValueError('`order` {!r} not understood'.format(order))

        range = tensor_space(domain.size, dtype=domain.dtype)
        super().__init__(domain, range, linear=True)

    def _call(self, x, out=None):
        """Flatten ``x``, writing to ``out``."""
        if out is None:
            out = np.ravel(x, order=self.order)
        else:
            out[:] = np.ravel(x, order=self.order)
        return out

    @property
    def order(self):
        """order of the flattening operation."""
        return self.__order

    @property
    def adjoint(self):
        """Adjoint of the flattening, a scaled version of the `inverse`.

        Examples
        --------
        >>> space = odl.uniform_discr([-1, -1], [1, 1], shape=(2, 4))
        >>> op = odl.FlatteningOperator(space)
        >>> y = op.range.element([1, 2, 3, 4, 5, 6, 7, 8])
        >>> 1 / space.cell_volume  # the scaling factor
        2.0
        >>> op.adjoint(y)
        uniform_discr([-1.0, -1.0], [1.0, 1.0], (2, 4)).element(
            [[2.0, 4.0, 6.0, 8.0],
             [10.0, 12.0, 14.0, 16.0]]
        )
        >>> x = space.element([[1, 2, 3, 4],
        ...                    [5, 6, 7, 8]])
        >>> op.adjoint(op(x)).inner(x) - op(x).inner(op(x)) < 1e-10
        True
        """
        scaling = getattr(self.domain, 'cell_volume', 1.0)
        return 1 / scaling * self.inverse

    @property
    def inverse(self):
        """Operator that reshapes to original shape.

        Examples
        --------
        >>> space = odl.uniform_discr([-1, -1], [1, 1], shape=(2, 4))
        >>> op = odl.FlatteningOperator(space)
        >>> y = op.range.element([1, 2, 3, 4, 5, 6, 7, 8])
        >>> op.inverse(y)
        uniform_discr([-1.0, -1.0], [1.0, 1.0], (2, 4)).element(
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0]]
        )
        >>> op = odl.FlatteningOperator(space, order='F')
        >>> op.inverse(y)
        uniform_discr([-1.0, -1.0], [1.0, 1.0], (2, 4)).element(
            [[1.0, 3.0, 5.0, 7.0],
             [2.0, 4.0, 6.0, 8.0]]
        )
        >>> op(op.inverse(y)) == y
        True
        """
        op = self
        scaling = getattr(self.domain, 'cell_volume', 1.0)

        class FlatteningOperatorInverse(Operator):

            """Inverse of `FlatteningOperator`.

            This operator reshapes a flat vector back to original shape::

                FlatteningOperatorInverse(x) == reshape(x, orig_shape)
            """

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(op.range, op.domain, linear=True)

            def _call(self, x, out=None):
                """Reshape ``x`` to nD shape, writing to ``out``."""
                if out is None:
                    return np.reshape(x.asarray(), self.range.shape,
                                      order=op.order)
                else:
                    out[:] = np.reshape(x.asarray(), self.range.shape,
                                        order=op.order)

            def adjoint(self):
                """Adjoint of this operator, a scaled `FlatteningOperator`."""
                return scaling * op

            def inverse(self):
                """Inverse of this operator."""
                return op

            def __repr__(self):
                """Return ``repr(self)``."""
                return '{!r}.inverse'.format(op)

            def __str__(self):
                """Return ``str(self)``."""
                return repr(self)

        return FlatteningOperatorInverse()

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.domain]
        optargs = [('order', self.order, self.domain.order)]
        sig_str = signature_string(posargs, optargs, mod=['!r', ''],
                                   sep=['', '', ',\n'])
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(sig_str))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
