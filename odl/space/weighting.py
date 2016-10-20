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

"""Weightings for finite-dimensional spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np
import scipy.linalg as linalg
from scipy.sparse.base import isspmatrix

from odl.space.base_ntuples import FnBaseVector
from odl.util.utility import array1d_repr, arraynd_repr


__all__ = ('MatrixWeightingBase', 'VectorWeightingBase', 'ConstWeightingBase',
           'NoWeightingBase',
           'CustomInnerProductBase', 'CustomNormBase', 'CustomDistBase')


class WeightingBase(object):

    """Abstract base class for weighting of finite-dimensional spaces.

    This class and its subclasses serve as a simple means to evaluate
    and compare weighted inner products, norms and metrics semantically
    rather than by identity on a pure function level.

    The functions are implemented similarly to `Operator`,
    but without extra type checks of input parameters - this is done in
    the callers of the `LinearSpace` instance where these
    functions are being used.
    """

    def __init__(self, impl, exponent=2.0, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        impl : string
            Specifier for the implementation backend
        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        dist_using_inner : bool, optional
            Calculate `dist` using the formula

                ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>``

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.

            This option can only be used if ``exponent`` is 2.0.

            Default: False.
        """
        self.__impl = str(impl).lower()
        self.__exponent = float(exponent)
        self.__dist_using_inner = bool(dist_using_inner)
        if self.exponent <= 0:
            raise ValueError('only positive exponents or inf supported, '
                             'got {}'.format(exponent))
        elif self.exponent != 2.0 and self.dist_using_inner:
            raise ValueError('`dist_using_inner` can only be used if the '
                             'exponent is 2.0')

    @property
    def impl(self):
        """Implementation backend of this weighting."""
        return self.__impl

    @property
    def exponent(self):
        """Exponent of this weighting."""
        return self.__exponent

    @property
    def dist_using_inner(self):
        """``True`` if the distance should be calculated using inner."""
        return self.__dist_using_inner

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equal : bool
            ``True`` if ``other`` is a the same weighting, ``False``
            otherwise.

        Notes
        -----
        This operation must be computationally cheap, i.e. no large
        arrays may be compared entry-wise. That is the task of the
        `equiv` method.
        """
        return (isinstance(other, WeightingBase) and
                self.impl == other.impl and
                self.exponent == other.exponent and
                self.dist_using_inner == other.dist_using_inner)

    def equiv(self, other):
        """Test if ``other`` is an equivalent weighting.

        Should be overridden, default tests for equality.

        Returns
        -------
        equivalent : bool
            ``True`` if ``other`` is a `WeightingBase` instance which
            yields the same result as this inner product for any
            input, ``False`` otherwise.
        """
        return self == other

    def inner(self, x1, x2):
        """Return the inner product of two elements.

        Parameters
        ----------
        x1, x2 : `LinearSpaceElement`
            Elements whose inner product is calculated.

        Returns
        -------
        inner : float or complex
            The inner product of the two provided elements.
        """
        raise NotImplementedError

    def norm(self, x):
        """Calculate the norm of an element.

        This is the standard implementation using `inner`.
        Subclasses should override it for optimization purposes.

        Parameters
        ----------
        x1 : `LinearSpaceElement`
            Element whose norm is calculated.

        Returns
        -------
        norm : float
            The norm of the element.
        """
        return float(np.sqrt(self.inner(x, x).real))

    def dist(self, x1, x2):
        """Calculate the distance between two elements.

        This is the standard implementation using `norm`.
        Subclasses should override it for optimization purposes.

        Parameters
        ----------
        x1, x2 : `LinearSpaceElement`
            Elements whose mutual distance is calculated.

        Returns
        -------
        dist : float
            The distance between the elements.
        """
        if self.dist_using_inner:
            dist_squared = (self.norm(x1) ** 2 + self.norm(x2) ** 2 -
                            2 * self.inner(x1, x2).real)
            if dist_squared < 0:  # Compensate for numerical error
                dist_squared = 0.0
            return float(np.sqrt(dist_squared))
        else:
            return self.norm(x1 - x2)


class MatrixWeightingBase(WeightingBase):

    """Weighting of a space by a matrix.

    The exact definition of the weighted inner product, norm and
    distance functions depend on the concrete space.

    The matrix must be Hermitian and posivive definite, otherwise it
    does not define an inner product or norm, respectively. This is not
    checked during initialization.
    """

    def __init__(self, matrix, impl, exponent=2.0, dist_using_inner=False,
                 **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        matrix :  `scipy.sparse.spmatrix` or 2-dim. `array-like`
            Square weighting matrix of the inner product
        impl : string
            Specifier for the implementation backend
        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
            If ``matrix`` is a sparse matrix, only 1.0, 2.0 and ``inf``
            are allowed.
        dist_using_inner : bool, optional
            Calculate `dist` using the following formula::

                ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.

            This option can only be used if ``exponent`` is 2.0.
        precomp_mat_pow : bool, optional
            If ``True``, precompute the matrix power ``W ** (1/p)``
            during initialization. This has no effect if ``exponent``
            is 1.0, 2.0 or ``inf``.

            Default: ``False``

        cache_mat_pow : bool, optional
            If ``True``, cache the matrix power ``W ** (1/p)``. This can
            happen either during initialization or in the first call to
            ``norm`` or ``dist``, resp. This has no effect if
            ``exponent`` is 1.0, 2.0 or ``inf``.

            Default: ``True``

        cache_mat_decomp : bool, optional
            If ``True``, cache the eigenbasis decomposition of the
            matrix. This can happen either during initialization or in
            the first call to ``norm`` or ``dist``, resp. This has no
            effect if ``exponent`` is 1.0, 2.0 or ``inf``.

            Default: ``False``

        Notes
        -----
        The matrix power ``W ** (1/p)`` is computed by eigenbasis
        decomposition::

            eigval, eigvec = scipy.linalg.eigh(matrix)
            mat_pow = (eigval ** p * eigvec).dot(eigvec.conj().T)

        Depending on the matrix size, this can be rather expensive.
        """
        # TODO: fix dead link `scipy.sparse.spmatrix`
        precomp_mat_pow = kwargs.pop('precomp_mat_pow', False)
        self._cache_mat_pow = bool(kwargs.pop('cache_mat_pow', True))
        self._cache_mat_decomp = bool(kwargs.pop('cache_mat_decomp', False))
        super().__init__(impl=impl, exponent=exponent,
                         dist_using_inner=dist_using_inner)

        # Check and set matrix
        if isspmatrix(matrix):
            self._matrix = matrix
        else:
            self._matrix = np.asarray(matrix)
            if self._matrix.dtype == object:
                raise ValueError('invalid matrix {}'.format(matrix))
            elif self._matrix.ndim != 2:
                raise ValueError('matrix {} is {}-dimensional instead of '
                                 '2-dimensional'
                                 ''.format(matrix, self._matrix.ndim))

        if self._matrix.shape[0] != self._matrix.shape[1]:
            raise ValueError('matrix has shape {}, expected a square matrix'
                             ''.format(self._matrix.shape))

        if (self.matrix_issparse and
                self.exponent not in (1.0, 2.0, float('inf'))):
            raise NotImplementedError('sparse matrices only supported for '
                                      'exponent 1.0, 2.0 or `inf`')

        # Compute the power and decomposition if desired
        self._eigval = self._eigvec = None
        if self.exponent in (1.0, float('inf')):
            self._mat_pow = self.matrix
        elif precomp_mat_pow and self.exponent != 2.0:
            eigval, eigvec = self.matrix_decomp()
            if self._cache_mat_decomp:
                self._eigval, self._eigvec = eigval, eigvec
                eigval_pow = eigval ** (1.0 / self.exponent)
            else:
                eigval_pow = eigval
                eigval_pow **= 1.0 / self.exponent
            self._mat_pow = (eigval_pow * eigvec).dot(eigvec.conj().T)
        else:
            self._mat_pow = None

    @property
    def matrix(self):
        """Weighting matrix of this inner product."""
        return self._matrix

    @property
    def matrix_issparse(self):
        """Whether the representing matrix is sparse or not."""
        return isspmatrix(self.matrix)

    def is_valid(self):
        """Test if the matrix is positive definite Hermitian.

        If the matrix decomposition is available, this test checks
        if all eigenvalues are positive.
        Otherwise, the test tries to calculate a Cholesky decomposition,
        which can be very time-consuming for large matrices. Sparse
        matrices are not supported.
        """
        if self.matrix_issparse:
            raise NotImplementedError('validation not supported for sparse '
                                      'matrices')
        elif self._eigval is not None:
            return np.all(np.greater(self._eigval, 0))
        else:
            try:
                np.linalg.cholesky(self.matrix)
                return np.array_equal(self.matrix, self.matrix.conj().T)
            except np.linalg.LinAlgError:
                return False

    def matrix_decomp(self, cache=None):
        """Compute a Hermitian eigenbasis decomposition of the matrix.

        Parameters
        ----------
        cache : bool or None, optional
            If ``True``, store the decomposition internally. For None,
            the ``cache_mat_decomp`` from class initialization is used.

        Returns
        -------
        eigval : `numpy.ndarray`
            One-dimensional array of eigenvalues. Its length is equal
            to the number of matrix rows.
        eigvec : `numpy.ndarray`
            Two-dimensional array of eigenvectors. It has the same shape
            as the decomposed matrix.

        See Also
        --------
        scipy.linalg.decomp.eigh :
            Implementation of the decomposition. Standard parameters
            are used here.

        Raises
        ------
        NotImplementedError
            if the matrix is sparse (not supported by scipy 0.17)
        """
        # TODO: fix dead link `scipy.linalg.decomp.eigh`
        if self.matrix_issparse:
            raise NotImplementedError('sparse matrix not supported')

        if cache is None:
            cache = self._cache_mat_decomp

        if self._eigval is None or self._eigvec is None:
            eigval, eigvec = linalg.eigh(self.matrix)
            if cache:
                self._eigval = eigval
                self._eigvec = eigvec
        else:
            eigval, eigvec = self._eigval, self._eigvec

        return eigval, eigvec

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if other is a `MatrixWeightingBase` instance
            with **identical** matrix, ``False`` otherwise.

        See Also
        --------
        equiv : test for equivalent inner products
        """
        if other is self:
            return True

        return (super().__eq__(other) and
                self.matrix is getattr(other, 'matrix', None))

    def equiv(self, other):
        """Test if other is an equivalent weighting.

        Returns
        -------
        equivalent : bool
            ``True`` if other is a `WeightingBase` instance with the same
            `WeightingBase.impl`, which yields the same result as this
            weighting for any input, ``False`` otherwise. This is checked
            by entry-wise comparison of matrices/vectors/constants.
        """
        # Optimization for equality
        if self == other:
            return True

        elif self.exponent != getattr(other, 'exponent', -1):
            return False

        elif isinstance(other, MatrixWeightingBase):
            if self.matrix.shape != other.matrix.shape:
                return False

            if self.matrix_issparse:
                if other.matrix_issparse:
                    # Optimization for different number of nonzero elements
                    if self.matrix.nnz != other.matrix.nnz:
                        return False
                    else:
                        # Most efficient out-of-the-box comparison
                        return (self.matrix != other.matrix).nnz == 0
                else:  # Worst case: compare against dense matrix
                    return np.array_equal(self.matrix.todense(), other.matrix)

            else:  # matrix of `self` is dense
                if other.matrix_issparse:
                    return np.array_equal(self.matrix, other.matrix.todense())
                else:
                    return np.array_equal(self.matrix, other.matrix)

        elif isinstance(other, VectorWeightingBase):
            if self.matrix_issparse:
                return (np.array_equiv(self.matrix.diagonal(),
                                       other.vector) and
                        np.array_equal(self.matrix.asformat('dia').offsets,
                                       np.array([0])))
            else:
                return np.array_equal(
                    self.matrix, other.vector * np.eye(self.matrix.shape[0]))

        elif isinstance(other, ConstWeightingBase):
            if self.matrix_issparse:
                return (np.array_equiv(self.matrix.diagonal(), other.const) and
                        np.array_equal(self.matrix.asformat('dia').offsets,
                                       np.array([0])))
            else:
                return np.array_equal(
                    self.matrix, other.const * np.eye(self.matrix.shape[0]))
        else:
            return False

    @property
    def repr_part(self):
        """Return a string usable in a space's ``__repr__`` method."""
        if self.matrix_issparse:
            part = 'weight={}'.format(self.matrix)
        else:
            part = 'weight={}'.format(arraynd_repr(self.matrix, nprint=10))
        if self.exponent != 2.0:
            part += ', exponent={}'.format(self.exponent)
        if self.dist_using_inner:
            part += ', dist_using_inner=True'
        return part

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.matrix_issparse:
            inner_fstr = ('<{shape} sparse matrix, format {fmt!r}, {nnz} '
                          'stored entries>')
            fmt = self.matrix.format
            nnz = self.matrix.nnz
            if self.exponent != 2.0:
                inner_fstr += ', exponent={ex}'
            if self.dist_using_inner:
                inner_fstr += ', dist_using_inner=True'
        else:
            inner_fstr = '\n{matrix!r}'
            fmt = ''
            nnz = 0
            if self.exponent != 2.0:
                inner_fstr += ',\nexponent={ex}'
            if self.dist_using_inner:
                inner_fstr += ',\ndist_using_inner=True'
            else:
                inner_fstr += '\n'

        inner_str = inner_fstr.format(shape=self.matrix.shape, fmt=fmt,
                                      nnz=nnz, ex=self.exponent,
                                      matrix=self.matrix)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        if self.exponent == 2.0:
            return 'Weighting: matrix =\n{}'.format(self.matrix)
        else:
            return 'Weighting: p = {}, matrix =\n{}'.format(self.exponent,
                                                            self.matrix)


class VectorWeightingBase(WeightingBase):

    """Weighting of a space by a vector.

    The exact definition of the weighted inner product, norm and
    distance functions depend on the concrete space.

    The vector may only have positive entries, otherwise it does not
    define an inner product or norm, respectively. This is not checked
    during initialization.
    """

    def __init__(self, vector, impl, exponent=2.0, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        vector : 1-dim. `array-like`
            Weighting vector of the inner product.
        impl : string
            Specifier for the implementation backend.
        exponent : positive float
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
            If ``matrix`` is a sparse matrix, only 1.0, 2.0 and ``inf``
            are allowed.
        dist_using_inner : bool, optional
            Calculate `dist` using the formula

                ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>``.

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.

            This option can only be used if ``exponent`` is 2.0.
        """
        super().__init__(impl=impl, exponent=exponent,
                         dist_using_inner=dist_using_inner)

        if isinstance(vector, FnBaseVector):
            self._vector = vector
        else:
            self._vector = np.asarray(vector)

        if self.vector.dtype == object:
            raise ValueError('invalid vector {}'.format(vector))
        elif self.vector.ndim != 1:
            raise ValueError('vector {} is {}-dimensional instead of '
                             '1-dimensional'
                             ''.format(vector, self._vector.ndim))

    @property
    def vector(self):
        """Weighting vector of this inner product."""
        return self._vector

    def is_valid(self):
        """Test if the vector is a valid weight, i.e. positive."""
        return np.all(np.greater(self.vector, 0))

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if other is a `VectorWeightingBase` instance with
            **identical** vector, ``False`` otherwise.

        See Also
        --------
        equiv : test for equivalent inner products
        """
        if other is self:
            return True

        return (super().__eq__(other) and
                self.vector is getattr(other, 'vector', None))

    def equiv(self, other):
        """Test if other is an equivalent weighting.

        Returns
        -------
        equivalent : bool
            ``True`` if other is a `WeightingBase` instance with the same
            `WeightingBase.impl`, which yields the same result as this
            weighting for any input, ``False`` otherwise. This is checked
            by entry-wise comparison of matrices/vectors/constants.
        """
        # Optimization for equality
        if self == other:
            return True
        elif (not isinstance(other, WeightingBase) or
              self.exponent != other.exponent):
            return False
        elif isinstance(other, MatrixWeightingBase):
            return other.equiv(self)
        elif isinstance(other, ConstWeightingBase):
            return np.array_equiv(self.vector, other.const)
        else:
            return np.array_equal(self.vector, other.vector)

    @property
    def repr_part(self):
        """String usable in a space's ``__repr__`` method."""
        part = 'weight={}'.format(array1d_repr(self.vector, nprint=10))
        if self.exponent != 2.0:
            part += ', exponent={}'.format(self.exponent)
        if self.dist_using_inner:
            part += ', dist_using_inner=True'
        return part

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_fstr = '{vector!r}'
        if self.exponent != 2.0:
            inner_fstr += ', exponent={ex}'
        if self.dist_using_inner:
            inner_fstr += ', dist_using_inner=True'

        inner_str = inner_fstr.format(vector=self.vector, ex=self.exponent)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        if self.exponent == 2.0:
            return 'Weighting: vector =\n{}'.format(self.vector)
        else:
            return 'Weighting: p = {}, vector =\n{}'.format(self.exponent,
                                                            self.vector)


class ConstWeightingBase(WeightingBase):

    """Weighting of a space by a constant.

    """

    def __init__(self, constant, impl, exponent=2.0, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        constant : positive float
            Weighting constant of the inner product.
        impl : string
            Specifier for the implementation backend.
        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        dist_using_inner : bool, optional
            Calculate `dist` using the formula

                ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>``.

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.

            This option can only be used if ``exponent`` is 2.0.
        """
        super().__init__(impl=impl, exponent=exponent,
                         dist_using_inner=dist_using_inner)
        self._const = float(constant)
        if self.const <= 0:
            raise ValueError('expected positive constant, got {}'
                             ''.format(constant))
        if not np.isfinite(self.const):
            raise ValueError('`constant` {} is invalid'.format(constant))

    @property
    def const(self):
        """Weighting constant of this inner product."""
        return self._const

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equal : bool
            ``True`` if other is a `ConstWeightingBase` instance with the
            same constant, ``False`` otherwise.
        """
        if other is self:
            return True

        return (super().__eq__(other) and
                self.const == getattr(other, 'const', None))

    def equiv(self, other):
        """Test if other is an equivalent weighting.

        Returns
        -------
        equivalent : bool
            ``True`` if other is a `WeightingBase` instance with the same
            `WeightingBase.impl`, which yields the same result as this
            weighting for any input, ``False`` otherwise. This is checked
            by entry-wise comparison of matrices/vectors/constants.
        """
        if isinstance(other, ConstWeightingBase):
            return self == other
        elif isinstance(other, (VectorWeightingBase, MatrixWeightingBase)):
            return other.equiv(self)
        else:
            return False

    @property
    def repr_part(self):
        """String usable in a space's ``__repr__`` method."""
        sep = ''
        if self.const != 1.0:
            part = 'weight={:.4}'.format(self.const)
            sep = ', '
        else:
            part = ''

        if self.exponent != 2.0:
            part += sep + 'exponent={}'.format(self.exponent)
            sep = ', '
        if self.dist_using_inner:
            part += sep + 'dist_using_inner=True'
        return part

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_fstr = '{}'
        if self.exponent != 2.0:
            inner_fstr += ', exponent={ex}'
        if self.dist_using_inner:
            inner_fstr += ', dist_using_inner=True'

        inner_str = inner_fstr.format(self.const, ex=self.exponent)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        if self.exponent == 2.0:
            return 'Weighting: const = {:.4}'.format(self.const)
        else:
            return 'Weighting: p = {}, const = {:.4}'.format(
                self.exponent, self.const)


class NoWeightingBase(ConstWeightingBase):

    """Weighting with constant 1."""

    def __init__(self, impl, exponent=2.0, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        exponent : positive float
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        impl : string
            Specifier for the implementation backend.
        dist_using_inner : bool, optional
            Calculate `dist` using the formula

                ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>``.

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.

            This option can only be used if ``exponent`` is 2.0.
        """
        # Support singleton pattern for subclasses
        if not hasattr(self, '_initialized'):
            ConstWeightingBase.__init__(
                self, constant=1.0, impl=impl, exponent=exponent,
                dist_using_inner=dist_using_inner)
            self._initialized = True

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_fstr = ''
        if self.exponent != 2.0:
            inner_fstr += ', exponent={ex}'
        if self.dist_using_inner:
            inner_fstr += ', dist_using_inner=True'
        inner_str = inner_fstr.format(ex=self.exponent).lstrip(', ')

        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        if self.exponent == 2.0:
            return 'NoWeighting'
        else:
            return 'NoWeighting: p = {}'.format(self.exponent)


class CustomInnerProductBase(WeightingBase):

    """Class for handling a user-specified inner product."""

    def __init__(self, inner, impl, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        inner : callable
            The inner product implementation. It must accept two
            `LinearSpaceElement` arguments, return an element from
            their space's field (real or complex number) and
            satisfy the following conditions for all space elements
            ``x, y, z`` and scalars ``s``:

            - ``<x, y> = conj(<y, x>)``
            - ``<s*x + y, z> = s * <x, z> + <y, z>``
            - ``<x, x> = 0``  if and only if  ``x = 0``

        impl : string
            Specifier for the implementation backend.
        dist_using_inner : bool, optional
            Calculate `dist` using the formula

                ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>``.

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.

            Can only be used if ``exponent`` is 2.0.
        """
        super().__init__(impl=impl, exponent=2.0,
                         dist_using_inner=dist_using_inner)

        if not callable(inner):
            raise TypeError('`inner` {!r} is not callable'
                            ''.format(inner))
        self._inner = inner

    @property
    def inner(self):
        """Custom inner product of this instance.."""
        return self._inner

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equal : bool
            ``True`` if other is a `CustomInnerProductBase`
            instance with the same inner product, ``False`` otherwise.
        """
        return super().__eq__(other) and self.inner == other.inner

    @property
    def repr_part(self):
        """String usable in a space's ``__repr__`` method."""
        part = 'inner={}'.format(self.inner)
        if self.exponent != 2.0:
            part += ', exponent={}'.format(self.exponent)
        if self.dist_using_inner:
            part += ', dist_using_inner=True'
        return part

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_fstr = '{!r}'
        if self.dist_using_inner:
            inner_fstr += ', dist_using_inner=True'

        inner_str = inner_fstr.format(self.inner)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class CustomNormBase(WeightingBase):

    """Class for handling a user-specified norm.

    Note that this removes ``inner``.
    """

    def __init__(self, norm, impl):
        """Initialize a new instance.

        Parameters
        ----------
        norm : callable
            The norm implementation. It must accept a
            `LinearSpaceElement` argument, return a float and satisfy
            the following conditions for all space elements
            ``x, y`` and scalars ``s``:

            - ``||x|| >= 0``
            - ``||x|| = 0``  if and only if  ``x = 0``
            - ``||s * x|| = |s| * ||x||``
            - ``||x + y|| <= ||x|| + ||y||``
        impl : string
            Specifier for the implementation backend
        """
        super().__init__(impl=impl, exponent=1.0, dist_using_inner=False)

        if not callable(norm):
            raise TypeError('`norm` {!r} is not callable'
                            ''.format(norm))
        self._norm = norm

    def inner(self, x1, x2):
        """Inner product is not defined for custom distance."""
        raise NotImplementedError

    @property
    def norm(self):
        """Custom norm of this instance.."""
        return self._norm

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equal : bool
            ``True`` if other is a `CustomNormBase` instance with the same
            norm, ``False`` otherwise.
        """
        return super().__eq__(other) and self.norm == other.norm

    @property
    def repr_part(self):
        """Return a string usable in a space's ``__repr__`` method."""
        part = 'norm={}'.format(self.norm)
        if self.exponent != 2.0:
            part += ', exponent={}'.format(self.exponent)
        if self.dist_using_inner:
            part += ', dist_using_inner=True'
        return part

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_fstr = '{!r}'
        inner_str = inner_fstr.format(self.norm)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class CustomDistBase(WeightingBase):

    """Class for handling a user-specified distance.

    Note that this removes ``inner`` and ``norm``.
    """

    def __init__(self, dist, impl):
        """Initialize a new instance.

        Parameters
        ----------
        dist : callable
            The distance function defining a metric on a `LinearSpace`.
            It must accept two `LinearSpaceElement` arguments, return a
            float and and fulfill the following mathematical conditions
            for any three space elements ``x, y, z``:

            - ``dist(x, y) >= 0``
            - ``dist(x, y) = 0``  if and only if  ``x = y``
            - ``dist(x, y) = dist(y, x)``
            - ``dist(x, y) <= dist(x, z) + dist(z, y)``
        impl : string
            Specifier for the implementation backend
        """
        super().__init__(impl=impl, exponent=1.0, dist_using_inner=False)

        if not callable(dist):
            raise TypeError('`dist` {!r} is not callable'
                            ''.format(dist))
        self._dist = dist

    @property
    def dist(self):
        """Custom distance of this instance.."""
        return self._dist

    def inner(self, x1, x2):
        """Inner product is not defined for custom distance."""
        raise NotImplementedError

    def norm(self, x):
        """Norm is not defined for custom distance."""
        raise NotImplementedError

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equal : bool
            ``True`` if other is a `CustomDistBase` instance with the same
            dist, ``False`` otherwise.
        """
        return super().__eq__(other) and self.dist == other.dist

    @property
    def repr_part(self):
        """Return a string usable in a space's ``__repr__`` method."""
        part = 'dist={}'.format(self.dist)
        if self.exponent != 2.0:
            part += ', exponent={}'.format(self.exponent)
        if self.dist_using_inner:
            part += ', dist_using_inner=True'
        return part

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_fstr = '{!r}'
        inner_str = inner_fstr.format(self.dist)
        return '{}({})'.format(self.__class__.__name__, inner_str)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
