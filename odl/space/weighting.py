﻿# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Weightings for finite-dimensional spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np
import scipy
import scipy.linalg as linalg
from scipy.sparse.base import isspmatrix

from odl.space.base_tensors import Tensor
from odl.util import array_str, signature_string, indent


__all__ = ('MatrixWeighting', 'ArrayWeighting', 'ConstWeighting',
           'NoWeighting',
           'CustomInner', 'CustomNorm', 'CustomDist')


class Weighting(object):

    """Abstract base class for weighting of finite-dimensional spaces.

    This class and its subclasses serve as a simple means to evaluate
    and compare weighted inner products, norms and metrics semantically
    rather than by identity on a pure function level.

    The functions are implemented similarly to `Operator`,
    but without extra type checks of input parameters - this is done in
    the callers of the `LinearSpace` instance where these
    functions are being used.
    """

    def __init__(self, impl, exponent=2.0):
        """Initialize a new instance.

        Parameters
        ----------
        impl : string
            Specifier for the implementation backend
        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        """
        self.__impl = str(impl).lower()
        self.__exponent = float(exponent)
        if self.exponent <= 0:
            raise ValueError('only positive exponents or inf supported, '
                             'got {}'.format(exponent))

    @property
    def impl(self):
        """Implementation backend of this weighting."""
        return self.__impl

    @property
    def exponent(self):
        """Exponent of this weighting."""
        return self.__exponent

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
        return (isinstance(other, Weighting) and
                self.impl == other.impl and
                self.exponent == other.exponent)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), self.impl, self.exponent))

    def equiv(self, other):
        """Test if ``other`` is an equivalent weighting.

        Should be overridden, default tests for equality.

        Returns
        -------
        equivalent : bool
            ``True`` if ``other`` is a `Weighting` instance which
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
        return self.norm(x1 - x2)


class MatrixWeighting(Weighting):

    """Weighting of a space by a matrix.

    The exact definition of the weighted inner product, norm and
    distance functions depend on the concrete space.

    The matrix must be Hermitian and posivive definite, otherwise it
    does not define an inner product or norm, respectively. This is not
    checked during initialization.
    """

    def __init__(self, matrix, impl, exponent=2.0, **kwargs):
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
        super().__init__(impl=impl, exponent=exponent)

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

        if (scipy.sparse.isspmatrix(self.matrix) and
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

    def is_valid(self):
        """Test if the matrix is positive definite Hermitian.

        If the matrix decomposition is available, this test checks
        if all eigenvalues are positive.
        Otherwise, the test tries to calculate a Cholesky decomposition,
        which can be very time-consuming for large matrices. Sparse
        matrices are not supported.
        """
        if scipy.sparse.isspmatrix(self.matrix):
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
        if scipy.sparse.isspmatrix(self.matrix):
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
            ``True`` if other is a `MatrixWeighting` instance
            with **identical** matrix, ``False`` otherwise.

        See Also
        --------
        equiv : test for equivalent inner products
        """
        if other is self:
            return True

        return (super().__eq__(other) and
                self.matrix is getattr(other, 'matrix', None))

    def __hash__(self):
        """Return ``hash(self)``."""
        # TODO: Better hash for matrix?
        return hash((super().__hash__(), self.matrix.tobytes()))

    def equiv(self, other):
        """Test if other is an equivalent weighting.

        Returns
        -------
        equivalent : bool
            ``True`` if ``other`` is a `Weighting` instance with the same
            `Weighting.impl`, which yields the same result as this
            weighting for any input, ``False`` otherwise. This is checked
            by entry-wise comparison of matrices/arrays/constants.
        """
        # Optimization for equality
        if self == other:
            return True

        elif self.exponent != getattr(other, 'exponent', -1):
            return False

        elif isinstance(other, MatrixWeighting):
            if self.matrix.shape != other.matrix.shape:
                return False

            if scipy.sparse.isspmatrix(self.matrix):
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

        elif isinstance(other, ArrayWeighting):
            if scipy.sparse.isspmatrix(self.matrix):
                return (np.array_equiv(self.matrix.diagonal(),
                                       other.array) and
                        np.array_equal(self.matrix.asformat('dia').offsets,
                                       np.array([0])))
            else:
                return np.array_equal(
                    self.matrix, other.array * np.eye(self.matrix.shape[0]))

        elif isinstance(other, ConstWeighting):
            if scipy.sparse.isspmatrix(self.matrix):
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
        if scipy.sparse.isspmatrix(self.matrix):
            part = 'weighting={}'.format(self.matrix)
        else:
            part = 'weighting={}'.format(array_str(self.matrix, nprint=10))
        if self.exponent != 2.0:
            part += ', exponent={}'.format(self.exponent)
        return part

    def __repr__(self):
        """Return ``repr(self)``."""
        if scipy.sparse.isspmatrix(self.matrix):
            inner_fstr = ('<{shape} sparse matrix, format {fmt!r}, {nnz} '
                          'stored entries>')
            fmt = self.matrix.format
            nnz = self.matrix.nnz
            if self.exponent != 2.0:
                inner_fstr += ', exponent={ex}'
        else:
            inner_fstr = '\n{matrix!r}'
            fmt = ''
            nnz = 0
            if self.exponent != 2.0:
                inner_fstr += ',\nexponent={ex}'
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


class ArrayWeighting(Weighting):

    """Weighting of a space by an array.

    The exact definition of the weighted inner product, norm and
    distance functions depend on the concrete space.

    The array may only have positive entries, otherwise it does not
    define an inner product or norm, respectively. This is not checked
    during initialization.
    """

    def __init__(self, array, impl, exponent=2.0):
        """Initialize a new instance.

        Parameters
        ----------
        array : `array-like`
            Weighting array of inner product, norm and distance.
            Native `Tensor` instances are stored as-is without copying.
        impl : string
            Specifier for the implementation backend.
        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        """
        super().__init__(impl=impl, exponent=exponent)

        # We store our "own" data structures as-is to retain Numpy
        # compatibility while avoiding copies. Other things are run through
        # numpy.asarray.
        if isinstance(array, Tensor):
            self.__array = array
        else:
            self.__array = np.asarray(array)

        if self.array.dtype == object:
            raise ValueError('invalid array {}'.format(array))

    @property
    def array(self):
        """Weighting array of this instance."""
        return self.__array

    def is_valid(self):
        """Return True if the array is a valid weight, i.e. positive."""
        return np.all(np.greater(self.array, 0))

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is an `ArrayWeighting` instance with
            **identical** array, False otherwise.

        See Also
        --------
        equiv : test for equivalent inner products
        """
        if other is self:
            return True

        return (super().__eq__(other) and
                self.array is getattr(other, 'array', None))

    def __hash__(self):
        """Return ``hash(self)``."""
        # TODO: Better hash for array?
        return hash((super().__hash__(), self.array.tobytes()))

    def equiv(self, other):
        """Return True if other is an equivalent weighting.

        Returns
        -------
        equivalent : bool
            ``True`` if ``other`` is a `Weighting` instance with the same
            `Weighting.impl`, which yields the same result as this
            weighting for any input, ``False`` otherwise. This is checked
            by entry-wise comparison of arrays/constants.
        """
        # Optimization for equality
        if self == other:
            return True
        elif (not isinstance(other, Weighting) or
              self.exponent != other.exponent):
            return False
        elif isinstance(other, MatrixWeighting):
            return other.equiv(self)
        elif isinstance(other, ConstWeighting):
            return np.array_equiv(self.array, other.const)
        else:
            return np.array_equal(self.array, other.array)

    @property
    def repr_part(self):
        """String usable in a space's ``__repr__`` method."""
        optargs = [('weighting', array_str(self.array, nprint=10), ''),
                   ('exponent', self.exponent, 2.0)]
        return signature_string([], optargs, sep=[',\n', ', ', ',\n'],
                                mod=[[], ['!s', '']])

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [array_str(self.array)]
        optargs = [('exponent', self.exponent, 2.0)]
        inner_str = signature_string(posargs, optargs,
                                     sep=[', ', ', ', ',\n'],
                                     mod=['!s', ''])
        return '{}(\n{}\n)'.format(self.__class__.__name__, indent(inner_str))

    __str__ = __repr__


class ConstWeighting(Weighting):

    """Weighting of a space by a constant."""

    def __init__(self, const, impl, exponent=2.0):
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
        """
        super().__init__(impl=impl, exponent=exponent)
        self.__const = float(const)
        if self.const <= 0:
            raise ValueError('expected positive constant, got {}'
                             ''.format(const))
        if not np.isfinite(self.const):
            raise ValueError('`const` {} is invalid'.format(const))

    @property
    def const(self):
        """Weighting constant of this inner product."""
        return self.__const

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equal : bool
            ``True`` if ``other`` is a `ConstWeighting` instance with the
            same constant, ``False`` otherwise.
        """
        if other is self:
            return True

        return (super().__eq__(other) and
                self.const == getattr(other, 'const', None))

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((super().__hash__(), self.const))

    def equiv(self, other):
        """Test if other is an equivalent weighting.

        Returns
        -------
        equivalent : bool
            ``True`` if other is a `Weighting` instance with the same
            `Weighting.impl`, which yields the same result as this
            weighting for any input, ``False`` otherwise. This is checked
            by entry-wise comparison of matrices/arrays/constants.
        """
        if isinstance(other, ConstWeighting):
            return self == other
        elif isinstance(other, (ArrayWeighting, MatrixWeighting)):
            return other.equiv(self)
        else:
            return False

    @property
    def repr_part(self):
        """String usable in a space's ``__repr__`` method."""
        optargs = [('weighting', self.const, 1.0),
                   ('exponent', self.exponent, 2.0)]
        return signature_string([], optargs,
                                mod=[[], [':.4', '']])

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.const]
        optargs = [('exponent', self.exponent, 2.0)]
        return '{}({})'.format(self.__class__.__name__,
                               signature_string(posargs, optargs))

    __str__ = __repr__


class NoWeighting(ConstWeighting):

    """Weighting with constant 1."""

    def __init__(self, impl, exponent=2.0):
        """Initialize a new instance.

        Parameters
        ----------
        impl : string
            Specifier for the implementation backend.
        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        """
        # Support singleton pattern for subclasses
        if not hasattr(self, '_initialized'):
            ConstWeighting.__init__(self, const=1.0, impl=impl,
                                    exponent=exponent)
            self._initialized = True

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = []
        optargs = [('exponent', self.exponent, 2.0)]
        return '{}({})'.format(self.__class__.__name__,
                               signature_string(posargs, optargs))

    __str__ = __repr__


class CustomInner(Weighting):

    """Class for handling a user-specified inner product."""

    def __init__(self, inner, impl):
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
        """
        super().__init__(impl=impl, exponent=2.0)

        if not callable(inner):
            raise TypeError('`inner` {!r} is not callable'
                            ''.format(inner))
        self.__inner = inner

    @property
    def inner(self):
        """Custom inner product of this instance.."""
        return self.__inner

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equal : bool
            ``True`` if other is a `CustomInner`
            instance with the same inner product, ``False`` otherwise.
        """
        return super().__eq__(other) and self.inner == other.inner

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((super().__hash__(), self.inner))

    @property
    def repr_part(self):
        """String usable in a space's ``__repr__`` method."""
        optargs = [('inner', self.inner, '')]
        return signature_string([], optargs, mod=[[], ['!r']])

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.inner]
        inner_str = signature_string(posargs, [], mod=['!r'])
        return '{}({})'.format(self.__class__.__name__, inner_str)


class CustomNorm(Weighting):

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
        super().__init__(impl=impl, exponent=1.0)

        if not callable(norm):
            raise TypeError('`norm` {!r} is not callable'
                            ''.format(norm))
        self.__norm = norm

    def inner(self, x1, x2):
        """Inner product is not defined for custom distance."""
        raise NotImplementedError('`inner` not defined for custom norm')

    @property
    def norm(self):
        """Custom norm of this instance.."""
        return self.__norm

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equal : bool
            ``True`` if other is a `CustomNorm` instance with the same
            norm, ``False`` otherwise.
        """
        return super().__eq__(other) and self.norm == other.norm

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((super().__hash__(), self.norm))

    @property
    def repr_part(self):
        """Return a string usable in a space's ``__repr__`` method."""
        optargs = [('norm', self.norm, ''),
                   ('exponent', self.exponent, 2.0)]
        return signature_string([], optargs, mod=[[], ['!r', '']])

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.norm]
        optargs = [('exponent', self.exponent, 2.0)]
        inner_str = signature_string(posargs, optargs, mod=['!r', ''])
        return '{}({})'.format(self.__class__.__name__, inner_str)


class CustomDist(Weighting):

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
        super().__init__(impl=impl, exponent=1.0)

        if not callable(dist):
            raise TypeError('`dist` {!r} is not callable'
                            ''.format(dist))
        self.__dist = dist

    @property
    def dist(self):
        """Custom distance of this instance.."""
        return self.__dist

    def inner(self, x1, x2):
        """Inner product is not defined for custom distance."""
        raise NotImplementedError('`inner` not defined for custom distance')

    def norm(self, x):
        """Norm is not defined for custom distance."""
        raise NotImplementedError('`norm` not defined for custom distance')

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equal : bool
            ``True`` if other is a `CustomDist` instance with the same
            dist, ``False`` otherwise.
        """
        return super().__eq__(other) and self.dist == other.dist

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((super().__hash__(), self.dist))

    @property
    def repr_part(self):
        """Return a string usable in a space's ``__repr__`` method."""
        optargs = [('dist', self.dist, '')]
        return signature_string([], optargs, mod=['', '!r'])

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.dist]
        optargs = []
        inner_str = signature_string(posargs, optargs, mod=['!r', ''])
        return '{}({})'.format(self.__class__.__name__, inner_str)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
