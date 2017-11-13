# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Weightings for finite-dimensional spaces."""

from __future__ import print_function, division, absolute_import
from builtins import object
import numpy as np

from odl.space.base_tensors import TensorSpace
from odl.util import (
    array_str, signature_string, indent, fast_1d_tensor_mult, writable_array)


__all__ = ('MatrixWeighting', 'ArrayWeighting', 'ConstWeighting',
           'CustomInner', 'CustomNorm', 'CustomDist', 'adjoint_weightings')


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
        # Lazy import to improve `import odl` time
        import scipy.sparse

        # TODO: fix dead link `scipy.sparse.spmatrix`
        precomp_mat_pow = kwargs.pop('precomp_mat_pow', False)
        self._cache_mat_pow = bool(kwargs.pop('cache_mat_pow', True))
        self._cache_mat_decomp = bool(kwargs.pop('cache_mat_decomp', False))
        super(MatrixWeighting, self).__init__(impl=impl, exponent=exponent)

        # Check and set matrix
        if scipy.sparse.isspmatrix(matrix):
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
        # Lazy import to improve `import odl` time
        import scipy.sparse

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
        # Lazy import to improve `import odl` time
        import scipy.linalg
        import scipy.sparse

        # TODO: fix dead link `scipy.linalg.decomp.eigh`
        if scipy.sparse.isspmatrix(self.matrix):
            raise NotImplementedError('sparse matrix not supported')

        if cache is None:
            cache = self._cache_mat_decomp

        if self._eigval is None or self._eigvec is None:
            eigval, eigvec = scipy.linalg.eigh(self.matrix)
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

        return (super(MatrixWeighting, self).__eq__(other) and
                self.matrix is getattr(other, 'matrix', None))

    def __hash__(self):
        """Return ``hash(self)``."""
        # TODO: Better hash for matrix?
        return hash((super(MatrixWeighting, self).__hash__(),
                     self.matrix.tobytes()))

    @property
    def repr_part(self):
        """Return a string usable in a space's ``__repr__`` method."""
        # Lazy import to improve `import odl` time
        import scipy.sparse

        if scipy.sparse.isspmatrix(self.matrix):
            optargs = [('matrix', str(self.matrix), '')]
        else:
            optargs = [('matrix', array_str(self.matrix, nprint=10), '')]

        optargs.append(('exponent', self.exponent, 2.0))
        return signature_string([], optargs, mod=[[], ['!s', ':.4']])

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.matrix_issparse:
            posargs = ['<{} sparse matrix, format {}, {} nonzero entries>'
                       ''.format(self.matrix.shape, self.matrix.format,
                                 self.matrix.nnz)]
        else:
            posargs = [array_str(self.matrix, nprint=10)]

        optargs = [('exponent', self.exponent, 2.0)]
        inner_str = signature_string(posargs, optargs, sep=',\n',
                                     mod=['!s', ''])
        return '{}(\n{}\n)'.format(self.__class__.__name__, indent(inner_str))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


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
        super(ArrayWeighting, self).__init__(impl=impl, exponent=exponent)

        # We apply array duck-typing to allow all kinds of Numpy-array-like
        # data structures without change
        array_attrs = ('shape', 'dtype', 'itemsize')
        if (all(hasattr(array, attr) for attr in array_attrs) and
                not isinstance(array, TensorSpace)):
            self.__array = array
        else:
            raise TypeError('`array` {!r} does not look like a valid array'
                            ''.format(array))

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

        return (super(ArrayWeighting, self).__eq__(other) and
                self.array is getattr(other, 'array', None))

    def __hash__(self):
        """Return ``hash(self)``."""
        # TODO: Better hash for array?
        return hash((super(ArrayWeighting, self).__hash__(),
                     self.array.tobytes()))

    @property
    def repr_part(self):
        """String usable in a space's ``__repr__`` method."""
        optargs = [('weighting', array_str(self.array, nprint=10), ''),
                   ('exponent', self.exponent, 2.0)]
        return signature_string([], optargs, sep=',\n',
                                mod=[[], ['!s', ':.4']])

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [array_str(self.array)]
        optargs = [('exponent', self.exponent, 2.0)]
        inner_str = signature_string(posargs, optargs, sep=',\n',
                                     mod=['!s', ':.4'])
        return '{}(\n{}\n)'.format(self.__class__.__name__, indent(inner_str))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class PerAxisWeighting(Weighting):

    """Weighting of a space with one weight per axis."""

    def __init__(self, factors, impl, exponent=2.0):
        """Initialize a new instance.

        Parameters
        ----------
        factors : sequence of `array-like`
            Weighting factors, one per axis. The factors can be constants
            or one-dimensional array-like objects.
        impl : string
            Specifier for the implementation backend.
        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        """
        super(PerAxisWeighting, self).__init__(impl=impl, exponent=exponent)
        try:
            iter(factors)
        except TypeError:
            raise TypeError('`factors` {!r} is not a sequence'.format(factors))
        self.__factors = tuple(factors)

    @property
    def factors(self):
        """Tuple of weighting factors for inner product, norm and distance."""
        return self.__factors

    # No further methods implemented here since that will require some
    # knowledge on the array type


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
        super(ConstWeighting, self).__init__(impl=impl, exponent=exponent)
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

        return (super(ConstWeighting, self).__eq__(other) and
                self.const == getattr(other, 'const', None))

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((super(ConstWeighting, self).__hash__(), self.const))

    @property
    def repr_part(self):
        """String usable in a space's ``__repr__`` method."""
        optargs = [('weighting', self.const, 1.0),
                   ('exponent', self.exponent, 2.0)]
        return signature_string([], optargs, mod=':.4')

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.const]
        optargs = [('exponent', self.exponent, 2.0)]
        return '{}({})'.format(self.__class__.__name__,
                               signature_string(posargs, optargs))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


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
        super(CustomInner, self).__init__(impl=impl, exponent=2.0)

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
        return (super(CustomInner, self).__eq__(other) and
                self.inner == other.inner)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((super(CustomInner, self).__hash__(), self.inner))

    @property
    def repr_part(self):
        """String usable in a space's ``__repr__`` method."""
        optargs = [('inner', self.inner, '')]
        return signature_string([], optargs, mod='!r')

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.inner]
        optargs = []
        inner_str = signature_string(posargs, optargs, mod='!r')
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
        super(CustomNorm, self).__init__(impl=impl, exponent=1.0)

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
        return (super(CustomNorm, self).__eq__(other) and
                self.norm == other.norm)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((super(CustomNorm, self).__hash__(), self.norm))

    @property
    def repr_part(self):
        """Return a string usable in a space's ``__repr__`` method."""
        optargs = [('norm', self.norm, ''),
                   ('exponent', self.exponent, 2.0)]
        return signature_string([], optargs, mod=[[], ['!r', ':.4']])

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.norm]
        optargs = [('exponent', self.exponent, 2.0)]
        inner_str = signature_string(posargs, optargs, mod=['!r', ':.4'])
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
        super(CustomDist, self).__init__(impl=impl, exponent=1.0)

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
        return (super(CustomDist, self).__eq__(other) and
                self.dist == other.dist)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((super(CustomDist, self).__hash__(), self.dist))

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


def adjoint_weightings(adj_domain, adj_range, extra_factor=1.0,
                       merge_1d_arrs=False):
    """Return functions that together implement correct adjoint weighting.

    This function determines the (presumably) best strategy to merge
    weights and distribute the constant, depending on the involved
    weighting types and space sizes.

    Generally, the domain weighting factors are multiplied with, and the
    range weighting factors appear as reciprocals.

    The optimization logic is thus as follows:

    1. For both domain and range, constant factors, 1D arrays and nD arrays
       are determined.
    2. The constants are merged into ``const = dom_const / ran_const``.
    3. If either domain or range have 1D weighting arrays, the constant
       is merged into the smallest of those.
       Otherwise, the constant is used as multiplicative factor in the
       smaller of the two spaces.
    4. If desired, 1D arrays are merged and used on the smaller space.
    5. Two functions are returned that implement the optimized domain
       and range weightings. They have ``out`` parameters that let them
       operate in-place if necessary.

    Parameters
    ----------
    adj_domain, adj_range : `TensorSpace`
        Domain and range of an adjoint operator for which the weighting
        strategy should be determined.
    extra_factor : float, optional
        Multiply the determined weighting constant by this value. This is for
        situations where additional scalar corrections are necessary.
    merge_1d_arrs : bool, optional
        If ``True``, try to merge the 1D arrays in axes where they occur
        in both domain and range. This is intended as an optimization for
        operators that have identical (or compatible) domain and range,
        at least in the axes in question.

    Returns
    -------
    apply_dom_weighting, apply_ran_weighting : function
        Python functions with signature ``apply(x, out=None)`` that
        implement the optimized strategy for domain and range, returning
        a Numpy array. If a function does not do anything, it returns its
        input ``x`` as Numpy array without making a copy.
    """
    dom_w = adj_domain.weighting
    ran_w = adj_range.weighting
    dom_size = adj_domain.size
    ran_size = adj_range.size
    const = extra_factor

    # Get relevant factors from the weightings

    # Domain weightings go in as multiplicative factors
    if isinstance(dom_w, ArrayWeighting):
        const *= 1.0
        dom_ndarr = dom_w.array
        dom_1darrs = []
        dom_1darr_axes = []
    elif isinstance(dom_w, PerAxisWeighting):
        const *= np.prod(dom_w.consts)
        dom_ndarr = None
        dom_1darrs = list(dom_w.arrays)
        dom_1darr_axes = list(dom_w.array_axes)
    elif isinstance(dom_w, ConstWeighting):
        const *= dom_w.const
        dom_ndarr = None
        dom_1darrs = []
        dom_1darr_axes = []
    else:
        raise TypeError('weighting type {} of the adjoint domain not '
                        'supported'.format(type(dom_w)))

    # Range weightings go in as reciprocal factors. For the later merging
    # of constants and arrays we proceed with the reciprocal factors.
    if isinstance(ran_w, ArrayWeighting):
        const /= 1.0
        ran_ndarr = ran_w.array
        ran_1darrs_recip = []
        ran_1darr_axes = []
    elif isinstance(ran_w, PerAxisWeighting):
        const /= np.prod(ran_w.consts)
        ran_ndarr = None
        ran_1darrs_recip = [1 / arr for arr in ran_w.arrays]
        ran_1darr_axes = list(ran_w.array_axes)
    elif isinstance(ran_w, ConstWeighting):
        const /= ran_w.const
        ran_ndarr = None
        ran_1darrs_recip = []
        ran_1darr_axes = []
    else:
        raise TypeError('weighting type {} of the adjoint range not '
                        'supported'.format(type(ran_w)))

    # Put constant where it "hurts least", i.e., to the smaller space if
    # both use constant weighting or if an array weighting is involved;
    # if 1d arrays are in play, multiply the constant with one of them.
    if const == 1.0:
        dom_const = ran_const = 1.0
    elif dom_1darrs != []:
        idx_smallest = np.argmin([len(arr) for arr in dom_1darrs])
        dom_1darrs[idx_smallest] = dom_1darrs[idx_smallest] * const
        dom_const = ran_const = 1.0
    elif ran_1darrs_recip != []:
        idx_smallest = np.argmin([len(arr) for arr in ran_1darrs_recip])
        ran_1darrs_recip[idx_smallest] *= const  # has been copied already
        dom_const = ran_const = 1.0
    else:
        if dom_size < ran_size:
            dom_const = const
            ran_const = 1.0
        else:
            dom_const = 1.0
            ran_const = const

    # If desired, merge 1d arrays; this works if domain and range have the
    # same shape
    if merge_1d_arrs:
        if adj_domain.ndim != adj_range.ndim:
            raise ValueError('merging only works with domain and range of '
                             'the same number of dimensions')

        for i in range(adj_domain.ndim):
            if i in dom_1darr_axes and i in ran_1darr_axes:
                if dom_size < ran_size:
                    dom_1darrs[i] = dom_1darrs[i] * ran_1darrs_recip[i]
                    ran_1darrs_recip.pop(ran_1darr_axes.index(i))
                    ran_1darr_axes.remove(i)
                else:
                    ran_1darrs_recip[i] *= dom_1darrs[i]  # already copied
                    dom_1darrs.pop(dom_1darr_axes.index(i))
                    dom_1darr_axes.remove(i)

    # Define weighting functions and return them
    def apply_dom_weighting(x, out=None):
        inp = x
        if dom_ndarr is not None:
            # TODO: use first variant when Numpy 1.13 is minimum
            if isinstance(x, np.ndarray):
                out = np.multiply(x, dom_ndarr, out=out)
            else:
                out = x.ufuncs.multiply(dom_ndarr, out=out)
            inp = out
        if dom_const != 1.0:
            if isinstance(x, np.ndarray):
                out = np.multiply(x, dom_const, out=out)
            else:
                out = inp.ufuncs.multiply(dom_const, out=out)
            inp = out
        if dom_1darrs != []:
            if out is None:
                out = fast_1d_tensor_mult(inp, dom_1darrs, dom_1darr_axes)
            else:
                # TODO: use impl when available
                with writable_array(out) as out_arr:
                    fast_1d_tensor_mult(inp, dom_1darrs, dom_1darr_axes,
                                        out=out_arr)

        if out is None:
            # Nothing has been done, return input as Numpy array. This will
            # not copy.
            return np.asarray(x)
        else:
            return np.asarray(out)

    def apply_ran_weighting(x, out=None):
        inp = x
        if ran_ndarr is not None:
            # TODO: use first variant when Numpy 1.13 is minimum
            if isinstance(x, np.ndarray):
                out = np.divide(x, ran_ndarr, out=out)
            else:
                out = x.ufuncs.divide(ran_ndarr, out=out)
            inp = out
        if ran_const != 1.0:
            if isinstance(x, np.ndarray):
                out = np.multiply(x, ran_const, out=out)
            else:
                out = inp.ufuncs.multiply(ran_const, out=out)
            inp = out
        if ran_1darrs_recip != []:
            if out is None:
                out = fast_1d_tensor_mult(inp, ran_1darrs_recip,
                                          ran_1darr_axes)
            else:
                with writable_array(out) as out_arr:
                    fast_1d_tensor_mult(inp, ran_1darrs_recip, ran_1darr_axes,
                                        out=out_arr)

        if out is None:
            # Nothing has been done, return input as Numpy array. This will
            # not copy.
            return np.asarray(x)
        else:
            return np.asarray(out)

    return apply_dom_weighting, apply_ran_weighting


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
