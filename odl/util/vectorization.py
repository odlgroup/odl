﻿# Copyright 2014-2016 The ODL development group
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

"""Utilities for internal functionality connected to vectorization."""


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

# External module imports
from functools import wraps
import numpy as np


__all__ = ('is_valid_input_array', 'is_valid_input_meshgrid',
           'meshgrid_input_order', 'vecs_from_meshgrid',
           'out_shape_from_meshgrid', 'out_shape_from_array',
           'vectorize')


def is_valid_input_array(x, ndim=None):
    """Test if ``x`` is a correctly shaped point array in R^d."""
    if not isinstance(x, np.ndarray):
        return False
    if ndim is None or ndim == 1:
        return x.ndim == 1 or x.ndim == 2 and x.shape[0] == 1
    else:
        return x.ndim == 2 and x.shape[0] == ndim


def is_valid_input_meshgrid(x, ndim):
    """Test if ``x`` is a meshgrid sequence for points in R^d."""
    # This case is triggered in FunctionSetVector.__call__ if the
    # domain does not have an 'ndim' attribute. We return False and
    # continue.
    if ndim is None:
        return False
    try:
        len(x)
    except TypeError:
        return False

    if isinstance(x, np.ndarray):
        return False

    if ndim > 1:
        try:
            np.broadcast(*x)
        except (ValueError, TypeError):  # cannot be broadcast
            return False

    return (len(x) == ndim and
            all(isinstance(xi, np.ndarray) for xi in x) and
            all(xi.ndim == ndim for xi in x))


def meshgrid_input_order(x):
    """Determine the ordering of a meshgrid argument."""
    # Case 1: all elements have the same shape -> non-sparse
    if all(xi.shape == x[0].shape for xi in x):
        # Contiguity check only works for meshgrid created with copy=True.
        # Otherwise, there is no way to find out the intended ordering.
        if all(xi.flags.c_contiguous for xi in x):
            return 'C'
        elif all(xi.flags.f_contiguous for xi in x):
            return 'F'
        else:
            raise ValueError('unable to determine ordering.')
    # Case 2: sparse meshgrid, each member's shape has at most one non-one
    # entry (corner case of all ones is included)
    elif all(xi.shape.count(1) >= len(x) - 1 for xi in x):
        # Reversed ordering of dimensions in the meshgrid tuple indicates
        # 'F' ordering intention
        # All other dimension except 'i' have length 1 -> 'C'
        if all(xi.shape[j] == 1
               for j in range(len(x))
               for i, xi in enumerate(x)
               if j != i):
            return 'C'
        # All other dimension except 'n - i' have length 1 -> 'C'
        if all(xi.shape[j] == 1
               for j in range(len(x))
               for i, xi in enumerate(x)
               if j != len(x) - 1 - i):
            return 'F'
        else:
            raise ValueError('unable to determine ordering.')


def vecs_from_meshgrid(mesh, order):
    """Get the coordinate vectors from a meshgrid (as a tuple)."""
    vecs = []
    order_ = str(order).upper()
    if order_ not in ('C', 'F'):
        raise ValueError("unknown ordering '{}'.".format(order))

    if order == 'C':
        seq = mesh
    else:
        seq = reversed(mesh)

    for ax, vec in enumerate(seq):
        select = [0] * len(mesh)
        select[ax] = np.s_[:]
        vecs.append(vec[select])

    return tuple(vecs)


def out_shape_from_meshgrid(mesh):
    """Get the broadcast output shape from a meshgrid."""
    if len(mesh) == 1:
        return (len(mesh[0]),)
    else:
        return np.broadcast(*mesh).shape


def out_shape_from_array(arr):
    """Get the output shape from an array."""
    if arr.ndim == 1:
        return arr.shape
    else:
        return (arr.shape[1],)


class OptionalArgDecorator(object):

    """Abstract class to create decorators with optional arguments.

    This class implements the functionality of a decorator that can
    be used with and without arguments, i.e. the following patterns
    both work::

        @decorator
        def myfunc(x, *args, **kwargs):
            pass

        @decorator(param, **dec_kwargs)
        def myfunc(x, *args, **kwargs):
            pass

    The arguments to the decorator are passed on to the underlying
    wrapper.

    To use this class, subclass it and implement the static ``_wrapper``
    method.
    """

    def __new__(cls, *args, **kwargs):
        """Create a new decorator instance.

        There are two cases to distinguish:

        1. Without arguments::

               @decorator
               def myfunc(x):
                   pass

           which is equivalent to
           ::

               def myfunc(x):
                   pass

               myfunc = decorator(myfunc)


           Hence, in this case, the ``__new__`` method of the decorator
           immediately returns the wrapped function.

        2. With arguments::

               @decorator(*dec_args, **dec_kwargs)
               def myfunc(x):
                   pass

           which is equivalent to
           ::

               def myfunc(x):
                   pass

               dec_instance = decorator(*dec_args, **dec_kwargs)
               myfunc = dec_instance(myfunc)

           Hence, in this case, the first call creates an actual class
           instance of ``decorator``, and in the second statement, the
           ``dec_instance.__call__`` method returns the wrapper using
           the stored ``dec_args`` and ``dec_kwargs``.
        """
        # Decorating without arguments: return wrapper w/o args directly
        instance = super().__new__(cls)

        if (not kwargs and
                len(args) == 1 and
                callable(args[0])):
            func = args[0]

            return instance._wrapper(func)

        # With arguments, return class instance
        else:
            instance.wrapper_args = args
            instance.wrapper_kwargs = kwargs
            return instance

    def __call__(self, func):
        """Return ``self(func)``.

        This method is invoked when the decorator was created with
        arguments.

        Parameters
        ----------
        func : callable
            Original function to be wrapped

        Returns
        -------
        wrapped : callable
            The wrapped function
        """
        return self._wrapper(func, *self.wrapper_args, **self.wrapper_kwargs)

    @staticmethod
    def _wrapper(func, *wrapper_args, **wrapper_kwargs):
        """Return the wrapped function."""
        raise NotImplementedError


class vectorize(OptionalArgDecorator):

    """Decorator class for function vectorization.

    This vectorizer expects a function with exactly one positional
    argument (input) and optional keyword arguments. The decorated
    function has an optional ``out`` parameter for in-place evaluation.

    Examples
    --------
    Use the decorator witout arguments:

    >>> @vectorize
    ... def f(x):
    ...     return x[0] + x[1] if x[0] < x[1] else x[0] - x[1]
    >>>
    >>> f([0, 1])  # np.vectorize'd functions always return an array
    array(1)
    >>> f([[0, -2], [1, 4]])  # corresponds to points [0, 1], [-2, 4]
    array([1, 2])

    The function may have ``kwargs``:

    >>> @vectorize
    ... def f(x, param=1.0):
    ...     return x[0] + x[1] if x[0] < param else x[0] - x[1]
    >>>
    >>> f([[0, -2], [1, 4]])
    array([1, 2])
    >>> f([[0, -2], [1, 4]], param=-1.0)
    array([-1,  2])

    You can pass arguments to the vectorizer, too:

    >>> @vectorize(otypes=['float32'])
    ... def f(x):
    ...     return x[0] + x[1] if x[0] < x[1] else x[0] - x[1]
    >>> f([[0, -2], [1, 4]])
    array([ 1.,  2.], dtype=float32)
    """

    @staticmethod
    def _wrapper(func, *vect_args, **vect_kwargs):
        """Return the vectorized wrapper function."""
        return wraps(func)(_NumpyVectorizeWrapper(func, *vect_args,
                                                  **vect_kwargs))


class _NumpyVectorizeWrapper(object):

    """Class for vectorization wrapping using `numpy.vectorize`.

    The purpose of this class is to store the vectorized version of
    a function when it is called for the first time.
    """

    def __init__(self, func, *vect_args, **vect_kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        func : callable
            Python function or method to be wrapped
        vect_args :
            positional arguments for `numpy.vectorize`
        vect_kwargs :
            keyword arguments for `numpy.vectorize`
        """
        self.func = func
        self.vfunc = None
        self.vect_args = vect_args
        self.vect_kwargs = vect_kwargs

    def __call__(self, x, out=None, **kwargs):
        """Vectorized function call.

        Parameters
        ----------
        x : array-like or sequence of array-like
            Input argument(s) to the wrapped function
        out : `numpy.ndarray`, optional
            Appropriately sized array to write to

        Returns
        -------
        out : `numpy.ndarray`
            Result of the vectorized function evaluation. If ``out``
            was given, the returned object is a reference to it.
        """
        if np.isscalar(x):
            x = np.array([x])
        elif isinstance(x, np.ndarray) and x.ndim == 1:
            x = x[None, :]

        if self.vfunc is None:
            # Not yet vectorized
            def _func(*x, **kw):
                return self.func(np.array(x), **kw)

            self.vfunc = np.vectorize(_func, *self.vect_args,
                                      **self.vect_kwargs)

        if out is None:
            return self.vfunc(*x, **kwargs)
        else:
            out[:] = self.vfunc(*x, **kwargs)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
