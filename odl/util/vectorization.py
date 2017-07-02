# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for internal functionality connected to vectorization."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from functools import wraps
import numpy as np


__all__ = ('is_valid_input_array', 'is_valid_input_meshgrid',
           'out_shape_from_meshgrid', 'out_shape_from_array',
           'OptionalArgDecorator', 'vectorize')


def is_valid_input_array(x, ndim=None):
    """Test if ``x`` is a correctly shaped point array in R^d."""
    x = np.asarray(x)

    if ndim is None or ndim == 1:
        return x.ndim == 1 and x.size > 1 or x.ndim == 2 and x.shape[0] == 1
    else:
        return x.ndim == 2 and x.shape[0] == ndim


def is_valid_input_meshgrid(x, ndim):
    """Test if ``x`` is a `meshgrid` sequence for points in R^d."""
    # This case is triggered in FunctionSpaceElement.__call__ if the
    # domain does not have an 'ndim' attribute. We return False and
    # continue.
    if ndim is None:
        return False

    if not isinstance(x, tuple):
        return False

    if ndim > 1:
        try:
            np.broadcast(*x)
        except (ValueError, TypeError):  # cannot be broadcast
            return False

    return (len(x) == ndim and
            all(isinstance(xi, np.ndarray) for xi in x) and
            all(xi.ndim == ndim for xi in x))


def out_shape_from_meshgrid(mesh):
    """Get the broadcast output shape from a `meshgrid`."""
    if len(mesh) == 1:
        return (len(mesh[0]),)
    else:
        return np.broadcast(*mesh).shape


def out_shape_from_array(arr):
    """Get the output shape from an array."""
    arr = np.asarray(arr)
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

           which is equivalent to ::

               def myfunc(x):
                   pass

               myfunc = decorator(myfunc)


           Hence, in this case, the ``__new__`` method of the decorator
           immediately returns the wrapped function.

        2. With arguments::

               @decorator(*dec_args, **dec_kwargs)
               def myfunc(x):
                   pass

           which is equivalent to ::

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
        """Make a wrapper for ``func`` and return it.

        This is a default implementation that simply returns the wrapped
        function, i.e., the resulting decorator is the identity.
        """
        return func


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
        if not hasattr(func, '__name__'):
            # Set name if not available. Happens if func is actually a function
            func.__name__ = '{}.__call__'.format(func.__class__.__name__)

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
        x : `array-like` or sequence of `array-like`'s
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
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
