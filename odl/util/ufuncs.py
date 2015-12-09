# Copyright 2014, 2015 The ODL development group
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

""" UFuncs for ODL vectors.

These functions are internal and should only be used as methods on
`NtuplesBaseVector` type spaces.

See `numpy.ufuncs
<http://docs.scipy.org/doc/numpy-1.10.0/reference/ufuncs.html#universal-functions-ufunc>`_
for more information.

Notes
-----
The default implementation of these methods make heavy use of the
`NtuplesBaseVector.__array__` to extract a `numpy.ndarray` from the vector,
and then apply a ufunc to it. Afterwards, `NtuplesBaseVector.__array_wrap__`
is used to re-wrap the data into the appropriate space.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()

# External module imports
import numpy as np

# Some are ignored since they dont cooperate with dtypes, needs fix

# Information:
# Name, input args, output args, docstring
RAW_UFUNCS = [('add', 2, 1, 'Add arguments element-wise with numpy.'),
              ('subtract', 2, 1, 'Subtract arguments, element-wise with numpy.'),
              ('multiply', 2, 1, 'Multiply arguments element-wise with numpy.'),
              ('divide', 2, 1, 'Divide arguments element-wise with numpy.'),
              ('logaddexp', 2, 1, 'Logarithm of the sum of exponentiations of the inputs.'),
              ('logaddexp2', 2, 1, 'Logarithm of the sum of exponentiations of the inputs in base-2.'),
              ('true_divide', 2, 1, 'Returns a true division of the inputs, element-wise.'),
              ('floor_divide', 2, 1, 'Return the largest integer smaller or equal to the division of the inputs.'),
              ('negative', 1, 1, 'Numerical negative, element-wise.'),
              ('power', 2, 1, 'First array elements raised to powers from second array, element-wise.'),
              ('remainder', 2, 1, 'Return element-wise remainder of division.'),
              ('mod', 2, 1, 'Return element-wise remainder of division.'),
              ('fmod', 2, 1, 'Return the element-wise remainder of division.'),
              ('absolute', 1, 1, 'Calculate the absolute value element-wise.'),
              ('rint', 1, 1, 'Round elements of the array to the nearest integer.'),
              ('sign', 1, 1, 'Returns an element-wise indication of the sign of a number.'),
              ('conj', 1, 1, 'Return the complex conjugate, element-wise.'),
              ('exp', 1, 1, 'Calculate the exponential of all elements in the input array.'),
              ('exp2', 1, 1, 'Calculate 2**p for all p in the input array.'),
              ('log', 1, 1, 'Natural logarithm, element-wise.'),
              ('log2', 1, 1, 'Base-2 logarithm of x.'),
              ('log10', 1, 1, 'Return the base 10 logarithm of the input array, element-wise.'),
              ('expm1', 1, 1, 'Calculate exp(x) - 1 for all elements in the array.'),
              ('log1p', 1, 1, 'Return the natural logarithm of one plus the input array, element-wise.'),
              ('sqrt', 1, 1, 'Return the positive square-root of an array, element-wise.'),
              ('square', 1, 1, 'Return the element-wise square of the input.'),
              ('reciprocal', 1, 1, 'Return the reciprocal of the argument, element-wise.'),
              ('sin', 1, 1, 'Trigonometric sine, element-wise.'),
              ('cos', 1, 1, 'Cosine element-wise.'),
              ('tan', 1, 1, 'Compute tangent element-wise.'),
              ('arcsin', 1, 1, 'Inverse sine, element-wise.'),
              ('arccos', 1, 1, 'Trigonometric inverse cosine, element-wise.'),
              ('arctan', 1, 1, 'Trigonometric inverse tangent, element-wise.'),
              ('arctan2', 2, 1, 'Element-wise arc tangent of x1/x2 choosing the quadrant correctly.'),
              ('hypot', 2, 1, 'Given the "legs" of a right triangle, return its hypotenuse.'),
              ('sinh', 1, 1, 'Hyperbolic sine, element-wise.'),
              ('cosh', 1, 1, 'Hyperbolic cosine, element-wise.'),
              ('tanh', 1, 1, 'Compute hyperbolic tangent element-wise.'),
              ('arcsinh', 1, 1, 'Inverse hyperbolic sine element-wise.'),
              ('arccosh', 1, 1, 'Inverse hyperbolic cosine, element-wise.'),
              ('arctanh', 1, 1, 'Inverse hyperbolic tangent element-wise.'),
              ('deg2rad', 1, 1, 'Convert angles from degrees to radians.'),
              ('rad2deg', 1, 1, 'Convert angles from radians to degrees.'),
              ('bitwise_and', 2, 1, 'Compute the bit-wise AND of two arrays element-wise.'),
              ('bitwise_or', 2, 1, 'Compute the bit-wise OR of two arrays element-wise.'),
              ('bitwise_xor', 2, 1, 'Compute the bit-wise XOR of two arrays element-wise.'),
              ('invert', 1, 1, 'Compute bit-wise inversion, or bit-wise NOT, element-wise.'),
              ('left_shift', 2, 1, 'Shift the bits of an integer to the left.'),
              ('right_shift', 2, 1, 'Shift the bits of an integer to the right.'),
              ('greater', 2, 1, 'Return the truth value of (x1 > x2) element-wise.'),
              ('greater_equal', 2, 1, 'Return the truth value of (x1 >= x2) element-wise.'),
              ('less', 2, 1, 'Return the truth value of (x1 < x2) element-wise.'),
              ('less_equal', 2, 1, 'Return the truth value of (x1 =< x2) element-wise.'),
              ('not_equal', 2, 1, 'Return (x1 != x2) element-wise.'),
              ('equal', 2, 1, 'Return (x1 == x2) element-wise.'),
              ('logical_and', 2, 1, 'Compute the truth value of x1 AND x2 element-wise.'),
              ('logical_or', 2, 1, 'Compute the truth value of x1 OR x2 element-wise.'),
              ('logical_xor', 2, 1, 'Compute the truth value of x1 XOR x2, element-wise.'),
              ('logical_not', 1, 1, 'Compute the truth value of NOT x element-wise.'),
              ('maximum', 2, 1, 'Element-wise maximum of array elements.'),
              ('minimum', 2, 1, 'Element-wise minimum of array elements.'),
              ('fmax', 2, 1, 'Element-wise maximum of array elements.'),
              ('fmin', 2, 1, 'Element-wise minimum of array elements.'),
              # ('isreal', 1, 0, 'Returns a bool array, where True if input element is real.'),
              # ('iscomplex', 1, 0, 'Returns a bool array, where True if input element is complex.'),
              ('isfinite', 1, 1, 'Test element-wise for finiteness (not infinity or not Not a Number).'),
              ('isinf', 1, 1, 'Test element-wise for positive or negative infinity.'),
              ('isnan', 1, 1, 'Test element-wise for NaN and return result as a boolean array.'),
              ('signbit', 1, 1, 'Returns element-wise True where signbit is set (less than zero).'),
              ('copysign', 2, 1, 'Change the sign of x1 to that of x2, element-wise.'),
              ('nextafter', 2, 1, 'Return the next floating-point value after x1 towards x2, element-wise.'),
              ('modf', 1, 2, 'Return the fractional and integral parts of an array, element-wise.'),
              # ('ldexp', 2, 1, 'Returns x1 * 2**x2, element-wise.'),
              # ('frexp', 1, 2, 'Decompose the elements of x into mantissa and twos exponent.'),
              ('fmod', 2, 1, 'Return the element-wise remainder of division.'),
              ('floor', 1, 1, 'Return the floor of the input, element-wise.'),
              ('ceil', 1, 1, 'Return the ceiling of the input, element-wise.'),
              ('trunc', 1, 1, 'Return the truncated value of the input, element-wise.')]

# Add some standardized information
UFUNCS = []
for name, n_args, n_opt, descr in RAW_UFUNCS:
    doc = descr + """

See also
--------
numpy.{}
""".format(name)
    UFUNCS += [(name, n_args, n_opt, doc)]

RAW_REDUCTIONS = [('sum', 'Sum of array elements.'),
                  ('prod', 'Product of array elements.'),
                  ('min', 'Minimum value in array.'),
                  ('max', 'Maximum value in array.')]

REDUCTIONS = []
for name, descr in RAW_REDUCTIONS:
    doc = descr + """

See also
--------
numpy.{}
""".format(name)
    REDUCTIONS += [(name, doc)]


# Wrap all numpy ufuncs
def wrap_ufunc_base(name, n_args, n_opt, descr):
    """Add ufunc methods to `NtuplesBaseVectorUFuncs`."""
    wrapped = getattr(np, name)
    if n_args == 1:
        if n_opt == 0:
            def wrapper(self):
                return wrapped(self.vector)

        elif n_opt == 1:
            def wrapper(self, out=None):
                if out is None:
                    out = self.vector.space.element()

                out[:] = wrapped(self.vector)
                return out

        elif n_opt == 2:
            def wrapper(self, out1=None, out2=None):
                if out1 is None:
                    out1 = self.vector.space.element()
                if out2 is None:
                    out2 = self.vector.space.element()

                [y1, y2] = wrapped(self.vector)
                out1[:] = y1
                out2[:] = y2
                return out1, out2

        else:
            raise NotImplementedError

    elif n_args == 2:
        if n_opt == 1:
            def wrapper(self, x2, out=None):
                if out is None:
                    return wrapped(self.vector, x2)
                else:
                    out[:] = wrapped(self.vector, x2)
                    return out

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    wrapper.__name__ = name
    wrapper.__doc__ = descr
    return wrapper


# Wrap reductions
def wrap_reduction_base(name, descr):
    """Add ufunc methods to `NtuplesBaseVectorUFuncs`."""
    wrapped = getattr(np, name)

    def wrapper(self):
        return wrapped(self.vector)

    wrapper.__name__ = name
    wrapper.__doc__ = descr
    return wrapper


class NtuplesBaseVectorUFuncs(object):
    """UFuncs for `NtuplesBaseVector` objects.

    Internal object, should not be created except in `NtuplesBaseVector`.
    """
    def __init__(self, vector):
        """Create ufunc wrapper for vector."""
        self.vector = vector


# Add ufunc methods to UFunc class
for name, n_args, n_opt, descr in UFUNCS:
    method = wrap_ufunc_base(name, n_args, n_opt, descr)
    setattr(NtuplesBaseVectorUFuncs, name, method)

# Add reduction methods to UFunc class
for name, descr in REDUCTIONS:
    method = wrap_reduction_base(name, descr)
    setattr(NtuplesBaseVectorUFuncs, name, method)


# Optimized implementation of ufuncs since we can use the out parameter
# as well as the data parameter to avoid one call to asarray() when using a
# NtuplesVector
def wrap_ufunc_ntuples(name, n_args, n_opt, descr):
    """Add ufunc methods to `NtuplesVectorUFuncs`."""

    # Get method from numpy
    wrapped = getattr(np, name)
    if n_args == 1:
        if n_opt == 0:
            def wrapper(self):
                return wrapped(self.vector)

        elif n_opt == 1:
            def wrapper(self, out=None):
                if out is None:
                    out = self.vector.space.element()
                wrapped(self.vector, out.data)
                return out

        elif n_opt == 2:
            def wrapper(self, out1=None, out2=None):
                if out1 is None:
                    out1 = self.vector.space.element()
                if out2 is None:
                    out2 = self.vector.space.element()

                y1, y2 = wrapped(self.vector, out1.data, out2.data)
                return out1, out2

        else:
            raise NotImplementedError

    elif n_args == 2:
        if n_opt == 1:
            def wrapper(self, x2, out=None):
                if out is None:
                    out = self.vector.space.element()

                wrapped(self.vector, x2, out.data)
                return out

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    wrapper.__name__ = name
    wrapper.__doc__ = descr
    return wrapper


class NtuplesVectorUFuncs(NtuplesBaseVectorUFuncs):
    """UFuncs for `NtuplesVector` objects.

    Internal object, should not be created except in `NtuplesVector`.
    """


# Add ufunc methods to UFunc class
for name, n_args, n_opt, descr in UFUNCS:
    method = wrap_ufunc_ntuples(name, n_args, n_opt, descr)
    setattr(NtuplesVectorUFuncs, name, method)


# Optimizations for CUDA
def _make_nullary_fun(name):
    def fun(self):
        return getattr(self.vector.data, name)()

    fun.__doc__ = getattr(NtuplesBaseVectorUFuncs, name).__doc__
    fun.__name__ = name
    return fun


def _make_unary_fun(name):
    def fun(self, out=None):
        if out is None:
            out = self.vector.space.element()
        getattr(self.vector.data, name)(out.data)
        return out

    fun.__doc__ = getattr(NtuplesBaseVectorUFuncs, name).__doc__
    fun.__name__ = name
    return fun


class CudaNtuplesVectorUFuncs(NtuplesBaseVectorUFuncs):
    # Ufuncs
    sin = _make_unary_fun('sin')
    cos = _make_unary_fun('cos')
    arcsin = _make_unary_fun('arcsin')
    arccos = _make_unary_fun('arccos')
    log = _make_unary_fun('log')
    exp = _make_unary_fun('exp')
    absolute = _make_unary_fun('absolute')
    sign = _make_unary_fun('sign')
    sqrt = _make_unary_fun('sqrt')

    # Reductions
    sum = _make_nullary_fun('sum')
    prod = _make_nullary_fun('prod')
    min = _make_nullary_fun('min')
    max = _make_nullary_fun('max')


# Optimized implementation of ufuncs since we can use the out parameter
# as well as the data parameter to avoid one call to asarray() when using a
# NtuplesVector
def wrap_ufunc_discretelp(name, n_args, n_opt, descr):
    """Add ufunc methods to `DiscreteLpVectorUFuncs`."""

    if n_args == 1:
        if n_opt == 0:
            def wrapper(self):
                method = getattr(self.vector.ntuple.ufunc, name)
                return self.vector.space.element(method())

        elif n_opt == 1:
            def wrapper(self, out=None):
                method = getattr(self.vector.ntuple.ufunc, name)
                if out is None:
                    return self.vector.space.element(method())
                else:
                    method(out=out.ntuple)
                    return out

        elif n_opt == 2:
            def wrapper(self, out1=None, out2=None):
                method = getattr(self.vector.ntuple.ufunc, name)
                if out1 is None:
                    out1 = self.vector.space.element()
                if out2 is None:
                    out2 = self.vector.space.element()

                y1, y2 = method(out1.ntuple, out2.ntuple)
                return out1, out2

        else:
            raise NotImplementedError

    elif n_args == 2:
        if n_opt == 1:
            def wrapper(self, x2, out=None):
                try:
                    x2 = x2.ntuple
                except AttributeError:
                    x2 = x2

                method = getattr(self.vector.ntuple.ufunc, name)
                if out is None:
                    return self.vector.space.element(method(x2))
                else:
                    method(x2, out.ntuple)
                    return out

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    wrapper.__name__ = name
    wrapper.__doc__ = descr
    return wrapper


class DiscreteLpVectorUFuncs(NtuplesBaseVectorUFuncs):
    """UFuncs for `DiscreteLpVector` objects.

    Internal object, should not be created except in `DiscreteLpVector`.
    """


# Add ufunc methods to UFunc class
for name, n_args, n_opt, descr in UFUNCS:
    method = wrap_ufunc_discretelp(name, n_args, n_opt, descr)
    setattr(DiscreteLpVectorUFuncs, name, method)
