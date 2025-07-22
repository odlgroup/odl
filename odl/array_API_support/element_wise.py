# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Element-wise functions expected by the python array API"""

__all__ = (
    'abs',
    'acos',
    'acosh',
    'add',
    'asin',
    'asinh',
    'atan',
    'atan2',
    'atanh',
    'bitwise_and',
    'bitwise_left_shift',
    'bitwise_invert',
    'bitwise_or',
    'bitwise_right_shift',
    'bitwise_xor',
    'ceil',
    'clip',
    'conj',
    'copysign',
    'cos',
    'cosh',
    'divide',
    'equal',
    'exp',
    'expm1',
    'floor',
    'floor_divide',
    'greater',
    'greater_equal',
    'hypot',
    'imag',
    'isfinite',
    'isinf',
    'isnan',
    'less',
    'less_equal',
    'log',
    'log1p',
    'log2',
    'log10',
    'logaddexp',
    'logical_and',
    'logical_not',
    'logical_or',
    'logical_xor',
    'maximum',
    'minimum',
    'multiply',
    'negative',
    'nextafter',
    'not_equal',
    'positive',
    'pow',
    'real',
    'reciprocal',
    'remainder',
    'round',
    'sign',
    'signbit',
    'sin',
    'sinh',
    'sqrt',
    'square',
    'subtract',
    'tan',
    'tanh',
    'trunc',
)


def _apply_element_wise(operation: str, x1, x2=None, out=None, **kwargs):
    """
    Helper function to apply an element-wise `operation` on:
        -> a python int/float/complex and a LinearSpaceElement
        -> two LinearSpaceElement
        -> a single LinearSpaceElement

    Args:
        operation (str): a string identifier to lookup the desired function in the LinearSpaceElement's namespace.
        x1 (int | float | complex | LinearSpaceElement): Left operand
        x2 (int | float | complex | LinearSpaceElement, optional): Right operand. Defaults to None.
        out (LinearSpaceElement, optional): Out LinearSpaceElement for inplace updates. Defaults to None.

    Returns:
        LinearSpaceElement: result of the element-wise operation on the array wrapped inside the element of an ODL space. 

    Notes:
        1) The output array is wrapped in a space of which type depends of the output array's. This is a change of behaviour compared to ODL < 0.8.2
        2) Although one could use it to perform an operation on array-specific backend, there is no clean way to infer a LinearSpace from the output. As such, one of the two operands must be a LinearSpaceElement

    Examples
    >>> e0 = odl.rn(3).zero()
    >>> e1 = odl.rn(3).one()
    >>> e2 = e0 + e1
    >>> print(e2)
    [ 1.,  1.,  1.]
    >>> e3 = odl.add(e0, e1)
    >>> print(e3)
    [ 1.,  1.,  1.]
    >>> e2 == e3 
    True
    >>> e2 in odl.rn(3)
    True
    >>> new_el = e0 + 3j 
    >>> new_el in odl.rn(3)
    False
    >>> odl.add(np.zeros(3), e1)
    rn(3, 'float64', 'numpy', 'cpu').element([ 1.,  1.,  1.])
    """
    # Lazy import of LinearSpaceElement and Operator for dispatching call
    from odl.operator import Operator
    from odl.set.space import LinearSpaceElement
    assert not isinstance(x1, Operator) or not isinstance(x2, Operator), f"ODL's array-API support for element-wise functions does not allow ODL Operators"
    if isinstance(x1, LinearSpaceElement):
        return x1.space._elementwise_num_operation(operation=operation, x1=x1, x2=x2, out=out, **kwargs)
    # Handling the left argument as a float/int/complex and right argument as a LinearSpaceElement
    elif isinstance(x2, LinearSpaceElement):
        return x2.space._elementwise_num_operation(operation=operation, x1=x1, x2=x2, out=out, **kwargs)   
    else:
        raise(AttributeError(f"Either x1 or x2 (if provided) need to be a LinearSpaceElement, got {type(x1)} and {type(x2)} with values {x1=} and {x2=}"))


def abs(x, out=None):
    """Calculates the absolute value for each element `x_i` of the input array
    `x`."""
    return _apply_element_wise('abs', x, out=out)


def acos(x, out=None):
    """Calculates an implementation-dependent approximation of the principal
    value of the inverse cosine for each element `x_i` of the input array
    `x`."""
    return _apply_element_wise('acos', x, out=out)


def acosh(x, out=None):
    """Calculates an implementation-dependent approximation to the inverse
    hyperbolic cosine for each element `x_i` of the input array `x`."""
    return _apply_element_wise('acosh', x, out=out)


def add(x1, x2, out=None):
    """Calculates the sum for each element `x1_i` of the input array `x1` with
    the respective element `x2_i` of the input array `x2`."""
    return _apply_element_wise('add', x1, x2=x2, out=out)


def asin(x, out=None):
    """Calculates an implementation-dependent approximation of the principal
    value of the inverse sine for each element `x_i` of the input array `x`."""
    return _apply_element_wise('asin', x, out=out)


def asinh(x, out=None):
    """Calculates an implementation-dependent approximation to the inverse
    hyperbolic sine for each element `x_i` in the input array `x`."""
    return _apply_element_wise('asinh', x, out=out)


def atan(x, out=None):
    """Calculates an implementation-dependent approximation of the principal
    value of the inverse tangent for each element `x_i` of the input array
    `x`."""
    return _apply_element_wise('atan', x, out=out)


def atan2(x1, x2, out=None):
    """Calculates an implementation-dependent approximation of the inverse
    tangent of the quotient `x1/x2`, having domain `[-infinity, +infinity]
    \times [-infinity, +infinity]` (where the `\times` notation denotes the set
    of ordered pairs of elements `(x1_i, x2_i)`) and codomain `[-pi, +pi]`,
    for each pair of elements `(x1_i, x2_i)` of the input arrays `x1` and `x2`,
    respectively."""
    return _apply_element_wise("atan2", x1, x2=x2, out=out)


def atanh(x, out=None):
    """Calculates an implementation-dependent approximation to the inverse
    hyperbolic tangent for each element `x_i` of the input array `x`."""
    return _apply_element_wise('atanh', x, out=out)


def bitwise_and(x1, x2, out=None):
    """Computes the bitwise AND of the underlying binary representation of each
    element `x1_i` of the input array `x1` with the respective element `x2_i`
    of the input array `x2`."""
    return _apply_element_wise('bitwise_and', x1, x2=x2, out=out)


def bitwise_left_shift(x1, x2, out=None):
    """Shifts the bits of each element `x1_i` of the input array `x1` to the
    left by appending `x2_i` (i.e., the respective element in the input array
    `x2`) zeros to the right of `x1_i`."""
    return _apply_element_wise('bitwise_left_shift', x1, x2=x2, out=out)


def bitwise_invert(x, out=None):
    """Inverts (flips) each bit for each element `x_i` of the input array
    `x`."""
    return _apply_element_wise('bitwise_invert', x, out=out)


def bitwise_or(x1, x2, out=None):
    """Computes the bitwise OR of the underlying binary representation of each
    element `x1_i` of the input array `x1` with the respective element `x2_i`
    of the input array `x2`."""
    return _apply_element_wise('bitwise_or', x1, x2=x2, out=out)


def bitwise_right_shift(x1, x2, out=None):
    """Shifts the bits of each element `x1_i` of the input array `x1` to the
    right according to the respective element `x2_i` of the input array
    `x2`."""
    return _apply_element_wise('bitwise_right_shift', x1, x2=x2, out=out)


def bitwise_xor(x1, x2, out=None):
    """Computes the bitwise XOR of the underlying binary representation of each
    element `x1_i` of the input array `x1` with the respective element `x2_i`
    of the input array `x2`."""
    return _apply_element_wise('bitwise_xor', x1, x2=x2, out=out)


def ceil(x, out=None):
    """Rounds each element `x_i` of the input array `x` to the smallest (i.e.,
    closest to `-infty`) integer-valued number that is not less than `x_i`."""
    return _apply_element_wise('ceil', x, out=out)


def clip(x, out=None, min=None, max=None):
    """Clamps each element `x_i` of the input array `x` to the range `[min,
    max]`."""
    return _apply_element_wise('clip', x, out=out, min=min, max=max)


def conj(x, out=None):
    """Returns the complex conjugate for each element `x_i` of the input array
    `x`."""
    return _apply_element_wise('conj', x, out=out)


def copysign(x1, x2, out=None):
    """Composes a floating-point value with the magnitude of `x1_i` and the
    sign of `x2_i` for each element of the input array `x1`."""
    return _apply_element_wise('copysign', x1, x2=x2, out=out)


def cos(x, out=None):
    """Calculates an implementation-dependent approximation to the cosine for
    each element `x_i` of the input array `x`."""
    return _apply_element_wise('cos', x, out=out)


def cosh(x, out=None):
    """Calculates an implementation-dependent approximation to the hyperbolic
    cosine for each element `x_i` in the input array `x`."""
    return _apply_element_wise('cosh', x, out=out)


def divide(x1, x2, out=None):
    """Calculates the division of each element `x1_i` of the input array `x1`
    with the respective element `x2_i` of the input array `x2`."""
    return _apply_element_wise('divide', x1, x2=x2, out=out)


def equal(x1, x2, out=None):
    """Computes the truth value of `x1_i == x2_i` for each element `x1_i` of
    the input array `x1` with the respective element `x2_i` of the input array
    `x2`."""
    return _apply_element_wise('equal', x1, x2=x2, out=out)


def exp(x1, out=None):
    """Calculates an implementation-dependent approximation to the exponential
    function for each element `x_i` of the input array `x` (`e` raised to the
    power of `x_i`, where `e` is the base of the natural logarithm)."""
    return _apply_element_wise('exp', x1, out=out)


def expm1(x1, out=None):
    """Calculates an implementation-dependent approximation to `exp(x_i) - 1`
    for each element `x_i` of the input array `x`."""
    return _apply_element_wise("expm1", x1, out=out)


def floor(x1, out=None):
    """Rounds each element `x_i` of the input array `x` to the largest (i.e.,
    closest to `+infty`) integer-valued number that is not greater than
    `x_i`."""
    return _apply_element_wise('floor', x1, out=out)


def floor_divide(x1, x2, out=None):
    """Calculates the largest integer-valued number that is not greater than
    the result of dividing each element `x1_i` of the input array `x1` by the
    respective element `x2_i` of the input array `x2`."""
    return _apply_element_wise('floor_divide', x1, x2=x2, out=out)


def greater(x1, x2, out=None):
    """Computes the truth value of `x1_i > x2_i` for each element `x1_i` of the
    input array `x1` with the respective element `x2_i` of the input array
    `x2`."""
    return _apply_element_wise('greater', x1, x2=x2, out=out)


def greater_equal(x1, x2, out=None):
    """Computes the truth value of `x1_i >= x2_i` for each element `x1_i` of
    the input array `x1` with the respective element `x2_i` of the input array
    `x2`."""
    return _apply_element_wise('greater_equal', x1, x2=x2, out=out)


def hypot(x1, x2, out=None):
    """Computes the square root of the sum of squares for each element `x1_i`
    of the input array `x1` with the respective element `x2_i` of the input
    array `x2`."""
    return _apply_element_wise('hypot', x1, x2=x2, out=out)


def imag(x1, out=None):
    """Returns the imaginary part of each element `x_i` of the input array
    `x`."""
    return _apply_element_wise('imag', x1, out=out)


def isfinite(x1, out=None):
    """Tests each element `x_i` of the input array `x` to determine if it is
    finite (i.e., not `NaN` and not an infinity)."""
    return _apply_element_wise('isfinite', x1, out=out)


def isinf(x1, out=None):
    """Tests each element `x_i` of the input array `x` to determine if it is a
    positive or negative infinity."""
    if x1 == float('Inf') or x1 == -float("Inf"):
        return True
    return _apply_element_wise('isinf', x1, out=out)


def isnan(x1, out=None):
    """Tests each element `x_i` of the input array `x` to determine if it is a
    `NaN`."""
    if x1 == float('Nan'):
        return True
    return _apply_element_wise('isnan', x1, out=out)


def less(x1, x2, out=None):
    """Computes the truth value of `x1_i < x2_i` for each element `x1_i` of the
    input array `x1` with the respective element `x2_i` of the input array
    `x2`."""
    return _apply_element_wise('less', x1, x2=x2, out=out)


def less_equal(x1, x2, out=None):
    """Computes the truth value of `x1_i <= x2_i` for each element `x1_i` of
    the input array `x1` with the respective element `x2_i` of the input array
    `x2`."""
    return _apply_element_wise('less_equal', x1, x2=x2, out=out)


def log(x1, out=None):
    """Calculates an implementation-dependent approximation to the natural
    logarithm for each element `x_i` of the input array `x`."""
    return _apply_element_wise('log', x1, out=out)


def log1p(x1, out=None):
    """Calculates an implementation-dependent approximation to `ln(1 + x_i)`
    for each element `x_i` of the input array `x`.

    For small `x`, the result of this function should be more accurate
    than `log(1 + x)`.
    """
    return _apply_element_wise("log1p", x1, out=out)


def log2(x1, out=None):
    """Calculates an implementation-dependent approximation to the base two
    logarithm for each element `x_i` of the input array `x`."""
    return _apply_element_wise("log2", x1, out=out)


def log10(x1, out=None):
    """Calculates an implementation-dependent approximation to the base ten
    logarithm for each element `x_i` of the input array `x`."""
    return _apply_element_wise("log10", x1, out=out)


def logaddexp(x1, x2, out=None):
    """Calculates the logarithm of the sum of exponentiations `log(exp(x1) +
    exp(x2))` for each element `x1_i` of the input array `x1` with the
    respective element `x2_i` of the input array `x2`."""
    return _apply_element_wise('logaddexp', x1, x2=x2, out=out)


def logical_and(x1, x2, out=None):
    """Computes the logical AND for each element `x1_i` of the input array `x1`
    with the respective element `x2_i` of the input array `x2`."""
    return _apply_element_wise('logical_and', x1, x2=x2, out=out)


def logical_not(x1, out=None):
    """Computes the logical NOT for each element `x_i` of the input array
    `x`."""
    return _apply_element_wise('logical_not', x1, out=out)


def logical_or(x1, x2, out=None):
    """Computes the logical OR for each element `x1_i` of the input array `x1`
    with the respective element `x2_i` of the input array `x2`."""
    return _apply_element_wise('logical_or', x1, x2=x2, out=out)


def logical_xor(x1, x2, out=None):
    """Computes the logical XOR for each element `x1_i` of the input array `x1`
    with the respective element `x2_i` of the input array `x2`."""
    return _apply_element_wise('logical_xor', x1, x2=x2, out=out)


def maximum(x1, x2, out=None):
    """Computes the maximum value for each element `x1_i` of the input array
    `x1` relative to the respective element `x2_i` of the input array `x2`."""
    return _apply_element_wise('maximum', x1, x2=x2, out=out)


def minimum(x1, x2, out=None):
    """Calculates an implementation-dependent approximation of the principal
    value of the inverse cosine for each element."""
    return _apply_element_wise('minimum', x1, x2=x2, out=out)


def multiply(x1, x2, out=None):
    """Calculates the product for each element `x1_i` of the input array `x1`
    with the respective element `x2_i` of the input array `x2`."""
    return _apply_element_wise('multiply', x1, x2=x2, out=out)


def negative(x1, out=None):
    """Numerically negates each element `x_i` of the input array `x`."""
    return _apply_element_wise('negative', x1, out=out)


def nextafter(x1, x2, out=None):
    """Returns the next representable floating-point value for each element
    `x1_i` of the input array `x1` in the direction of the respective element
    `x2_i` of the input array `x2`."""
    return _apply_element_wise('nextafter', x1, x2=x2, out=out)


def not_equal(x1, x2, out=None):
    """Computes the truth value of `x1_i != x2_i` for each element `x1_i` of
    the input array `x1` with the respective element `x2_i` of the input array
    `x2`."""
    return _apply_element_wise('not_equal', x1, x2=x2, out=out)


def positive(x1, out=None):
    """Numerically positive each element `x_i` of the input array `x`."""
    return _apply_element_wise('positive', x1, out=out)


def pow(x1, x2, out=None):
    """Calculates an implementation-dependent approximation of `x1_i` raised to
    the power of `x2_i` for each element `x1_i` of the input array `x1`, where
    `x2_i` is the corresponding element in the input array `x2`."""
    return _apply_element_wise('pow', x1, x2=x2, out=out)


def real(x1, out=None):
    """Returns the real part of each element `x_i` of the input array `x`."""
    return _apply_element_wise('real', x1, out=out)


def reciprocal(x1, out=None):
    """Returns the reciprocal for each element `x_i` of the input array `x`."""
    return _apply_element_wise('reciprocal', x1, out=out)


def remainder(x1, x2, out=None):
    """Calculates the remainder of dividing each element `x1_i` of the input
    array `x1` by the respective element `x2_i` of the input array `x2`.

    The result has the same sign as the dividend `x1`, and the magnitude
    is less than the magnitude of the divisor `x2`. This is often called
    the "Euclidean modulo" operation.
    """
    return _apply_element_wise('remainder', x1, x2=x2, out=out)


def round(x1, out=None):
    """Rounds each element `x_i` of the input array `x` to the nearest integer.

    Halfway cases (i.e., numbers with a fractional part of `0.5`) are rounded
    to the nearest even integer.
    """
    return _apply_element_wise('round', x1, out=out)


def sign(x1, out=None):
    """Returns an indication of the sign of each element `x_i` of the input
    array `x`.

    The returned array has the same shape as `x`.
    """
    return _apply_element_wise('sign', x1, out=out)


def signbit(x1, out=None):
    """Determines whether the sign bit is set for each element `x_i` of the
    input array `x`"""
    return _apply_element_wise('signbit', x1, out=out)


def sin(x1, out=None):
    """Calculates an implementation-dependent approximation to the sine for
    each element `x_i` of the input array `x`."""
    return _apply_element_wise('sin', x1, out=out)


def sinh(x1, out=None):
    """Calculates an implementation-dependent approximation to the hyperbolic
    sine for each element `x_i` in the input array `x`."""
    return _apply_element_wise('sinh', x1, out=out)


def sqrt(x1, out=None):
    """Calculates the square root for each element `x_i` of the input array
    `x`."""
    return _apply_element_wise('sqrt', x1, out=out)


def square(x1, out=None):
    """Calculates the square of each element `x_i` (i.e., `x_i * x_i`) of the
    input array `x`"""
    return _apply_element_wise('square', x1, out=out)


def subtract(x1, x2, out=None):
    """Calculates the difference for each element `x1_i` of the input array
    `x1` with the respective element `x2_i` of the input array `x2`."""
    return _apply_element_wise('subtract', x1, x2=x2, out=out)


def tan(x1, out=None):
    """Calculates an implementation-dependent approximation to the tangent for
    each element `x_i` of the input array `x`."""
    return _apply_element_wise('tan', x1, out=out)


def tanh(x1, out=None):
    """Calculates an implementation-dependent approximation to the hyperbolic
    tangent for each element `x_i` in the input array `x`."""
    return _apply_element_wise('tanh', x1, out=out)


def trunc(x1, out=None):
    """Rounds each element `x_i` of the input array `x` to the nearest integer
    towards zero."""
    return _apply_element_wise('trunc', x1, out=out)
