# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Ufuncs to be used by spaces."""

from __future__ import absolute_import, division, print_function

import numpy as np

__all__ = ()

_UFUNCS_NUMPY_110 = {
    'abs',
    'absolute',
    'add',
    'arccos',
    'arccosh',
    'arcsin',
    'arcsinh',
    'arctan',
    'arctan2',
    'arctanh',
    'bitwise_and',
    'bitwise_not',
    'bitwise_or',
    'bitwise_xor',
    'cbrt',
    'ceil',
    'conj',
    'conjugate',
    'copysign',
    'cos',
    'cosh',
    'deg2rad',
    'degrees',
    'divide',
    'equal',
    'exp',
    'exp2',
    'expm1',
    'fabs',
    'floor',
    'floor_divide',
    'fmax',
    'fmin',
    'fmod',
    'frexp',
    'greater',
    'greater_equal',
    'hypot',
    'invert',
    'isfinite',
    'isinf',
    'isnan',
    'ldexp',
    'left_shift',
    'less',
    'less_equal',
    'log',
    'log10',
    'log1p',
    'log2',
    'logaddexp',
    'logaddexp2',
    'logical_and',
    'logical_not',
    'logical_or',
    'logical_xor',
    'maximum',
    'minimum',
    'mod',
    'modf',
    'multiply',
    'negative',
    'nextafter',
    'not_equal',
    'power',
    'rad2deg',
    'radians',
    'reciprocal',
    'remainder',
    'right_shift',
    'rint',
    'sign',
    'signbit',
    'sin',
    'sinh',
    'spacing',
    'sqrt',
    'square',
    'subtract',
    'tan',
    'tanh',
    'true_divide',
    'trunc',
}

_UFUNCS_NUMPY_111 = _UFUNCS_NUMPY_110
_UFUNCS_NUMPY_112 = _UFUNCS_NUMPY_111 | {'float_power'}
# NumPy 1.13 also defines `isnat`, which is for datetime type arrays only
# and thus not useful for us
_UFUNCS_NUMPY_113 = _UFUNCS_NUMPY_112 | {'divmod', 'heaviside', 'positive'}
_UFUNCS_NUMPY_114 = _UFUNCS_NUMPY_113
_UFUNCS_NUMPY_115 = _UFUNCS_NUMPY_114 | {'gcd', 'lcm'}
# NumPy 1.16 introduces `matmul`, but it is not like other ufuncs that
# operate elementwise, thus we don't include it
_UFUNCS_NUMPY_116 = _UFUNCS_NUMPY_115

_UFUNCS = {
    '1.10': _UFUNCS_NUMPY_110,
    '1.11': _UFUNCS_NUMPY_111,
    '1.12': _UFUNCS_NUMPY_112,
    '1.13': _UFUNCS_NUMPY_113,
    '1.14': _UFUNCS_NUMPY_114,
    '1.15': _UFUNCS_NUMPY_115,
    '1.16': _UFUNCS_NUMPY_116,
}

_np_ver_maj_min = '.'.join(np.__version__.split('.')[:2])
UFUNCS = _UFUNCS[_np_ver_maj_min]

def _ufunc_call_11(ufunc, domain, x, out=None):
    from odl.space.pspace import ProductSpace
    from odl.space.base_tensors import TensorSpace

    if isinstance(domain, TensorSpace):
        return ufunc(x, out=out)
    elif isinstance(domain, ProductSpace):
        if out is None:
            return [ufunc(xi) for xi in x]
        else:
            for xi, oi in zip(x, out):
                ufunc(xi, out=oi)
            return out
    else:
        raise RuntimeError


def _ufunc_call_12(ufunc, domain, x, out=None):
    from odl import ProductSpace
    from odl.space.base_tensors import TensorSpace

    if isinstance(domain, TensorSpace):
        if out is None:
            return ufunc(x)
        else:
            ufunc(x, out=(out[0], out[1]))
            return out
    elif isinstance(domain, ProductSpace):
        if out is None:
            return [ufunc(xi) for xi in x]
        else:
            for xi, oi in zip(x, out):
                ufunc(xi, out=(oi[0], oi[1]))
            return out
    else:
        raise RuntimeError


def _ufunc_call_21(ufunc, domain, x, out=None):
    from odl import ProductSpace
    from odl.space.base_tensors import TensorSpace

    if isinstance(domain[0], TensorSpace):
        return ufunc(x[0], x[1], out=out)
    elif isinstance(domain[0], ProductSpace):
        if out is None:
            return [ufunc(xi[0], xi[1]) for xi in x]
        else:
            for xi, oi in zip(x, out):
                ufunc(xi[0], xi[1], out=oi)
            return out
    else:
        raise RuntimeError


def _ufunc_call_22(ufunc, domain, x, out=None):
    from odl import ProductSpace
    from odl.space.base_tensors import TensorSpace

    if isinstance(domain[0], TensorSpace):
        if out is None:
            return ufunc(x[0], x[1])
        else:
            ufunc(x[0], x[1], out=(out[0], out[1]))
            return out
    elif isinstance(domain[0], ProductSpace):
        if out is None:
            return [ufunc(xi[0], xi[1]) for xi in x]
        else:
            for xi, oi in zip(x, out):
                ufunc(xi[0], xi[1], out=(oi[0], oi[1]))
            return out
    else:
        raise RuntimeError
