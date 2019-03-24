# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Operators based on NumPy UFuncs."""

import warnings

import numpy as np

from odl._ufunc._ufuncs import (
    _ufunc_call_11, _ufunc_call_12, _ufunc_call_21, _ufunc_call_22)

LINEAR_UFUNCS = {
    'negative', 'degrees', 'rad2deg', 'radians', 'deg2rad', 'add', 'subtract'
}

def _ufunc_op_range(domain, nin, nout, types_dict):
    """Infer the range of a ufunc operator."""
    if nin == 1:
        try:
            dom_type = domain.dtype.char
            dom_base = domain
        except AttributeError:
            # TODO(kohr-h): better error message
            raise ValueError('bad `domain`')
    elif nin == 2:
        try:
            dom_type = domain[0].dtype.char + domain[1].dtype.char
            dom_base = domain[0]
        except (TypeError, IndexError, AttributeError):
            # TODO(kohr-h): better error message
            raise ValueError('bad `domain`')
    else:
        raise NotImplementedError

    ran_type = types_dict[dom_type]

    if nout == 1:
        return dom_base.astype(ran_type)
    elif nout == 2:
        return dom_base.astype(ran_type[0]) * dom_base.astype(ran_type[1])
    else:
        raise NotImplementedError


def ufunc_op___init__(self, domain):
    """Initialize a new instance.

    Parameters
    ----------
    domain : `TensorSpace` or `ProductSpace`
        Space of elements to which this ufunc operator can be applied.
    """
    from odl import Operator

    range = _ufunc_op_range(
        domain, self.ufunc.nin, self.ufunc.nout, self.types
    )
    Operator.__init__(
        self, domain, range, linear=self.ufunc.__name__ in LINEAR_UFUNCS
    )


def ufunc_op_derivative(name):
    from odl import MultiplyOperator, Operator

    if name == 'sin':
        def derivative(self, point):
            cos = ufunc_op_cls('cos')(self.domain)
            return MultiplyOperator(self.domain, cos(point))
    elif name == 'cos':
        def derivative(self, point):
            sin = ufunc_op_cls('sin')(self.domain)
            return MultiplyOperator(self.domain, -sin(point))
    elif name == 'tan':
        def derivative(self, point):
            tan = self
            return MultiplyOperator(self.domain, 1 + tan(point) ** 2)
    elif name == 'sqrt':
        def derivative(self, point):
            sqrt = self
            return MultiplyOperator(self.domain, 0.5 / sqrt(point))
    elif name == 'square':
        def derivative(self, point):
            return MultiplyOperator(self.domain, 2.0 * point)
    elif name == 'log':
        def derivative(self, point):
            return MultiplyOperator(self.domain, 1.0 / point)
    elif name == 'exp':
        def derivative(self, point):
            exp = self
            return MultiplyOperator(self.domain, exp(point))
    elif name == 'reciprocal':
        def derivative(self, point):
            reciprocal = self
            return MultiplyOperator(self.domain, reciprocal(point) ** 2)
    elif name == 'sinh':
        def derivative(self, point):
            cosh = ufunc_op_cls('cosh')(self.domain)
            return MultiplyOperator(self.domain, cosh(point))
    elif name == 'cosh':
        def derivative(self, point):
            sinh = ufunc_op_cls('sinh')(self.domain)
            return MultiplyOperator(self.domain, sinh(point))
    else:
        # Fallback to default
        derivative = Operator.derivative

    derivative.__doc__ = 'Return the derivative operator.'
    return derivative


in1_default = [-1.0, 1.0, 2.0]
in2_default = [[-1.0, 1.0, 2.0], [0.5, -1.0, 2.0]]
UFUNC_INPUT_FOR_DOC = {
    'abs': {'type': 'd', 'input': in1_default},  # = absolute
    'absolute': {'type': 'd', 'input': in1_default},
    'add': {'type': 'dd', 'input': in2_default},
    'arccos': {'type': 'd', 'input': [-1.0, 0.0, 1.0]},
    'arccosh': {'type': 'd', 'input': [1.0, 1.5, 2.0]},
    'arcsin': {'type': 'd', 'input': [-1.0, 0.0, 1.0]},
    'arcsinh': {'type': 'd', 'input': in1_default},
    'arctan': {'type': 'd', 'input': in1_default},
    'arctan2': {'type': 'dd', 'input': in2_default},
    'arctanh': {'type': 'd', 'input': [-0.5, 0.0, 0.5]},
    'bitwise_and': {'type': 'll', 'input': [[-1, 1, 0], [1, 0, 0]]},
    'bitwise_not': {'type': 'l', 'input': [[-2, 0, 1]]},  # = invert
    'bitwise_or': {'type': 'll', 'input': [[-1, 1, 0], [1, 0, 0]]},
    'bitwise_xor': {'type': 'll', 'input': [[-1, 1, 0], [1, 0, 0]]},
    'cbrt': {'type': 'd', 'input': in1_default},
    'ceil': {'type': 'd', 'input': [-0.5, 0.0, 0.5]},
    'conj': {'type': 'D',
             'input': [-0.5, 0.0 + 1.0j, 0.5 - 0.5j]},  # = conjugate
    'conjugate': {'type': 'D', 'input': [-0.5, 0.0 + 1.0j, 0.5 - 0.5j]},
    'copysign': {'type': 'dd', 'input': in2_default},
    'cos': {'type': 'd', 'input': in1_default},
    'cosh': {'type': 'd', 'input': in1_default},
    'deg2rad': {'type': 'd', 'input': in1_default},
    'degrees': {'type': 'd', 'input': in1_default},
    'divide': {'type': 'dd', 'input': in2_default},  # = true_divide
    'divmod': {'type': 'll', 'input': [[-1, 2, 3], [2, 2, 2]]},
    'equal': {'type': 'dd', 'input': in2_default},
    'exp': {'type': 'd', 'input': in1_default},
    'exp2': {'type': 'd', 'input': in1_default},
    'expm1': {'type': 'd', 'input': in1_default},
    'fabs': {'type': 'd', 'input': in1_default},
    'float_power': {'type': 'dd',
                    'input': [[2.0, 1.0, 2.0], [0.5, -1.0, 2.0]]},
    'floor': {'type': 'd', 'input': in1_default},
    'floor_divide': {'type': 'll', 'input': [[-1, 2, 3], [2, 2, 2]]},
    'fmax': {'type': 'dd', 'input': in2_default},
    'fmin': {'type': 'dd', 'input': in2_default},
    'fmod': {'type': 'dd', 'input': [[0.5, -1.0, 2.0], [2.0, 1.0, 2.0]]},
    'frexp': {'type': 'd', 'input': in1_default},
    'gcd': {'type': 'll', 'input': [[-2, 6, 0], [2, 9, 0]]},
    'greater': {'type': 'dd', 'input': in2_default},
    'greater_equal': {'type': 'dd', 'input': in2_default},
    'heaviside': {'type': 'dd', 'input': [[0.5, 0.0, 2.0], [2.0, 1.0, 2.0]]},
    'hypot': {'type': 'dd', 'input': in2_default},
    'invert': {'type': 'l', 'input': [-2, 0, 1]},
    'isfinite': {'type': 'd', 'input': [1.0, float('inf'), float('nan')]},
    'isinf': {'type': 'd', 'input': [1.0, float('inf'), float('nan')]},
    'isnan': {'type': 'd', 'input': [1.0, float('inf'), float('nan')]},
    'lcm': {'type': 'll', 'input': [[-2, 6, 0], [2, 9, 0]]},
    'ldexp': {'type': 'dl', 'input': [[0.5, -1.0, 2.0], [2, 1, -2]]},
    'left_shift': {'type': 'll', 'input': [[-2, 1, 2], [2, 1, 0]]},
    'less': {'type': 'dd', 'input': in2_default},
    'less_equal': {'type': 'dd', 'input': in2_default},
    'log': {'type': 'd', 'input': [0.5, 1.0, 2.0]},
    'log10': {'type': 'd', 'input': [0.5, 1.0, 2.0]},
    'log1p': {'type': 'd', 'input': [0.0, 0.5, 1.0]},
    'log2': {'type': 'd', 'input': [0.5, 1.0, 2.0]},
    'logaddexp': {'type': 'dd', 'input': in2_default},
    'logaddexp2': {'type': 'dd', 'input': in2_default},
    'logical_and': {'type': '??', 'input': [[True, False, True, False],
                                            [True, True, False, False]]},
    'logical_not': {'type': '?', 'input': [True, False]},
    'logical_or': {'type': '??', 'input': [[True, False, True, False],
                                           [True, True, False, False]]},
    'logical_xor': {'type': '??', 'input': [[True, False, True, False],
                                            [True, True, False, False]]},
    'maximum': {'type': 'dd', 'input': in2_default},
    'minimum': {'type': 'dd', 'input': in2_default},
    'mod': {'type': 'll', 'input': [[-1, 2, 3], [2, 2, 2]]},  # = remainder
    'modf': {'type': 'd', 'input': in1_default},
    'multiply': {'type': 'dd', 'input': in2_default},
    'negative': {'type': 'd', 'input': in1_default},
    'nextafter': {'type': 'dd', 'input': in2_default},
    'not_equal': {'type': 'dd', 'input': in2_default},
    'positive': {'type': 'd', 'input': in1_default},
    'power': {'type': 'dd', 'input': in2_default},
    'rad2deg': {'type': 'd', 'input': in1_default},
    'radians': {'type': 'd', 'input': in1_default},
    'reciprocal': {'type': 'd', 'input': [-0.5, 1.0, 2.0]},
    'remainder': {'type': 'll', 'input': [[-1, 2, 3], [2, 2, 2]]},
    'right_shift': {'type': 'll', 'input': [[-2, 1, 2], [2, 1, 0]]},
    'rint': {'type': 'd', 'input': in1_default},
    'sign': {'type': 'd', 'input': in1_default},
    'signbit': {'type': 'd', 'input': in1_default},
    'sin': {'type': 'd', 'input': in1_default},
    'sinh': {'type': 'd', 'input': in1_default},
    'spacing': {'type': 'd', 'input': in1_default},
    'sqrt': {'type': 'd', 'input': [0.0, 0.5, 1.0]},
    'square': {'type': 'd', 'input': in1_default},
    'subtract': {'type': 'dd', 'input': in2_default},
    'tan': {'type': 'd', 'input': in1_default},
    'tanh': {'type': 'd', 'input': in1_default},
    'true_divide': {'type': 'dd', 'input': in2_default},
    'trunc': {'type': 'd', 'input': [-0.5, 0.0, 1.5]},
}


def ufunc_op_cls(name):
    """Dynamically generate a ufunc operator class for a given ufunc name."""
    from odl import Operator, tensor_space

    # --- Get ufunc, map to impl --- #

    ufunc = getattr(np, name)
    assert isinstance(ufunc, np.ufunc)

    if ufunc.nin == 1 and ufunc.nout == 1:
        _call_impl = _ufunc_call_11
    elif ufunc.nin == 1 and ufunc.nout == 2:
        _call_impl = _ufunc_call_12
    elif ufunc.nin == 2 and ufunc.nout == 1:
        _call_impl = _ufunc_call_21
    elif ufunc.nin == 2 and ufunc.nout == 2:
        _call_impl = _ufunc_call_22
    else:
        raise NotImplementedError

    def _call(self, x, out=None):
        return _call_impl(ufunc, self.domain, x, out)

    # --- Generate docstring --- #

    types = dict(t.split('->') for t in ufunc.types)
    try:
        ufunc_input = UFUNC_INPUT_FOR_DOC[name]
    except KeyError:
        # Unknown ufunc, try to find some appropriate type
        if 'd' * ufunc.nin in types:
            in_type = 'd' * ufunc.nin
            out_type = types[in_type]
        elif 'l' * ufunc.nin in types:
            in_type = 'l' * ufunc.nin
            out_type = types[in_type]
        elif '?' * ufunc.nin in types:
            in_type = '?' * ufunc.nin
            out_type = types[in_type]
        else:
            in_type = 'd' * ufunc.nin
            out_type = 'd' * ufunc.nout

        if ufunc.nin == 1:
            inp = np.array(in1_default, dtype=in_type).tolist()
        else:
            inp = [
                np.array(in2_default[0], dtype=in_type[0]).tolist(),
                np.array(in2_default[1], dtype=in_type[1]).tolist(),
            ]

        warnings.warn(
            'ufunc {!r} not known, assuming default input type {!r}'
            ''.format(name, in_type)
        )

    else:
        in_type = ufunc_input['type']
        inp = ufunc_input['input']
        out_type = types[in_type]

    if ufunc.nin == 1:
        space_in = tensor_space(len(inp), dtype=in_type)
        space_str = 'odl.{!r}'.format(space_in)
        result = ufunc(inp)
    elif ufunc.nin == 2:
        space_in = (
            tensor_space(len(inp[0]), dtype=in_type[0])
            * tensor_space(len(inp[1]), dtype=in_type[1])
        )
        space_str = 'odl.{!r} * odl.{!r}'.format(
            space_in[0], space_in[1]
        )
        result = ufunc(*inp)

    if ufunc.nout == 1:
        outp = result.astype(out_type, copy=False)
    elif ufunc.nout == 2:
        outp = np.empty(2, dtype=object)
        outp[0] = result[0]
        outp[1] = result[1]

    summary = ufunc.__doc__.splitlines()[2]
    docstring = """
    {summary}

    Examples
    --------
    >>> space = {space}
    >>> op = odl.ufunc_ops.{name}(space)
    >>> op({arg})
    {result!r}
    """.format(
        summary=summary, space=space_str, name=name, arg=inp, result=outp
    )

    # --- Make class --- #

    attrs = {
        'ufunc': ufunc,
        'types': types,
        '__init__': ufunc_op___init__,
        '_call': _call,
        '__doc__': docstring,
        'derivative': ufunc_op_derivative(name),
    }
    return type(name, (Operator,), attrs)


ufunc_ops = type(
    'ufunc_ops', (object,), {'__getattr__': staticmethod(ufunc_op_cls)}
)()


# --- Functionals --- #


def ufunc_func___init__(self, domain):
    """Initialize a new instance.

    Parameters
    ----------
    domain : `Field`
        Scalar field to which this ufunc functional can be applied.
    """
    from odl.solvers.functional import Functional

    Functional.__init__(
        self, domain, linear=self.ufunc.__name__ in LINEAR_UFUNCS
    )


def ufunc_func_gradient(name):
    from odl.solvers.functional import (
        Functional, ConstantFunctional, FunctionalQuotient, ScalingFunctional)

    if name == 'sin':
        def gradient(self):
            cos = ufunc_func_cls('cos')(self.domain)
            return cos
    elif name == 'cos':
        def gradient(self):
            sin = ufunc_func_cls('sin')(self.domain)
            return -sin
    elif name == 'tan':
        def gradient(self):
            tan = self
            square = ufunc_func_cls('square')(self.domain)
            return 1 + square * tan
    elif name == 'sqrt':
        def gradient(self):
            sqrt = self
            return FunctionalQuotient(
                ConstantFunctional(self.domain, 0.5), sqrt
            )
    elif name == 'square':
        def gradient(self):
            return ScalingFunctional(self.domain, 2.0)
    elif name == 'log':
        def gradient(self):
            reciprocal = ufunc_func_cls('reciprocal')(self.domain)
            return reciprocal
    elif name == 'exp':
        def gradient(self):
            exp = self
            return exp
    elif name == 'reciprocal':
        def gradient(self):
            square = ufunc_func_cls('square')(self.domain)
            return FunctionalQuotient(
                ConstantFunctional(self.domain, -1.0), square
            )
    elif name == 'sinh':
        def gradient(self):
            cosh = ufunc_func_cls('cosh')(self.domain)
            return cosh
    elif name == 'cosh':
        def gradient(self):
            sinh = ufunc_func_cls('sinh')(self.domain)
            return sinh
    else:
        # Fallback to default
        gradient = Functional.gradient

    gradient.__doc__ = 'Return the gradient operator.'
    return gradient


def ufunc_func_cls(name):
    """Dynamically generate a ufunc functional class for a given ufunc name."""
    from odl import RealNumbers, ComplexNumbers, Integers
    from odl.solvers.functional import Functional

    # --- Get ufunc, map to impl --- #

    ufunc = getattr(np, name)
    assert isinstance(ufunc, np.ufunc)

    if ufunc.nin != 1 or ufunc.nout !=1:
        raise ValueError(
            'ufunc functionals only defined for ufuncs with 1 input and '
            '1 output'
        )

    def _call(self, x):
        return ufunc(x)

    # --- Generate docstring --- #

    types = dict(t.split('->') for t in ufunc.types)
    try:
        ufunc_input = UFUNC_INPUT_FOR_DOC[name]
    except KeyError:
        raise ValueError('ufunc `{}` not supported'.format(name))

    in_type = ufunc_input['type']
    inp = ufunc_input['input'][-1]
    out_type = types[in_type]

    if in_type == 'd':
        domain = RealNumbers()
    elif in_type == 'D':
        domain = ComplexNumbers()
    elif in_type in {'l', '?'}:
        domain = Integers()
    else:
        raise RuntimeError

    space_str = 'odl.{!r}'.format(domain)

    result = ufunc(inp)
    outp = domain.astype(out_type).element(result)

    summary = ufunc.__doc__.splitlines()[2]
    docstring = """
    {summary}

    Examples
    --------
    >>> space = {space}
    >>> func = odl.ufunc_funcs.{name}(space)
    >>> func({arg})
    {result!r}
    """.format(
        summary=summary, space=space_str, name=name, arg=inp, result=outp
    )

    # --- Make class --- #

    attrs = {
        'ufunc': ufunc,
        'types': types,
        '__init__': ufunc_func___init__,
        '_call': _call,
        '__doc__': docstring,
        'gradient': property(ufunc_func_gradient(name)),
    }
    return type(name, (Functional,), attrs)


ufunc_funcs = type(
    'ufunc_funcs', (object,), {'__getattr__': staticmethod(ufunc_func_cls)}
)()
