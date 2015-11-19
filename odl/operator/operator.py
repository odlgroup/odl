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

"""Abstract mathematical (linear) operators."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object, super
from odl.util.utility import with_metaclass

# External module imports
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Number, Integral

# ODL imports
from odl.set.space import LinearSpace, UniversalSpace
from odl.set.sets import Set, UniversalSet, Field

__all__ = ('Operator', 'OperatorComp', 'OperatorSum',
           'OperatorLeftScalarMult', 'OperatorRightScalarMult',
           'FunctionalLeftVectorMult',
           'OperatorLeftVectorMult', 'OperatorRightVectorMult',
           'OperatorPointwiseProduct')


def _default_call_out_of_place(op, x, **kwargs):
    """Default out-of-place evaluation.

    Parameters
    ----------
    x : ``domain`` element
        An object in the operator domain. The operator is applied
        to it.

    Returns
    -------
    out : ``range`` element
        An object in the operator range. The result of an operator
        evaluation.
    """
    print('use _default_call_out_of_place')
    out = op.range.element()
    op._call_in_place(x, out, **kwargs)
    return out


def _default_call_in_place(op, x, out, **kwargs):
    """Default in-place evaluation using ``Operator._call()``.

    Parameters
    ----------
    x : ``domain`` element
        An object in the operator domain. The operator is applied
        to it.

    out : ``range`` element
        An object in the operator range. The result of an operator
        evaluation.

    Returns
    -------
    `None`
    """
    print('use _default_call_in_place')
    out.assign(op._call_out_of_place(x, **kwargs))


def _dispatch_call_args(cls, unbound_call=None):
    """Check the arguments of ``_call()`` in ``cls`` for conformity.

    The ``_call()`` method of :class:`Operator` is allowed to have the
    following signatures:

    Python 2 and 3:
        - ``_call(self, x)``
        - ``_call(self, vec, out)``
        - ``_call(self, x, out=None)``

    Python 3 only:
        - ``_call(self, x, *, out=None)`` (``out`` as keyword-only
          argument)

    For disambiguation, the instance name (the first argument) **must**
    be 'self'.

    The name of the ``out`` argument **must** be 'out', the second
    argument may have any name.

    Additional variable ``**kwargs`` are also allowed.

    Not allowed:
        - ``_call(self)`` -- No arguments except instance:
        - ``_call(x)`` -- 'self' missing, i.e. ``@staticmethod``
        - ``_call(cls, x)``  -- 'self' missing, i.e. ``@classmethod``
        - ``_call(self, out, x)`` -- ``out`` as second argument
        - ``_call(self, *x)`` -- Variable arguments
        - ``_call(self, x, y, out=None)`` -- more positional arguments
        - ``_call(self, x, out=False)`` -- default other than `None` for
          ``out``

    In particular, static or class methods are not allowed.
    """

    import inspect
    import sys

    py3 = (sys.version_info.major > 2)

    py2_specs = ('_call(self, x[, **kwargs])',
                 '_call(self, x, out[, **kwargs])',
                 '_call(self, x, out=None[, **kwargs])')

    py3only_specs = ('_call(self, x, *, out=None[, **kwargs])',)

    spec_msg = "\nPossible signatures are ('[, **kwargs]' means optional):\n\n"
    spec_msg += '\n'.join(py2_specs)
    if py3:
        spec_msg += '\n' + '\n'.join(py3only_specs)
    spec_msg += '\n\nStatic or class methods are not allowed.'

    if unbound_call is None:
        call = getattr(cls, '_call', None)
        if call is None:
            raise ValueError("class {!r} has no '_call' method."
                             "".format(cls))
        # Static and class methods are not allowed
        if isinstance(cls.__dict__['_call'], staticmethod):
            raise TypeError("'{}._call()' is a static method. " + spec_msg)
        elif isinstance(cls.__dict__['_call'], classmethod):
            raise TypeError("'{}._call()' is a class method. " + spec_msg)
    else:
        call = unbound_call

    if py3:
        # support kw-only args and annotations
        spec = inspect.getfullargspec(call)
        kw_only = spec.kwonlyargs
        kw_only_defaults = spec.kwonlydefaults
    else:
        spec = inspect.getargspec(call)
        kw_only = ()
        kw_only_defaults = {}

    print(spec)

    pos_args = spec.args
    if unbound_call is not None:
        # Add 'self' to positional arg list to satisfy the checker
        pos_args.insert(0, 'self')

    pos_defaults = spec.defaults
    varargs = spec.varargs

    out_optional = False

    # Variable args are not allowed
    if varargs is not None:
        raise ValueError("Variable arguments not allowed in '_call()'." +
                         spec_msg)

    if len(pos_args) not in (2, 3):
        raise ValueError("Bad signature of '_call()'. " + spec_msg)

    # 'self' must be the first argument
    elif pos_args[0] != 'self':
        raise ValueError("'self' is not the first argument in '_call()'." +
                         spec_msg)

    true_pos_args = pos_args[1:]
    if len(true_pos_args) == 1:  # 'out' kw-only
        if 'out' in true_pos_args:  # 'out' positional and 'x' kw-only -> no
            raise ValueError("'out' cannot be only positional argument except "
                             "'self' in '_call()'." + spec_msg)
        else:
            if len(kw_only) == 0:
                has_out = False
            elif len(kw_only) == 1:
                if 'out' not in kw_only:
                    raise ValueError("Output parameter must be called 'out'"
                                     " in '_call()'." + spec_msg)
                else:
                    has_out = True
                    if kw_only_defaults['out'] is not None:
                        raise ValueError("'out' can only default to None in "
                                         "'_call()'." + spec_msg)
                    else:
                        out_optional = True
            else:
                raise ValueError("Bad signature of '_call()'." + spec_msg)

    elif len(true_pos_args) == 2:  # Both args positional
        if true_pos_args[0] == 'out':  # 'out' must come second
            py3_txt = 'or keyword-only ' if py3 else ''
            raise ValueError("'out' can only be the second positional "
                             "argument " + py3_txt + "in '_call()'." +
                             spec_msg)
        elif true_pos_args[1] != 'out':  # 'out' must be 'out'
            raise ValueError("Output parameter must be called 'out'"
                             " in '_call()'." + spec_msg)
        else:
            has_out = True
            out_optional = bool(pos_defaults)
            if pos_defaults and pos_defaults[-1] is not None:
                raise ValueError("'out' can only default to None in "
                                 "'_call()'." + spec_msg)

    else:  # Too many positional args
        raise ValueError("Bad signature of '_call()'. " + spec_msg)

    print('has_out: ', has_out)
    print('out_optional: ', out_optional)
    return has_out, out_optional


class Operator(with_metaclass(ABCMeta, object)):

    """Abstract operator.

    **Abstract attributes and methods**

    :class:`Operator` is an **abstract** class, i.e. it can only be
    subclassed, not used directly.

    Any subclass of :class:`Operator` **must** have the following
    attributes:

    ``domain`` : :class:`~odl.Set`
        The set of elements this operator can be applied to

    ``range`` : :class:`~odl.Set`
        The set this operator maps to

    It is **highly** recommended to call
    ``super().__init__(dom, ran)`` (Note: add
    ``from builtins import super`` in Python 2) in the ``__init__()``
    method of any subclass, where ``dom`` and ``ran`` are the arguments
    specifying domain and range of the new
    operator. In that case, the attributes :attr:`Operator.domain` and
    :attr:`Operator.range` are automatically provided by
    :class:`Operator`.

    In addition, any subclass **must** implement **at least one** of the
    methods ``_apply()`` and ``_call()``, which are explained in the
    following.

    **In-place evaluation:** ``_apply()``

    In-place evaluation means that the operator is applied, and the
    result is written to an existing element provided as an additional
    argument. In this case, a subclass has to implement the method

        ``_apply(self, x, out)  <==>  out <-- operator(x)``

    **Parameters:**

    x : :attr:`Operator.domain` element
        An object in the operator domain to which the operator is
        applied.

    out : :attr:`Operator.range` element
        An object in the operator range to which the result of the
        operator evaluation is written.

    **Returns:**

    `None`

    **Out-of-place evaluation:** ``_call()``

    Out-of-place evaluation means that the operator is applied,
    and the result is written to a **new** element which is returned.
    In this case, a subclass has to implement the method

        ``_call(self, x)  <==>  operator(x)``

    **Parameters:**

    x : :attr:`Operator.domain` element
        An object in the operator domain to which the operator is
        applied.

    out : :attr:`Operator.range` element
        An object in the operator range to which the result of the
        operator evaluation is written.

    Notes
    -----
    If not both ``_apply()`` and ``_call()`` are implemented and the
    :attr:`Operator.range` is a :class:`~odl.LinearSpace`, a default
    implementation of the respective other is provided.
    """

    def __new__(cls, *args, **kwargs):
        """Create a new instance."""
        instance = super().__new__(cls)

        call_has_out, call_out_optional = _dispatch_call_args(cls)
        if not call_has_out:
            # Out-of-place _call
            instance._call_in_place = partial(_default_call_in_place,
                                              instance)
            instance._call_out_of_place = instance._call
        elif call_out_optional:
            # Dual-use _call
            instance._call_in_place = instance._call
            instance._call_out_of_place = instance._call
        else:
            # In-place only _call
            instance._call_in_place = instance._call
            instance._call_out_of_place = partial(_default_call_out_of_place,
                                                  instance)
        return instance

    def __init__(self, domain, range, linear=False):
        """Initialize a new instance.

        Parameters
        ----------
        dom : :class:`~odl.Set`
            The domain of this operator, i.e., the set of elements to
            which this operator can be applied

        ran : :class:`~odl.Set`
            The range of this operator, i.e., the set this operator
            maps to
        """
        if not isinstance(domain, Set):
            raise TypeError('domain {!r} not a `Set` instance.'.format(domain))
        if not isinstance(range, Set):
            raise TypeError('range {!r} not a `Set` instance.'.format(range))

        self._domain = domain
        self._range = range
        self._is_linear = bool(linear)
        self._is_functional = isinstance(range, Field)

        if self.is_linear:
            if not isinstance(domain, (LinearSpace, Field)):
                raise TypeError('domain {!r} not a `LinearSpace` or `Field` '
                                'instance.'.format(domain))
            if not isinstance(range, (LinearSpace, Field)):
                raise TypeError('range {!r} not a `LinearSpace` or `Field` '
                                'instance.'.format(range))

    @abstractmethod
    def _call(self, *args, **kwargs):
        """Raw evaluation method. Needs to be overridden by subclasses."""

    @property
    def domain(self):
        """The domain of this operator."""
        return self._domain

    @property
    def range(self):
        """The range of this operator."""
        return self._range

    @property
    def is_linear(self):
        """`True` if this operator is linear."""
        return self._is_linear

    @property
    def is_functional(self):
        """`True` if the this operator's range is a :class:`~odl.Field`."""
        return self._is_functional

    @property
    def adjoint(self):
        """The operator adjoint."""
        raise NotImplementedError('adjoint not implemented for operator {!r}.'
                                  ''.format(self))

    def derivative(self, point):
        """Return the operator derivative at ``point``."""
        if self.is_linear:
            return self
        else:
            raise NotImplementedError('derivative not implemented for operator'
                                      ' {!r}'.format(self))

    @property
    def inverse(self):
        """Return the operator inverse."""
        raise NotImplementedError('inverse not implemented for operator '
                                  '{!r}'.format(self))

    # Implicitly defined operators
    def __call__(self, x, out=None, **kwargs):
        """``op.__call__(x) <==> op(x)``.

        Implementation of the call pattern ``op(x)`` with the private
        ``_call()`` method and added error checking.

        Parameters
        ----------
        x : :attr:`Operator.domain` element
            An object in the operator domain to which the operator is
            applied. The object is treated as immutable, hence it is
            not modified during evaluation.
        out : :attr:`Operator.range` element, optional
            An object in the operator range to which the result of the
            operator evaluation is written. The result is independent
            of the initial state of this object.
        **kwargs : Further arguments to the function, optional

        Returns
        -------
        elem : :attr:`Operator.range` element
            An object in the operator range, the result of the operator
            evaluation. It is identical to ``out`` if provided.

        Examples
        --------
        >>> from odl import Rn, ScalingOperator
        >>> rn = Rn(3)
        >>> op = ScalingOperator(rn, 2.0)
        >>> x = rn.element([1, 2, 3])

        Out-of-place evaluation:

        >>> op(x)
        Rn(3).element([2.0, 4.0, 6.0])

        In-place evaluation:

        >>> y = rn.element()
        >>> op(x, out=y)
        Rn(3).element([2.0, 4.0, 6.0])
        >>> y
        Rn(3).element([2.0, 4.0, 6.0])
        """
        if x not in self.domain:
            raise TypeError('input {!r} not an element of the domain {!r} '
                            'of {!r}.'
                            ''.format(x, self.domain, self))

        if out is not None:  # In-place evaluation
            if out not in self.range:
                raise TypeError('output {!r} not an element of the range {!r} '
                                'of {!r}.'
                                ''.format(out, self.range, self))

            if self.is_functional:
                raise TypeError('`out` parameter cannot be used'
                                'when range is a field')

            print('in op.__call__: in-place')
            self._call_in_place(x, out=out, **kwargs)
            return out

        else:  # Out-of-place evaluation
            print('in op.__call__: out-of-place')
            result = self._call_out_of_place(x, **kwargs)

            if result not in self.range:
                raise TypeError('result {!r} not an element of the range {!r} '
                                'of {!r}.'
                                ''.format(result, self.range, self))
            return result

    def __add__(self, other):
        """``op.__add__(other) <==> op + other``."""
        return OperatorSum(self, other)

    def __sub__(self, other):
        """``op.__add__(other) <==> op - other``."""
        return OperatorSum(self, -1 * other)

    def __mul__(self, other):
        """``op.__mul__(other) <==> op * other``.

        If ``other`` is an operator, this corresponds to
        operator composition:

            ``op1 * op2 <==> (x --> op1(op2(x))``

        If ``other`` is a scalar, this corresponds to right
        multiplication of scalars with operators:

            ``op * scalar <==> (x --> op(scalar * x))``

        If ``other`` is a vector, this corresponds to right
        multiplication of vectors with operators:

            ``op * vector <==> (x --> op(vector * x))``

        Note that left and right multiplications are generally
        different.

        Parameters
        ----------
        other : {:class:`Operator`, :class:`~odl.LinearSpace.Vector`, scalar}
            :class:`Operator`:
                The :attr:`Operator.domain` of ``other`` must match this
                operator's :attr:`Operator.range`.

            :class:`~odl.LinearSpace.Vector`:
                ``other`` must be an element of this operator's
                :attr:`Operator.domain`.

            scalar:
                The :attr:`Operator.domain` of this operator must be a
                :class:`~odl.LinearSpace` and ``other`` must be an
                element of the ``field`` of this operator's
                :attr:`Operator.domain`.

        Returns
        -------
        mul : :class:`Operator`
            The multiplication operator.

            If ``other`` is an operator, ``mul`` is an
            :class:`OperatorComp`.

            If ``other`` is a scalar, ``mul`` is an
            :class:`OperatorRightScalarMult`.

            If ``other`` is a vector, ``mul`` is an
            :class:`OperatorRightVectorMult`.

        Examples
        --------
        >>> from odl import Rn, IdentityOperator
        >>> rn = Rn(3)
        >>> op = IdentityOperator(rn)
        >>> x = rn.element([1, 2, 3])
        >>> op(x)
        Rn(3).element([1.0, 2.0, 3.0])
        >>> Scaled = op * 3
        >>> Scaled(x)
        Rn(3).element([3.0, 6.0, 9.0])
        """
        if isinstance(other, Operator):
            return OperatorComp(self, other)
        elif isinstance(other, Number):
            # Left multiplication is more efficient, so we can use this in the
            # case of linear operator.
            if self.is_linear:
                return OperatorLeftScalarMult(self, other)
            else:
                return OperatorRightScalarMult(self, other)
        elif isinstance(other, LinearSpace.Vector) and other in self.domain:
            return OperatorRightVectorMult(self, other.copy())
        else:
            return NotImplemented

    def __matmul__(self, other):
        """``op.__matmul__(other) <==> op @ other``.

        See :meth:`Operator.__mul__`
        """
        return self.__mul__(other)

    def __rmul__(self, other):
        """``op.__rmul__(s) <==> s * op``.

        If ``other`` is an :class:`Operator`, this corresponds to
        operator composition:

        ``op1 * op2 <==> (x --> op1(op2(x)))``

        If ``other`` is a scalar, this corresponds to left
        multiplication of scalars with operators:

        ``scalar * op <==> (x --> scalar * op(x))``

        If ``other`` is a vector, this corresponds to left
        multiplication of vector with operators:

        ``vector * op <==> (x --> vector * op(x))``

        Note that left and right multiplications are generally
        different.

        Parameters
        ----------
        other : {:class:`Operator`, :class:`~odl.LinearSpace.Vector`, scalar}
            :class:`Operator`:
                The :attr:`Operator.range` of ``other`` must match this
                operator's :attr:`Operator.domain`

            :class:`~odl.LinearSpace.Vector`:
                ``other`` must be an element of :attr:`Operator.range`.

            scalar:
                :attr:`Operator.range` must be a
                :class:`~odl.LinearSpace` and ``other`` must be an
                element of ``self.range.field``.

        Returns
        -------
        mul : :class:`Operator`
            The multiplication operator.

            If ``other`` is an operator, ``mul`` is an
            :class:`OperatorComp`.

            If ``other`` is a scalar, ``mul`` is an
            :class:`OperatorLeftScalarMult`.

            If ``other`` is a vector, ``mul`` is an
            :class:`OperatorLeftVectorMult`.

        Examples
        --------
        >>> from odl import Rn, IdentityOperator
        >>> rn = Rn(3)
        >>> op = IdentityOperator(rn)
        >>> x = rn.element([1, 2, 3])
        >>> op(x)
        Rn(3).element([1.0, 2.0, 3.0])
        >>> Scaled = 3 * op
        >>> Scaled(x)
        Rn(3).element([3.0, 6.0, 9.0])
        """
        if isinstance(other, Operator):
            return OperatorComp(other, self)
        elif isinstance(other, Number):
            return OperatorLeftScalarMult(self, other)
        elif other in self.range:
            return OperatorLeftVectorMult(self, other.copy())
        elif (isinstance(other, LinearSpace.Vector) and
              other.space.field == self.range):
            return FunctionalLeftVectorMult(self, other.copy())
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        """``op.__rmatmul__(other) <==> other @ op``.

        See :meth:`Operator.__rmul__`
        """
        return self.__rmul__(other)

    def __pow__(self, n):
        """``op.__pow__(s) <==> op**s``.

        This corresponds to the power of an operator:

        ``op ** 1 <==> (x --> op(x))``
        ``op ** 2 <==> (x --> op(op(x)))``
        ``op ** 3 <==> (x --> op(op(op(x))))``
        ...

        Parameters
        ----------
        n : positive `int`
            The power the operator should be taken to.

        Returns
        -------
        pow : :class:`Operator`
            The power of this operator. If ``n == 1``, ``pow`` is
            this operator, for ``n > 1``, a :class:`OperatorComp`

        Examples
        --------
        >>> from odl import Rn, ScalingOperator
        >>> rn = Rn(3)
        >>> op = ScalingOperator(rn, 3)
        >>> x = rn.element([1, 2, 3])
        >>> op(x)
        Rn(3).element([3.0, 6.0, 9.0])
        >>> squared = op**2
        >>> squared(x)
        Rn(3).element([9.0, 18.0, 27.0])
        >>> squared = op**3
        >>> squared(x)
        Rn(3).element([27.0, 54.0, 81.0])
        """
        if isinstance(n, Integral) and n > 0:
            op = self
            while n > 1:
                op = OperatorComp(self, op)
                n -= 1
            return op
        else:
            return NotImplemented

    def __truediv__(self, other):
        """``op.__truediv__(s) <==> op / other``.

        If ``other`` is a scalar, this corresponds to right
        division of operators with scalars:

        ``op / scalar <==> (x --> op(x / scalar))``

        Parameters
        ----------
        other : scalar
            If :attr:`Operator.range` is a :class:`~odl.LinearSpace`,
            ``scalar`` must be an element of this operator's
            ``field``.

        Returns
        -------
        rmul : :class:`OperatorRightScalarMult`
            The 'divided' operator.

        Examples
        --------
        >>> from odl import Rn, IdentityOperator
        >>> rn = Rn(3)
        >>> op = IdentityOperator(rn)
        >>> x = rn.element([3, 6, 9])
        >>> op(x)
        Rn(3).element([3.0, 6.0, 9.0])
        >>> Scaled = op / 3.0
        >>> Scaled(x)
        Rn(3).element([1.0, 2.0, 3.0])
        """
        if isinstance(other, Number):
            return OperatorRightScalarMult(self, 1.0 / other)
        else:
            return NotImplemented

    def __neg__(self):
        """``op.__neg__(s) <==> -op``."""
        return -1 * self

    def __pos__(self):
        """``op.__pos__(s) <==> +op``.

        The operator itself.
        """
        return self

    def __repr__(self):
        """``op.__repr__() <==> repr(op)``.

        The default `repr` implementation. Should be overridden by
        subclasses.
        """
        return '{}: {!r} -> {!r}'.format(self.__class__.__name__, self.domain,
                                         self.range)

    def __str__(self):
        """``op.__str__() <==> str(op)``.

        The default `str` implementation. Should be overridden by
        subclasses.
        """
        return self.__class__.__name__

    # Give a `Operator` a higher priority than any NumPy array type. This
    # forces the usage of `__op__` of `Operator` if the other operand
    # is a NumPy object (applies also to scalars!).
    # Set higher than Space.Vector.__array_priority__ to handle mult with
    # vector properly
    __array_priority__ = 2000000.0


class OperatorSum(Operator):

    """Expression type for the sum of operators.

    ``OperatorSum(op1, op2) <==> (x --> op1(x) + op2(x))``

    The sum is only well-defined for :class:`Operator` instances where
    :attr:`Operator.range` is a :class:`~odl.LinearSpace`.

    """

    # pylint: disable=abstract-method
    def __init__(self, op1, op2, tmp_ran=None, tmp_dom=None):
        """Initialize a new instance.

        Parameters
        ----------
        op1 : :class:`Operator`
            The first summand. Its :attr:`Operator.range` must be a
            :class:`~odl.LinearSpace` or :class:`~odl.Field`.
        op2 : :class:`Operator`
            The second summand. Must have the same
            :attr:`Operator.domain` and :attr:`Operator.range` as
            ``op1``.
        tmp_ran : :attr:`Operator.range` element, optional
            Used to avoid the creation of a temporary when applying the
            operator.
        tmp_dom : :attr:`Operator.domain` element, optional
            Used to avoid the creation of a temporary when applying the
            operator adjoint.
        """
        if op1.range != op2.range:
            raise TypeError('operator ranges {!r} and {!r} do not match.'
                            ''.format(op1.range, op2.range))

        if not isinstance(op1.range, (LinearSpace, Field)):
            raise TypeError('range {!r} not a `LinearSpace` instance.'
                            ''.format(op1.range))

        if op1.domain != op2.domain:
            raise TypeError('operator domains {!r} and {!r} do not match.'
                            ''.format(op1.domain, op2.domain))

        if tmp_ran is not None and tmp_ran not in op1.range:
            raise TypeError('tmp_ran {!r} not an element of the operator '
                            'range {!r}.'.format(tmp_ran, op1.range))

        if tmp_dom is not None and tmp_dom not in op1.domain:
            raise TypeError('tmp_dom {!r} not an element of the operator '
                            'domain {!r}.'.format(tmp_dom, op1.domain))

        super().__init__(op1.domain, op1.range,
                         linear=op1.is_linear and op2.is_linear)
        self._op1 = op1
        self._op2 = op2
        self._tmp_ran = tmp_ran
        self._tmp_dom = tmp_dom

    def _apply(self, x, out):
        """``op._apply(x, out) <==> out <-- op(x)``.

        Examples
        --------
        >>> from odl import Rn, IdentityOperator
        >>> r3 = Rn(3)
        >>> op = IdentityOperator(r3)
        >>> x = r3.element([1, 2, 3])
        >>> out = r3.element()
        >>> OperatorSum(op, op)(x, out)
        Rn(3).element([2.0, 4.0, 6.0])
        >>> out
        Rn(3).element([2.0, 4.0, 6.0])
        """
        # pylint: disable=protected-access
        tmp = (self._tmp_ran if self._tmp_ran is not None
               else self.range.element())
        self._op1._apply(x, out)
        self._op2._apply(x, tmp)
        out += tmp

    def _call(self, x):
        """``op.__call__(x) <==> op(x)``.

        Examples
        --------
        >>> from odl import Rn, ScalingOperator
        >>> r3 = Rn(3)
        >>> A = ScalingOperator(r3, 3.0)
        >>> B = ScalingOperator(r3, -1.0)
        >>> C = OperatorSum(A, B)
        >>> C(r3.element([1, 2, 3]))
        Rn(3).element([2.0, 4.0, 6.0])
        """
        # pylint: disable=protected-access
        return self._op1._call(x) + self._op2._call(x)

    def derivative(self, x):
        """Return the operator derivative at ``x``.

        # TODO: finish doc

        The derivative of a sum of two operators is equal to the sum of
        the derivatives.
        """
        return OperatorSum(self._op1.derivative(x), self._op2.derivative(x))

    @property
    def adjoint(self):
        """The operator adjoint.

        The adjoint of the operator sum is the sum of the operator
        adjoints:

        ``OperatorSum(op1, op2).adjoint ==
        OperatorSum(op1.adjoint, op2.adjoint)``
        """
        if not self.is_linear:
            raise NotImplementedError('Nonlinear operators have no adjoint')

        return OperatorSum(self._op1.adjoint, self._op2.adjoint,
                           self._tmp_dom, self._tmp_ran)

    def __repr__(self):
        """``op.__repr__() <==> repr(op)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self._op1, self._op2)

    def __str__(self):
        """``op.__str__() <==> str(op)``."""
        return '({} + {})'.format(self._op1, self._op2)


class OperatorComp(Operator):

    """Expression type for the composition of operators.

    ``OperatorComp(left, right) <==> (x --> left(right(x)))``

    The composition is only well-defined if
    ``left.domain == right.range``.
    """

    def __init__(self, left, right, tmp=None):
        """Initialize a new :class:`OperatorComp` instance.

        Parameters
        ----------
        left : :class:`Operator`
            The left ("outer") operator
        right : :class:`Operator`
            The right ("inner") operator. Its range must coincide with the
            domain of ``left``.
        tmp : element of the range of ``right``, optional
            Used to avoid the creation of a temporary when applying the
            operator.
        """
        if right.range != left.domain:
            raise TypeError('range {!r} of the right operator {!r} not equal '
                            'to the domain {!r} of the left operator {!r}.'
                            ''.format(right.range, right,
                                      left.domain, left))

        if tmp is not None and tmp not in left.domain:
            raise TypeError('temporary {!r} not an element of the left '
                            'operator domain {!r}.'.format(tmp, left.domain))

        super().__init__(right.domain, left.range,
                         linear=left.is_linear and right.is_linear)
        self._left = left
        self._right = right
        self._tmp = tmp

    def _apply(self, x, out):
        """``op._apply(x, out) <==> out <-- op(x)``."""
        # pylint: disable=protected-access
        tmp = (self._tmp if self._tmp is not None
               else self._right.range.element())
        self._right._apply(x, tmp)
        self._left._apply(tmp, out)

    def _call(self, x):
        """``op.__call__(x) <==> op(x)``."""
        # pylint: disable=protected-access
        return self._left._call(self._right._call(x))

    @property
    def inverse(self):
        """The operator inverse.

        The inverse of the operator composition is the composition of
        the inverses in reverse order:

        ``OperatorComp(left, right).inverse ==``
        ``OperatorComp(right.inverse, left.inverse)``
        """
        return OperatorComp(self._right.inverse, self._left.inverse, self._tmp)

    def derivative(self, point):
        """Return the operator derivative.

        The derivative of the operator composition follows the chain
        rule:

        ``OperatorComp(left, right).derivative(point) ==
        OperatorComp(left.derivative(right(point)), right.derivative(point))``
        """
        left_deriv = self._left.derivative(self._right(point))
        right_deriv = self._right.derivative(point)

        return OperatorComp(left_deriv, right_deriv)

    @property
    def adjoint(self):
        """The operator adjoint.

        The adjoint of the operator composition is the composition of
        the operator adjoints in reverse order:

        ``OperatorComp(left, right).adjoint ==
        OperatorComp(right.adjoint, left.adjoint)``
        """
        if not self.is_linear:
            raise NotImplementedError('Nonlinear operators have no adjoint')

        return OperatorComp(self._right.adjoint, self._left.adjoint,
                            self._tmp)

    def __repr__(self):
        """``op.__repr__() <==> repr(op)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self._left, self._right)

    def __str__(self):
        """``op.__str__() <==> str(op)``."""
        return '{} o {}'.format(self._left, self._right)


class OperatorPointwiseProduct(Operator):

    """Expression type for the pointwise operator mulitplication.

    ``OperatorPointwiseProduct(op1, op2) <==> (x --> op1(x) * op2(x))``
    """

    # pylint: disable=abstract-method
    def __init__(self, op1, op2):
        """Initialize a new instance.

        Parameters
        ----------
        op1 : :class:`Operator`
            The first factor
        op2 : :class:`Operator`
            The second factor. Must have the same domain and range as
            ``op1``.
        """
        if op1.range != op2.range:
            raise TypeError('operator ranges {!r} and {!r} do not match.'
                            ''.format(op1.range, op2.range))

        if not isinstance(op1.range, (LinearSpace, Field)):
            raise TypeError('range {!r} not a `LinearSpace` or `Field` '
                            'instance.'.format(op1.range))

        if op1.domain != op2.domain:
            raise TypeError('operator domains {!r} and {!r} do not match.'
                            ''.format(op1.domain, op2.domain))

        super().__init__(op1.domain, op1.range, linear=False)
        self._op1 = op1
        self._op2 = op2

    def _apply(self, x, out):
        """``op._apply(x, out) <==> out <-- op(x)``."""
        # pylint: disable=protected-access
        tmp = self._op2.range.element()
        self._op1._apply(x, out)
        self._op2._apply(x, tmp)
        out *= tmp

    def _call(self, x):
        """``op.__call__(x) <==> op(x)``."""
        # pylint: disable=protected-access
        return self._op1._call(x) * self._op2._call(x)

    def __repr__(self):
        """``op.__repr__() <==> repr(op)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self._op1, self._op2)

    def __str__(self):
        """``op.__str__() <==> str(op)``."""
        return '{} * {}'.format(self._op1, self._op2)


class OperatorLeftScalarMult(Operator):

    """Expression type for the operator left scalar multiplication.

    ``OperatorLeftScalarMult(op, scalar) <==> (x --> scalar * op(x))``

    The scalar multiplication is well-defined only if ``op.range`` is
    a :class:`~odl.LinearSpace`.
    """

    def __init__(self, op, scalar):
        """Initialize a new :class:`OperatorLeftScalarMult` instance.

        Parameters
        ----------
        op : :class:`Operator`
            The range of ``op`` must be a :class:`~odl.LinearSpace`
            or :class:`~odl.Field`.
        scalar : ``op.range.field`` element
            A real or complex number, depending on the field of
            the range.
        """
        if not isinstance(op.range, (LinearSpace, Field)):
            raise TypeError('range {!r} not a `LinearSpace` or `Field` '
                            'instance.'.format(op.range))

        if scalar not in op.range.field:
            raise TypeError('scalar {!r} not in the field {!r} of the '
                            'operator range {!r}.'
                            ''.format(scalar, op.range.field, op.range))

        super().__init__(op.domain, op.range, linear=op.is_linear)
        self._op = op
        self._scalar = scalar

    def _call(self, x):
        """``op.__call__(x) <==> op(x)``."""
        # pylint: disable=protected-access
        return self._scalar * self._op._call(x)

    def _apply(self, x, out):
        """``op._apply(x, out) <==> out <-- op(x)``."""
        # pylint: disable=protected-access
        self._op._apply(x, out)
        out *= self._scalar

    @property
    def inverse(self):
        """The inverse operator.

        The inverse of ``scalar * op`` is given by
        ``op.inverse * 1/scalar`` if ``scalar != 0``. If ``scalar == 0``,
        the inverse is not defined.

        ``OperatorLeftScalarMult(op, scalar).inverse <==>``
        ``OperatorRightScalarMult(op.inverse, 1.0/scalar)``
        """
        if self.scalar == 0.0:
            raise ZeroDivisionError('{} not invertible.'.format(self))
        return OperatorLeftScalarMult(self._op.inverse, 1.0 / self._scalar)

    def derivative(self, x):
        """Return the derivative at ``x``.

        Left scalar multiplication and derivative are commutative:

        ``OperatorLeftScalarMult(op, scalar).derivative(x) <==>``
        ``OperatorLeftScalarMult(op.derivative(x), scalar)``

        See also
        --------
        OperatorLeftScalarMult : the result
        """
        return OperatorLeftScalarMult(self._op.derivative(x), self._scalar)

    @property
    def adjoint(self):
        """The operator adjoint.

        The adjoint of the operator scalar multiplication is the
        scalar multiplication of the operator adjoint:

        ``OperatorLeftScalarMult(op, scalar).adjoint ==``
        ``OperatorLeftScalarMult(op.adjoint, scalar)``
        """

        if not self.is_linear:
            raise NotImplementedError('Nonlinear operators have no adjoint')

        return OperatorRightScalarMult(self._op.adjoint,
                                       self._scalar.conjugate())

    def __repr__(self):
        """``op.__repr__() <==> repr(op)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self._op, self._scalar)

    def __str__(self):
        """``op.__str__() <==> str(op)``."""
        return '{} * {}'.format(self._scalar, self._op)


class OperatorRightScalarMult(Operator):

    """Expression type for the operator right scalar multiplication.

    ``OperatorRightScalarMult(op, scalar) <==> (x --> op(scalar * x))``

    The scalar multiplication is well-defined only if ``op.domain`` is
    a :class:`~odl.LinearSpace`.
    """

    def __init__(self, op, scalar, tmp=None):
        """Initialize a new :class:`OperatorLeftScalarMult` instance.

        Parameters
        ----------
        op : :class:`Operator`
            The domain of ``op`` must be a :class:`~odl.LinearSpace` or
            :class:`~odl.Field`.
        scalar : ``op.range.field`` element
            A real or complex number, depending on the field of
            the operator domain.
        tmp : domain element, optional
            Used to avoid the creation of a temporary when applying the
            operator.
        """
        if not isinstance(op.domain, (LinearSpace, Field)):
            raise TypeError('domain {!r} not a `LinearSpace` or `Field` '
                            'instance.'.format(op.domain))

        if scalar not in op.domain.field:
            raise TypeError('scalar {!r} not in the field {!r} of the '
                            'operator domain {!r}.'
                            ''.format(scalar, op.domain.field, op.domain))

        if tmp is not None and tmp not in op.domain:
            raise TypeError('temporary {!r} not an element of the '
                            'operator domain {!r}.'.format(tmp, op.domain))

        super().__init__(op.domain, op.range, op.is_linear)
        self._op = op
        self._scalar = scalar
        self._tmp = tmp

    def _call(self, x):
        """``op.__call__(x) <==> op(x)``."""
        # pylint: disable=protected-access
        return self._op._call(self._scalar * x)

    def _apply(self, x, out):
        """``op._apply(x, out) <==> out <-- op(x)``."""
        # pylint: disable=protected-access
        tmp = self._tmp if self._tmp is not None else self.domain.element()
        tmp.lincomb(self._scalar, x)
        self._op._apply(tmp, out)

    @property
    def inverse(self):
        """The inverse operator.

        The inverse of ``op * scalar`` is given by
        ``1/scalar * op.inverse`` if ``scalar != 0``. If ``scalar == 0``,
        the inverse is not defined.

        ``OperatorRightScalarMult(op, scalar).inverse <==>``
        ``OperatorLeftScalarMult(op.inverse, 1.0/scalar)``
        """
        if self.scalar == 0.0:
            raise ZeroDivisionError('{} not invertible.'.format(self))

        return OperatorLeftScalarMult(self._op.inverse, 1.0 / self._scalar)

    def derivative(self, x):
        """Return the derivative at ``x``.

        The derivative of the right scalar operator multiplication
        follows the chain rule:

        ``OperatorRightScalarMult(op, scalar).derivative(x) <==>``
        ``OperatorLeftScalarMult(op.derivative(scalar * x), scalar)``
        """
        return OperatorLeftScalarMult(self._op.derivative(self._scalar * x),
                                      self._scalar)

    @property
    def adjoint(self):
        """The operator adjoint.

        The adjoint of the operator scalar multiplication is the
        scalar multiplication of the operator adjoint:

        ``OperatorLeftScalarMult(op, scalar).adjoint ==``
        ``OperatorLeftScalarMult(op.adjoint, scalar)``
        """

        if not self.is_linear:
            raise NotImplementedError('Nonlinear operators have no adjoint')

        return OperatorRightScalarMult(self._op.adjoint,
                                       self._scalar.conjugate())

    def __repr__(self):
        """``op.__repr__() <==> repr(op)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self._op, self._scalar)

    def __str__(self):
        """``op.__str__() <==> str(op)``."""
        return '{} * {}'.format(self._op, self._scalar)


class FunctionalLeftVectorMult(Operator):

    """Expression type for the functional left vector multiplication.

    A functional is a :class:`Operator` whose :attr:`Operator.range` is
    a :class:`~odl.Field`.

    ``FunctionalLeftVectorMult(op, vector)(x) <==> vector * op(x)``

    """

    def __init__(self, op, vector):
        """Initialize a new :class:`FunctionalLeftVectorMult` instance.

        Parameters
        ----------
        op : :class:`Operator`
            The range of ``op`` must be a :class:`~odl.Field`.
        vector : :class:`~odl.LinearSpace.Vector`
            The vector to multiply by. its space's
            :attr:`~odl.LinearSpace.field` must be the same as
            ``op.range``
        """
        if not isinstance(vector, LinearSpace.Vector):
            raise TypeError('Vector {!r} not is not a LinearSpace.Vector'
                            ''.format(vector))

        if op.range != vector.space.field:
            raise TypeError('range {!r} not is not vector.space.field {!r}'
                            ''.format(op.range, vector.space.field))

        super().__init__(op.domain, vector.space, linear=op.is_linear)
        self._op = op
        self._vector = vector

    def _call(self, x):
        """``op.__call__(x) <==> op(x)``."""
        # pylint: disable=protected-access
        return self._vector * self._op._call(x)

    def _apply(self, x, out):
        """``op._apply(x, out) <==> out <-- op(x)``."""
        # pylint: disable=protected-access
        scalar = self._op._call(x)
        out.lincomb(scalar, self._vector)

    def derivative(self, x):
        """Return the derivative at ``x``.

        Left scalar multiplication and derivative are commutative:

        ``FunctionalLeftVectorMult(op, vector).derivative(x) <==>``
        ``FunctionalLeftVectorMult(op.derivative(x), vector)``

        See also
        --------
        FunctionalLeftVectorMult : the result
        """
        return FunctionalLeftVectorMult(self._op.derivative(x), self._vector)

    @property
    def adjoint(self):
        """The operator adjoint.

        The adjoint of the operator scalar multiplication is the
        scalar multiplication of the operator adjoint:

        ``FunctionalLeftVectorMult(op, vector).adjoint ==
        OperatorComp(op.adjoint, vector.T)``

        ``(x * A)^T = A^T * x^T``

        """

        if not self.is_linear:
            raise NotImplementedError('Nonlinear operators have no adjoint')

        return OperatorComp(self._op.adjoint, self._vector.T)

    def __repr__(self):
        """``op.__repr__() <==> repr(op)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self._op, self._vector)

    def __str__(self):
        """``op.__str__() <==> str(op)``."""
        return '{} * {}'.format(self._vector, self._op)


class OperatorLeftVectorMult(Operator):
    """Expression type for the operator left vector multiplication.

    ``OperatorLeftVectorMult(op, vector)(x) <==> vector * op(x)``

    The scalar multiplication is well-defined only if ``op.range`` is
    a ``vector.space.field``.
    """

    def __init__(self, op, vector):
        """Initialize a new :class:`OperatorLeftVectorMult` instance.

        Parameters
        ----------
        op : :class:`Operator`
            The range of ``op`` must be a :class:`~odl.LinearSpace`.
        vector : :class:`~odl.LinearSpace.Vector` in ``op.range``
            The vector to multiply by
        """
        if vector not in op.range:
            raise TypeError('vector {!r} not in op.range {!r}'
                            ''.format(vector, op.range))

        super().__init__(op.domain, op.range, linear=op.is_linear)
        self._op = op
        self._vector = vector

    def _call(self, x):
        """``op.__call__(x) <==> op(x)``."""
        # pylint: disable=protected-access
        return self._vector * self._op._call(x)

    def _apply(self, x, out):
        """``op._apply(x, out) <==> out <-- op(x)``."""
        # pylint: disable=protected-access
        self._op._apply(x, out)
        out *= self._vector

    def derivative(self, x):
        """Return the derivative at ``x``.

        Left scalar multiplication and derivative are commutative:

        ``OperatorLeftVectorMult(op, vector).derivative(x) <==>``
        ``OperatorLeftVectorMult(op.derivative(x), vector)``

        See also
        --------
        OperatorLeftVectorMult : the result
        """
        return OperatorLeftVectorMult(self._op.derivative(x), self._vector)

    @property
    def adjoint(self):
        """The operator adjoint.

        The adjoint of the operator vector multiplication is the
        vector multiplication of the operator adjoint:

        ``OperatorLeftVectorMult(op, vector).adjoint ==``
        ``OperatorRightVectorMult(op.adjoint, vector)``

        ``(x * A)^T = A^T * x``

        """

        if not self.is_linear:
            raise NotImplementedError('Nonlinear operators have no adjoint')

        return OperatorRightVectorMult(self._op.adjoint, self._vector)

    def __repr__(self):
        """``op.__repr__() <==> repr(op)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self._op, self._vector)

    def __str__(self):
        """``op.__str__() <==> str(op)``."""
        return '{} * {}'.format(self._vector, self._op)


class OperatorRightVectorMult(Operator):

    """Expression type for the operator right vector multiplication.

    ``OperatorRightVectorMult(op, vector)(x) <==> op(vector * x)``

    The scalar multiplication is well-defined only if
    ``vector in op.domain == True``.
    """

    def __init__(self, op, vector):
        """Initialize a new :class:`OperatorRightVectorMult` instance.

        Parameters
        ----------
        op : :class:`Operator`
            The domain of ``op`` must be a ``vector.space``.
        vector : :class:`~odl.LinearSpace.Vector` in ``op.domain``
            The vector to multiply by
        """
        if vector not in op.domain:
            raise TypeError('vector {!r} not in op.domain {!r}'
                            ''.format(vector.space, op.domain))

        super().__init__(op.domain, op.range, linear=op.is_linear)
        self._op = op
        self._vector = vector

    def _call(self, x):
        """``op.__call__(x) <==> op(x)``."""
        # pylint: disable=protected-access
        return self._op._call(self._vector * x)

    def _apply(self, x, out):
        """``op._apply(x, out) <==> out <-- op(x)``."""
        # pylint: disable=protected-access
        tmp = self.domain.element()
        tmp.multiply(self._vector, x)
        self._op._apply(tmp, out)

    def derivative(self, x):
        """Return the derivative at ``x``.

        Left vector multiplication and derivative are commutative:

        ``OperatorRightVectorMult(op, vector).derivative(x) <==>
        OperatorRightVectorMult(op.derivative(x), vector)``

        See also
        --------
        OperatorRightVectorMult : the result
        """
        return OperatorRightVectorMult(self._op.derivative(x), self._vector)

    @property
    def adjoint(self):
        """The operator adjoint.

        The adjoint of the operator vector multiplication is the
        vector multiplication of the operator adjoint:

        ``OperatorRightVectorMult(op, vector).adjoint ==``
        ``OperatorLeftVectorMult(op.adjoint, vector)``

        ``(A x)^T = x * A^T``

        """

        if not self.is_linear:
            raise NotImplementedError('Nonlinear operators have no adjoint')

        # TODO: handle complex vectors
        return OperatorLeftVectorMult(self._op.adjoint, self._vector)

    def __repr__(self):
        """``op.__repr__() <==> repr(op)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self._op, self._vector)

    def __str__(self):
        """``op.__str__() <==> str(op)``."""
        return '{} * {}'.format(self._op, self._vector)


def simple_operator(call=None, inv=None, deriv=None, dom=None, ran=None,
                    linear=False):
    """Create a simple operator.

    Mostly intended for simple prototyping rather than final use.

    Parameters
    ----------
    call : `callable`
        A function taking one argument and returning the result.
        It will be used for the operator call pattern
        ``out = op(x)``.
        TODO update
    inv : :class:`Operator`, optional
        The operator inverse
    deriv : :class:`Operator`, optional
        The operator derivative, linear
    dom : :class:`~odl.Set`, optional
        The domain of the operator
        Default: `UniversalSpace` if linear, else `UniversalSet`
    ran : :class:`~odl.Set`, optional
        The range of the operator
        Default: `UniversalSpace` if linear, else `UniversalSet`
    linear : `bool`, optional
        `True` if the operator is linear
        Default: `False`

    Returns
    -------
    op : :class:`Operator`
        An operator with the provided attributes and methods.

    Notes
    -----
    It suffices to supply one of the functions ``call`` and ``apply``.
    If ``dom`` is a :class:`~odl.LinearSpace`, a default implementation of the
    respective other method is automatically provided; if not, a
    `NotImplementedError` is raised when the other method is called.

    Examples
    --------
    >>> A = operator(lambda x: 3*x)
    >>> A(5)
    15
    """
    if dom is None:
        dom = UniversalSpace() if linear else UniversalSet()

    if ran is None:
        ran = UniversalSpace() if linear else UniversalSet()

    call_has_out, call_out_optional = _dispatch_call_args(object, call)

    if not call_has_out:
        # Out-of-place _call
        class SimpleOperator(Operator):

            def _call(self, x):
                return call(x)

    elif call_out_optional:
        # Dual-use _call
        class SimpleOperator(Operator):

            def _call(self, x, out=None):
                return call(x, out=None)
    else:
        # In-place only _call
        class SimpleOperator(Operator):

            def _call(self, x, out):
                return call(x, out)

    return SimpleOperator(dom, ran, linear)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
