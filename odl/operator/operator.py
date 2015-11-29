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

"""Abstract mathematical operators."""

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
from odl.set.space import (LinearSpace, LinearSpaceVector,
                           UniversalSpace)
from odl.set.sets import Set, UniversalSet, Field


__all__ = ('Operator', 'OperatorComp', 'OperatorSum',
           'OperatorLeftScalarMult', 'OperatorRightScalarMult',
           'FunctionalLeftVectorMult',
           'OperatorLeftVectorMult', 'OperatorRightVectorMult',
           'OperatorPointwiseProduct', 'simple_operator')


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
    out.assign(op._call_out_of_place(x, **kwargs))


def _dispatch_call_args(cls=object, unbound_call=None, attr='_call'):
    """Check the arguments of ``_call()`` or similar for conformity.

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

    Parameters
    ----------
    cls : `class`, optional
        The ``_call()`` method of this class is checked. If omitted,
        provide ``unbound_call`` instead to check directly.
    unbound_call: `callable`, optional
        Check this unbound function instead of ``cls``
    attr : `string`, optional
        Check this attribute instead of ``_call``, e.g. ``__call__``

    Returns
    -------
    has_out : `bool`
        Whether the call has an ``out`` argument
    out_is_optional : `bool`
        Whether the ``out`` argument is optional
    spec : :class:`~inspect.ArgSpec` or :class:`~inspect.FullArgSpec`
        Argument specification of the checked call function

    Raises
    ------
    ValueError
        if the signature of the function is malformed
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
        for parent in cls.mro():
            call = parent.__dict__.get(attr, None)
            if call is not None:
                break

        # Static and class methods are not allowed
        if isinstance(call, staticmethod):
            raise TypeError("'{}.{}' is a static method. ".format(attr) +
                            spec_msg)
        elif isinstance(call, classmethod):
            raise TypeError("'{}.{}' is a class method. ".format(attr) +
                            spec_msg)
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

    pos_args = spec.args
    if unbound_call is not None:
        # Add 'self' to positional arg list to satisfy the checker
        pos_args.insert(0, 'self')

    pos_defaults = spec.defaults
    varargs = spec.varargs

    out_optional = False

    # Variable args are not allowed
    if varargs is not None:
        raise ValueError("Variable arguments not allowed in '{}()'."
                         "".format(attr) + spec_msg)

    if len(pos_args) not in (2, 3):
        raise ValueError("Bad signature of '{}()'. ".format(attr) + spec_msg)

    # 'self' must be the first argument
    elif pos_args[0] != 'self':
        raise ValueError("'self' is not the first argument in '{}()'."
                         "".format(attr) + spec_msg)

    true_pos_args = pos_args[1:]
    if len(true_pos_args) == 1:  # 'out' kw-only
        if 'out' in true_pos_args:  # 'out' positional and 'x' kw-only -> no
            raise ValueError("'out' cannot be only positional argument except "
                             "'self' in '{}()'.".format(attr) + spec_msg)
        else:
            if len(kw_only) == 0:
                has_out = False
            elif len(kw_only) == 1:
                if 'out' not in kw_only:
                    raise ValueError("Output parameter must be called 'out'"
                                     " in '{}()'.".format(attr) + spec_msg)
                else:
                    has_out = True
                    if kw_only_defaults['out'] is not None:
                        raise ValueError("'out' can only default to None in "
                                         "'{}()'.".format(attr) + spec_msg)
                    else:
                        out_optional = True
            else:
                raise ValueError("Bad signature of '{}()'.".format(attr) +
                                 spec_msg)

    elif len(true_pos_args) == 2:  # Both args positional
        if true_pos_args[0] == 'out':  # 'out' must come second
            py3_txt = 'or keyword-only ' if py3 else ''
            raise ValueError("'out' can only be the second positional "
                             "argument " + py3_txt + "in '{}()'."
                             "".format(attr) + spec_msg)
        elif true_pos_args[1] != 'out':  # 'out' must be 'out'
            raise ValueError("Output parameter must be called 'out'"
                             " in '{}()'.".format(attr) + spec_msg)
        else:
            has_out = True
            out_optional = bool(pos_defaults)
            if pos_defaults and pos_defaults[-1] is not None:
                raise ValueError("'out' can only default to None in "
                                 "'{}()'.".format(attr) + spec_msg)

    else:  # Too many positional args
        raise ValueError("Bad signature of '{}()'. ".format(attr) + spec_msg)

    return has_out, out_optional, spec


def _indent_length(lst):
    """Calculate indentation length from a string list."""
    if len(lst) == 0:
        return 0
    else:
        return min((len(line) - len(line.lstrip())
                   for line in lst if line))


def _decompose_doc(obj):
    """Create a structured dictionary from the docstring of ``obj``.

    The dictionary contains the following entries:

        ``'indent'`` : `int`
            indentation length

    If ``obj.__doc__`` is not `None`:

        ``0`` : `dict`
            ``{'underline': '', 'lines': [<summary line>]}``

    If ``obj.__doc__`` has more than one line:

        ``1`` : `dict`
            ``{'underline': '', 'lines': [<extended summary lines>]}``

    If ``obj.__doc__`` has sections like ``'Parameters'``:

        ``'<section name>'`` : `dict`
            ``{'underline': <underline string>,
            'lines': [<section lines>]}``
    """
    sect_dict = {}
    docstring = obj.__doc__
    if docstring is None:
        return sect_dict

    # Summary line
    doclines = docstring.splitlines()
    sect_dict[0] = {'underline': '', 'lines': doclines[0]}
    if len(doclines) == 1:
        return sect_dict

    # Docstring body
    sect_dict[1] = {'underline': '', 'lines': []}
    body_lines = doclines[1:]
    indent = _indent_length(body_lines)
    sect_dict['indent'] = indent
    # Dedent - too short lines shouldn't occur, but just in case...
    body_lines = [line[indent:] if len(line) >= indent else ''
                  for line in body_lines]

    # Chars which signalize underlining
    underl_chars = ['-', '~', '=']
    underl_starts = tuple(c * 4 for c in underl_chars)

    cur_secname = 1  # start with section 1
    for line, nextline in zip(body_lines[:-1], body_lines[1:]):
        if nextline.startswith(underl_starts):
            # New dict entry
            cur_secname = line.rstrip()
            sect_dict[cur_secname] = {'underline': '', 'lines': []}
        elif line.startswith(underl_starts):
            # Save as underline string
            sect_dict[cur_secname]['underline'] = line
        else:
            # Save as regular line
            sect_dict[cur_secname]['lines'].append(line.rstrip())

    return sect_dict


def _decompose_params(lst):
    """Create a structured dictionary from the 'Parameters' line list.

    This function looks for unindented lines of the form
    ``<arg> : <type>``. It then creates an entry

        ``'<arg>'`` :
            ``{'typename': <type name>, 'lines': [<arg section lines>]}``

    in the dictionary and adds the lines until the next such line to
    the list.
    """

    def is_arg_line(line):
        # Empty or indented line
        if not line or line.startswith(' '):
            return False
        argname, sep, _ = line.partition(':')

        # No colon found or colon comes first
        if not sep or not argname:
            return False

        # True if argname is a single word, False otherwise
        return argname.strip().strip("'*").isalpha()

    arg_dict = {}
    if not lst:
        return arg_dict

    # Intro text. Shouldn't be there, but anyway...
    # Possible outro text, which shouldn't be there either, will be added
    # to the last arg section
    arg_dict[0] = {'typename': '', 'lines': []}

    cur_key = 0
    for line in lst:
        if not is_arg_line(line):
            arg_dict[cur_key]['lines'].append(line)
        else:
            argname, _, typename = line.partition(':')
            argname = argname.strip().strip("'*")
            typename = typename.strip()
            arg_dict[argname] = {'typename': typename, 'lines': []}
            cur_key = argname

    return arg_dict


def _sync_call_args(cls, sdict, attr):
    """Synchronize argument names etc. between spec and doc.

    This function reads the argument names from the signature of
    ``cls._call`` (or similar) and adapts the name of the first
    positional argument (``x``) in the doc according to the spec.
    For the ``out`` argument, the string ``', optional'`` is added
    to or removed from the type specification if necessary.
    """
    has_out, out_is_optional, spec = _dispatch_call_args(cls, attr=attr)
    specarg_names = spec.args.remove('self')
    if spec.keywords is not None:
        specarg_names.append(spec.keywords)

    docarg_dict = _decompose_params(sdict['Parameters']['lines'])
    docarg_names = docarg_dict.keys().remove(0)

    wrong_args = [darg for darg in docarg_names if darg not in specarg_names]

    if len(wrong_args) > 1:
        raise ValueError('arguments {} not in function spec of {}. Cannot '
                         'resolve ambiguity.'.format(wrong_args,
                                                     getattr(cls, attr)))

    missing_args = [sarg for sarg in specarg_names if sarg not in docarg_names]

    # TODO: continue here


def _compile_call_dict(cls):

    # TODO: this is out of date

    # From class _call: get doc dict, spec, out and kwargs info
    sdict = _decompose_doc(cls._call)
    has_out, out_optional, spec = _dispatch_call_args(cls)

    # From Operator.__call__: get doc dict and out & kwargs index
    std_dict = _decompose_doc(Operator.__call__)
    std_out_idx, std_out_end_idx = out_info(std_dict)[:2]

    new_dict = {sec: {'underline': '', 'lines': []} for sec in sdict.keys()}
    new_dict['Parameters'] = {'underline': sdict['Parameters']['underline'],
                              'lines': []}

    if 'Parameters' in sdict:
        # Get lines dealing with parameter explanations
        param_lines = sdict['Parameters']['lines']
        std_param_lines = std_dict['Parameters']['lines']
        out_idx, out_end_idx, doc_has_out, doc_out_optional = out_info(sdict)
        doc_has_kwargs = has_kwargs(sdict)

        if len(param_lines) == 0 or out_idx == 0:
            # Need to add section on first argument, get its name
            x_name = spec.args[0]

            # Replace each citation of 'x' in the std doc with the new name
            for sec in std_dict:
                for i, line in enumerate(std_dict[sec]['lines']):
                    line = line.replace("'x'", "'{}'".format(x_name))
                    line = line.replace("``x``", "``{}``".format(x_name))
                    line = line.replace("(x)", "({})".format(x_name))
                    std_dict[sec]['lines'][i] = line

            # Add the section on 'x' with 'x' replaced
            x_defline = x_name + std_param_lines[0].lstrip("'`x'")
            new_dict['Parameters']['lines'].append(x_defline)
            for line in std_dict['Parameters']['lines'][:std_out_idx]:
                new_dict['Parameters']['lines'].append(line)

        elif has_out and not doc_has_out:
            # Need to add doc on 'out'
            out_defline = std_dict['Parameters']['lines'][std_out_idx]

            if out_optional:
                out_defline += ', optional'
            new_dict['Parameters']['lines'].append(out_defline)

            for line in std_dict['Parameters']['lines'][std_out_idx + 1:
                                                        std_out_end_idx]:
                new_dict['Parameters']['lines'].append(line)

        elif has_out and doc_has_out:
            # Can just add doc on 'out'
            for line in sdict['Parameters']['lines'][out_idx:out_end_idx]:
                new_dict['Parameters']['lines'].append(line)

        else:
            # Should not happen
            raise ValueError("'out section in doc but no 'out' parameter.'")

        has_kwargs = spec.keywords is not None
        if has_kwargs and not doc_has_kwargs:
            # Need to add doc on kwargs
            pass


class Operator(with_metaclass(ABCMeta, object)):

    """Abstract operator.

    **Abstract attributes and methods**

    `Operator` is an **abstract** class, i.e. it can only be
    subclassed, not used directly.

    Any subclass of `Operator` **must** have the following
    attributes:

    ``domain`` : `Set`
        The set of elements this operator can be applied to

    ``range`` : `Set`
        The set this operator maps to

    It is **highly** recommended to call
    ``super().__init__(dom, ran)`` (Note: add
    ``from builtins import super`` in Python 2) in the ``__init__()``
    method of any subclass, where ``dom`` and ``ran`` are the arguments
    specifying domain and range of the new
    operator. In that case, the attributes `Operator.domain` and
    `Operator.range` are automatically provided by
    `Operator`.

    In addition, any subclass **must** implement **at least one** of the
    methods ``_apply()`` and ``_call()``, which are explained in the
    following.

    **In-place evaluation:** ``_apply()``

    In-place evaluation means that the operator is applied, and the
    result is written to an existing element provided as an additional
    argument. In this case, a subclass has to implement the method

        ``_apply(self, x, out)  <==>  out <-- operator(x)``

    **Parameters:**

    x : `Operator.domain` element
        An object in the operator domain to which the operator is
        applied.

    out : `Operator.range` element
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

    x : `Operator.domain` element
        An object in the operator domain to which the operator is
        applied.

    out : `Operator.range` element
        An object in the operator range to which the result of the
        operator evaluation is written.

    Notes
    -----
    If not both ``_apply()`` and ``_call()`` are implemented and the
    `Operator.range` is a `LinearSpace`, a default
    implementation of the respective other is provided.
    """

    def __new__(cls, *args, **kwargs):
        """Create a new instance."""
        instance = super().__new__(cls)

        call_has_out, call_out_optional, _ = _dispatch_call_args(cls)
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
        domain : `Set`
            The domain of this operator, i.e., the set of elements to
            which this operator can be applied
        range : `Set`
            The range of this operator, i.e., the set this operator
            maps to
        linear : bool
            If `True`, the operator is considered as linear. In this
            case, `domain` and `range` have to be instances of
            `LinearSpace`, `RealNumbers` or `ComplexNumbers`.
        """
        if not isinstance(domain, Set):
            raise TypeError('domain {!r} is not a `Set` instance.'
                            ''.format(domain))
        if not isinstance(range, Set):
            raise TypeError('range {!r} is not a `Set` instance.'
                            ''.format(range))

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
        """`True` if the this operator's range is a `Field`."""
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
        x : `Operator.domain` element
            An object in the operator domain to which the operator is
            applied. The object is treated as immutable, hence it is
            not modified during evaluation.
        out : `Operator.range` element, optional
            An object in the operator range to which the result of the
            operator evaluation is written. The result is independent
            of the initial state of this object.
        **kwargs : Further arguments to the function, optional

        Returns
        -------
        elem : `Operator.range` element
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

            self._call_in_place(x, out=out, **kwargs)
            return out

        else:  # Out-of-place evaluation
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
        other : {`Operator`, `LinearSpaceVector`, scalar}
            `Operator`:
                The `Operator.domain` of ``other`` must match this
                operator's `Operator.range`.

            `LinearSpaceVector`:
                ``other`` must be an element of this operator's
                `Operator.domain`.

            scalar:
                The `Operator.domain` of this operator must be a
                `LinearSpace` and ``other`` must be an
                element of the ``field`` of this operator's
                `Operator.domain`.

        Returns
        -------
        mul : `Operator`
            The multiplication operator.

            If ``other`` is an operator, ``mul`` is an
            `OperatorComp`.

            If ``other`` is a scalar, ``mul`` is an
            `OperatorRightScalarMult`.

            If ``other`` is a vector, ``mul`` is an
            `OperatorRightVectorMult`.

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
        elif isinstance(other, LinearSpaceVector) and other in self.domain:
            return OperatorRightVectorMult(self, other.copy())
        else:
            return NotImplemented

    def __matmul__(self, other):
        """``op.__matmul__(other) <==> op @ other``.

        See `Operator.__mul__`
        """
        return self.__mul__(other)

    def __rmul__(self, other):
        """``op.__rmul__(s) <==> s * op``.

        If ``other`` is an `Operator`, this corresponds to
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
        other : {`Operator`, `LinearSpaceVector`, scalar}
            `Operator`:
                The `Operator.range` of ``other`` must match this
                operator's `Operator.domain`

            `LinearSpaceVector`:
                ``other`` must be an element of `Operator.range`.

            scalar:
                `Operator.range` must be a
                `LinearSpace` and ``other`` must be an
                element of ``self.range.field``.

        Returns
        -------
        mul : `Operator`
            The multiplication operator.

            If ``other`` is an operator, ``mul`` is an
            `OperatorComp`.

            If ``other`` is a scalar, ``mul`` is an
            `OperatorLeftScalarMult`.

            If ``other`` is a vector, ``mul`` is an
            `OperatorLeftVectorMult`.

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
        elif (isinstance(other, LinearSpaceVector) and
              other.space.field == self.range):
            return FunctionalLeftVectorMult(self, other.copy())
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        """``op.__rmatmul__(other) <==> other @ op``.

        See `Operator.__rmul__`
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
        pow : `Operator`
            The power of this operator. If ``n == 1``, ``pow`` is
            this operator, for ``n > 1``, a `OperatorComp`

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
            If `Operator.range` is a `LinearSpace`,
            ``scalar`` must be an element of this operator's
            ``field``.

        Returns
        -------
        rmul : `OperatorRightScalarMult`
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
    # Set higher than LinearSpaceVector.__array_priority__ to handle mult with
    # vector properly
    __array_priority__ = 2000000.0


class OperatorSum(Operator):

    """Expression type for the sum of operators.

    ``OperatorSum(op1, op2) <==> (x --> op1(x) + op2(x))``

    The sum is only well-defined for `Operator` instances where
    `Operator.range` is a `LinearSpace`.

    """

    # pylint: disable=abstract-method
    def __init__(self, op1, op2, tmp_ran=None, tmp_dom=None):
        """Initialize a new instance.

        Parameters
        ----------
        op1 : `Operator`
            The first summand. Its `Operator.range` must be a
            `LinearSpace` or `Field`.
        op2 : `Operator`
            The second summand. Must have the same
            `Operator.domain` and `Operator.range` as
            ``op1``.
        tmp_ran : `Operator.range` element, optional
            Used to avoid the creation of a temporary when applying the
            operator.
        tmp_dom : `Operator.domain` element, optional
            Used to avoid the creation of a temporary when applying the
            operator adjoint.
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

    def _call(self, x, out):
        """``op._call(x, out) <==> out <-- op(x)``.

        Examples
        --------
        >>> from odl import Rn, IdentityOperator
        >>> r3 = Rn(3)
        >>> op = IdentityOperator(r3)
        >>> x = r3.element([1, 2, 3])
        >>> out = r3.element()
        >>> OperatorSum(op, op)(x, out)  # In place, returns out
        Rn(3).element([2.0, 4.0, 6.0])
        >>> out
        Rn(3).element([2.0, 4.0, 6.0])
        >>> OperatorSum(op, op)(x)  # In place, returns out
        Rn(3).element([2.0, 4.0, 6.0])
        """
        tmp = (self._tmp_ran if self._tmp_ran is not None
               else self.range.element())
        self._op1(x, out=out)
        self._op2(x, out=tmp)
        out += tmp

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
        """Initialize a new `OperatorComp` instance.

        Parameters
        ----------
        left : `Operator`
            The left ("outer") operator
        right : `Operator`
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

    def _call(self, x, out):
        """``op._call(x, out) <==> out <-- op(x)``."""
        tmp = (self._tmp if self._tmp is not None
               else self._right.range.element())
        self._right(x, out=tmp)
        self._left(tmp, out=out)

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
        op1 : `Operator`
            The first factor
        op2 : `Operator`
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

    def _call(self, x, out):
        """``op._call(x, out) <==> out <-- op(x)``."""
        tmp = self._op2.range.element()
        self._op1(x, out=out)
        self._op2(x, out=tmp)
        out *= tmp

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
    a `LinearSpace`.
    """

    def __init__(self, op, scalar):
        """Initialize a new `OperatorLeftScalarMult` instance.

        Parameters
        ----------
        op : `Operator`
            The range of ``op`` must be a `LinearSpace`
            or `Field`.
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

    def _call(self, x, out):
        """``op._call(x, out) <==> out <-- op(x)``."""
        self._op(x, out=out)
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
    a `LinearSpace`.
    """

    def __init__(self, op, scalar, tmp=None):
        """Initialize a new `OperatorLeftScalarMult` instance.

        Parameters
        ----------
        op : `Operator`
            The domain of ``op`` must be a `LinearSpace` or
            `Field`.
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

    def _call(self, x, out):
        """``op._call(x, out) <==> out <-- op(x)``."""
        tmp = self._tmp if self._tmp is not None else self.domain.element()
        tmp.lincomb(self._scalar, x)
        self._op(tmp, out=out)

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

    A functional is a `Operator` whose `Operator.range` is
    a `Field`.

    ``FunctionalLeftVectorMult(op, vector)(x) <==> vector * op(x)``
    """

    def __init__(self, op, vector):
        """Initialize a new instance.

        Parameters
        ----------
        op : `Operator`
            The range of ``op`` must be a `Field`.
        vector : `LinearSpaceVector`
            The vector to multiply by. its space's
            `LinearSpace.field` must be the same as
            ``op.range``
        """
        if not isinstance(vector, LinearSpaceVector):
            raise TypeError('Vector {!r} not is not a LinearSpaceVector'
                            ''.format(vector))

        if op.range != vector.space.field:
            raise TypeError('range {!r} not is not vector.space.field {!r}'
                            ''.format(op.range, vector.space.field))

        super().__init__(op.domain, vector.space, linear=op.is_linear)
        self._op = op
        self._vector = vector

    def _call(self, x, out):
        """``op._call(x, out) <==> out <-- op(x)``."""
        scalar = self._op(x)
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
        """Initialize a new `OperatorLeftVectorMult` instance.

        Parameters
        ----------
        op : `Operator`
            The range of ``op`` must be a `LinearSpace`.
        vector : `LinearSpaceVector` in ``op.range``
            The vector to multiply by
        """
        if vector not in op.range:
            raise TypeError('vector {!r} not in op.range {!r}'
                            ''.format(vector, op.range))

        super().__init__(op.domain, op.range, linear=op.is_linear)
        self._op = op
        self._vector = vector

    def _call(self, x, out):
        """``op._call(x, out) <==> out <-- op(x)``."""
        self._op(x, out=out)
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
        """Initialize a new `OperatorRightVectorMult` instance.

        Parameters
        ----------
        op : `Operator`
            The domain of ``op`` must be a ``vector.space``.
        vector : `LinearSpaceVector` in ``op.domain``
            The vector to multiply by
        """
        if vector not in op.domain:
            raise TypeError('vector {!r} not in op.domain {!r}'
                            ''.format(vector.space, op.domain))

        super().__init__(op.domain, op.range, linear=op.is_linear)
        self._op = op
        self._vector = vector

    def _call(self, x, out):
        """``op._call(x, out) <==> out <-- op(x)``."""
        tmp = self.domain.element()
        tmp.multiply(self._vector, x)
        self._op(tmp, out=out)

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
    deriv : `Operator`, optional
        The operator derivative, linear
    dom : `Set`, optional
        The domain of the operator
        Default: `UniversalSpace` if linear, else `UniversalSet`
    ran : `Set`, optional
        The range of the operator
        Default: `UniversalSpace` if linear, else `UniversalSet`
    linear : `bool`, optional
        `True` if the operator is linear
        Default: `False`

    Returns
    -------
    op : `Operator`
        An operator with the provided attributes and methods.

    Notes
    -----
    It suffices to supply one of the functions ``call`` and ``apply``.
    If ``dom`` is a `LinearSpace`, a default implementation of the
    respective other method is automatically provided; if not, a
    `NotImplementedError` is raised when the other method is called.

    Examples
    --------
    >>> A = simple_operator(lambda x: 3*x)
    >>> A(5)
    15
    """
    if dom is None:
        dom = UniversalSpace() if linear else UniversalSet()

    if ran is None:
        ran = UniversalSpace() if linear else UniversalSet()

    call_has_out, call_out_optional, _ = _dispatch_call_args(object, call)

    attrs = {'inverse': inv, 'derivative': deriv}

    if not call_has_out:
        # Out-of-place _call

        def _call(self, x):
            return call(x)

        attrs['_call_in_place'] = _default_call_in_place
        attrs['_call_out_of_place'] = _call
    elif call_out_optional:
        # Dual-use _call

        def _call(self, x, out=None):
            return call(x, out=out)

        attrs['_call_in_place'] = _call
        attrs['_call_out_of_place'] = _call
    else:
        # In-place only _call

        def _call(self, x, out):
            return call(x, out)

        attrs['_call_in_place'] = _call
        attrs['_call_out_of_place'] = _default_call_out_of_place

    attrs['_call'] = _call

    SimpleOperator = ABCMeta('SimpleOperator', (Operator,), attrs)
    return SimpleOperator(dom, ran, linear)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
