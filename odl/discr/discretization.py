# Copyright 2014-2016 The ODL development group
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

"""Base classes for discretization."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from odl.operator import Operator
from odl.space.base_ntuples import (NtuplesBase, NtuplesBaseVector,
                                    FnBase, FnBaseVector)
from odl.space import FunctionSet, FN_IMPLS, NTUPLES_IMPLS
from odl.set import RealNumbers, ComplexNumbers, LinearSpace
from odl.util.utility import (
    arraynd_repr, arraynd_str,
    is_real_floating_dtype, is_complex_floating_dtype, is_scalar_dtype)


__all__ = ('DiscretizedSet', 'DiscretizedSetElement',
           'DiscretizedSpace', 'DiscretizedSpaceElement')


class DiscretizedSet(NtuplesBase):

    """Abstract discretization class for general sets.

    A discretization in ODL is a way to encode the transition from
    an arbitrary set to a set of n-tuples explicitly representable
    in a computer. The most common use case is the discretization of
    an infinite-dimensional vector space of functions by means of
    storing coefficients in a finite basis.

    The minimal information required to create a discretization is
    the set to be discretized ("undiscretized space") and a backend
    for storage and processing of the n-tuples ("data space" or
    "discretized space").

    As additional information, two mappings can be provided.
    The first one is an explicit way to map an (abstract) element from
    the source set to an ``n``-tuple. This mapping is called
    **sampling** in ODL.
    The second one encodes the converse way of mapping an ``n``-tuple to
    an element of the original set. This mapping is called
    **interpolation**.
    """

    def __init__(self, uspace, dspace, sampling=None, interpol=None):
        """Abstract initialization method.

        Intended to be called by subclasses for proper type checking
        and setting of attributes.

        Parameters
        ----------
        uspace : `FunctionSet`
            The undiscretized (abstract) set to be discretized
        dspace : `NtuplesBase`
            Data space providing containers for the values of a
            discretized object
        sampling : `Operator`, optional
            Operator mapping a `uspace` element to a `dspace` element.
            Must satisfy ``sampling.domain == uspace``,
            ``sampling.range == dspace``.
        interpol : `Operator`, optional
            Operator mapping a `dspace` element to a `uspace` element.
            Must satisfy ``interpol.domain == dspace``,
            ``interpol.range == uspace``.
            """
        if not isinstance(uspace, FunctionSet):
            raise TypeError('`uspace` {!r} not a `Set` instance'
                            ''.format(uspace))
        if not isinstance(dspace, NtuplesBase):
            raise TypeError('`dspace` {!r} not an `NtuplesBase` instance'
                            ''.format(dspace))

        if sampling is not None:
            if not isinstance(sampling, Operator):
                raise TypeError('`sampling` {!r} not an `Operator` '
                                'instance'.format(sampling))

            if sampling.domain != uspace:
                raise ValueError('`sampling.domain` {} not equal to '
                                 'the undiscretized space {}'
                                 ''.format(sampling.domain, dspace))

            if sampling.range != dspace:
                raise ValueError('`sampling.range` {} not equal to'
                                 'the data space {}'
                                 ''.format(sampling.range, dspace))

        if interpol is not None:
            if not isinstance(interpol, Operator):
                raise TypeError('`interpol` {!r} not an Operator '
                                'instance'.format(interpol))

            if interpol.domain != dspace:
                raise ValueError('`interpol.domain` {} not equal '
                                 'to the data space {}'
                                 ''.format(interpol.domain, dspace))

            if interpol.range != uspace:
                raise ValueError('`interpol.range` {} not equal to'
                                 'the undiscretized space {}'
                                 ''.format(interpol.range, uspace))

        super().__init__(dspace.size, dspace.dtype)
        self.__uspace = uspace
        self.__dspace = dspace
        self.__sampling = sampling
        self.__interpolation = interpol

    @property
    def uspace(self):
        """Undiscretized/continuous space of this discretization.

        Returns
        -------
        uspace : `FunctionSet`
        """
        return self.__uspace

    @property
    def dspace(self):
        """Space for the coefficients of the elements of this space.

        Returns
        -------
        dspace : `NtuplesBase`
        """
        return self.__dspace

    @property
    def dspace_type(self):
        """Data space type of this discretization."""
        return type(self.dspace)

    @property
    def sampling(self):
        """Operator mapping a `uspace` element to an n-tuple."""
        if self.__sampling is not None:
            return self.__sampling
        else:
            raise NotImplementedError('no sampling operator provided')

    @property
    def interpolation(self):
        """Operator mapping an n-tuple to a `uspace` element."""
        if self.__interpolation is not None:
            return self.__interpolation
        else:
            raise NotImplementedError('no interpolation operator provided')

    def element(self, inp=None, **kwargs):
        """Create an element from ``inp`` or from scratch.

        Parameters
        ----------
        inp : optional
            Input data to create an element from. It needs to be
            understood by either the `sampling` operator of this
            instance or by its ``dspace.element`` method.
        kwargs :
            Additional arguments passed on to `sampling` when called
            on ``inp``, in the form ``sampling(inp, **kwargs)``.
            This can be used e.g. for functions with parameters.

        Returns
        -------
        element : `DiscretizedSetElement`
            The discretized element, calculated as ``sampling(inp)`` or
            ``dspace.element(inp)``, tried in this order.

        See Also
        --------
        sampling : create a discrete element from an undiscretized one
        """
        if inp is None:
            return self.element_type(self, self.dspace.element())
        elif inp in self:
            return inp
        elif callable(inp):
            return self.element_type(self, self.sampling(inp, **kwargs))
        else:
            return self.element_type(self, self.dspace.element(inp))

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is a `DiscretizedSet`
            instance and all attributes `uspace`, `dspace`,
            `DiscretizedSet.sampling` and `DiscretizedSet.interpolation`
            of ``other`` and this discretization are equal, ``False``
            otherwise.
        """
        # Optimizations for simple cases
        if other is self:
            return True
        elif other is None:
            return False
        else:
            return (NtuplesBase.__eq__(self, other) and
                    other.uspace == self.uspace and
                    other.dspace == self.dspace and
                    other.sampling == self.sampling and
                    other.interpolation == self.interpolation)

    @property
    def impl(self):
        """Underlying implmentation type for the dspace."""
        return self.dspace.impl

    @property
    def domain(self):
        """Domain of the continuous space."""
        return self.uspace.domain

    @property
    def element_type(self):
        """`DiscretizedSetElement`"""
        return DiscretizedSetElement


class DiscretizedSetElement(NtuplesBaseVector):

    """Representation of a `DiscretizedSet` element.

    Basically only a wrapper class for dspace's element class."""

    def __init__(self, space, ntuple):
        """Initialize a new instance."""
        assert isinstance(space, DiscretizedSet)
        assert ntuple in space.dspace

        NtuplesBaseVector.__init__(self, space)
        self.__ntuple = ntuple

    @property
    def ntuple(self):
        """Structure for data storage."""
        return self.__ntuple

    @property
    def dtype(self):
        """Type of data storage."""
        return self.ntuple.dtype

    @property
    def size(self):
        """Size of data storage."""
        return self.ntuple.size

    def __len__(self):
        """Return ``len(self)``.

        Size of data storage.
        """
        return self.size

    def copy(self):
        """Create an identical (deep) copy of this element."""
        return self.space.element(self.ntuple.copy())

    def asarray(self, out=None):
        """Extract the data of this array as a numpy array.

        Parameters
        ----------
        out : `numpy.ndarray`, optional
            Array in which the result should be written in-place.
            Has to be contiguous and of the correct dtype.
        """
        return self.ntuple.asarray(out=out)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if all entries of ``other`` are equal to this
            element's entries, ``False`` otherwise.
        """
        return (other in self.space and
                self.ntuple == other.ntuple)

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        Parameters
        ----------
        indices : int or `slice`
            The position(s) that should be accessed

        Returns
        -------
        values : `NtuplesBaseVector`
            The value(s) at the index (indices)
        """
        return self.ntuple.__getitem__(indices)

    def __setitem__(self, indices, values):
        """Implement ``self[indices] = values``.

        Parameters
        ----------
        indices : int or `slice`
            The position(s) that should be set
        values : scalar, `array-like` or `NtuplesBaseVector`
            The value(s) that are to be assigned.

            If ``index`` is an int, ``value`` must be single value.

            If ``index`` is a slice, ``value`` must be broadcastable
            to the size of the slice (same size, shape (1,)
            or single value).
        """
        input_data = getattr(values, 'ntuple', values)
        self.ntuple.__setitem__(indices, input_data)

    def sampling(self, ufunc, **kwargs):
        """Restrict a continuous function and assign to this element.

        Parameters
        ----------
        ufunc : ``self.space.uspace`` element
            The continuous function that should be samplingicted.
        kwargs :
            Additional arugments for the sampling operator implementation

        Examples
        --------
        >>> X = odl.uniform_discr(0, 1, 5)
        >>> x = X.element()

        Assign x according to a continuous function:

        >>> x.sampling(lambda x: x)
        >>> print(x)  # Print values at grid points (which are centered)
        [0.1, 0.3, 0.5, 0.7, 0.9]

        See Also
        --------
        DiscretizedSet.sampling : For full description
        """
        self.space.sampling(ufunc, out=self.ntuple, **kwargs)

    @property
    def interpolation(self):
        """Interpolation operator associated with this element.

        Returns
        -------
        interpolation_op : `FunctionSetMapping`
            Operatior representing a continuous interpolation of this
            element.

        Examples
        --------
        Create continuous version of a discrete 1d function with nearest
        neighbour interpolation:

        >>> X = odl.uniform_discr(0, 1, 3, nodes_on_bdry=True)
        >>> x = X.element([0, 1, 0])
        >>> x.interpolation(np.array([0.24, 0.26]))
        array([ 0.,  1.])

        Linear interpolation:

        >>> X = odl.uniform_discr(0, 1, 3, nodes_on_bdry=True, interp='linear')
        >>> x = X.element([0, 1, 0])
        >>> x.interpolation(np.array([0.24, 0.26]))
        array([ 0.48,  0.52])

        See Also
        --------
        DiscretizedSet.interpolation : For full description
        """
        return self.space.interpolation(self.ntuple)

    def __str__(self):
        """Return ``str(self)``."""
        return arraynd_str(self.asarray())

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{!r}.element({})'.format(self.space,
                                         arraynd_repr(self.asarray()))


class DiscretizedSpace(DiscretizedSet, FnBase):

    """Abstract class for discretizations of linear vector spaces.

    This variant of `DiscretizedSet` adds linear structure
    to all its members. The `DiscretizedSet.uspace` is a
    `LinearSpace`, the `DiscretizedSet.dspace`
    for the data representation is an implementation of
    :math:`\mathbb{F}^n`, where :math:`\mathbb{F}` is some
    `Field`, and both `DiscretizedSet.sampling`
    and `DiscretizedSet.interpolation` are linear
    `Operator`'s.
    """

    def __init__(self, uspace, dspace, sampling=None, interpol=None):
        """Abstract initialization method.

        Intended to be called by subclasses for proper type checking
        and setting of attributes.

        Parameters
        ----------
        uspace : `LinearSpace`
            The (abstract) space to be discretized
        dspace : `FnBase`
            Data space providing containers for the values of a
            discretized object. Its `FnBase.field` attribute
            must be the same as ``uspace.field``.
        sampling : `Operator`, linear, optional
            Operator mapping a `DiscretizedSet.uspace` element
            to a `DiscretizedSet.dspace` element. Must satisfy
            ``sampling.domain == uspace``, ``sampling.range == dspace``
        interpol : `Operator`, linear, optional
            Operator mapping a `DiscretizedSet.dspace` element
            to a `DiscretizedSet.uspace` element. Must satisfy
            ``interpol.domain == dspace``, ``interpol.range == uspace``.
        """
        DiscretizedSet.__init__(self, uspace, dspace, sampling, interpol)
        FnBase.__init__(self, dspace.size, dspace.dtype)

        if not isinstance(uspace, LinearSpace):
            raise TypeError('`uspace` {!r} not a LinearSpace '
                            'instance'.format(uspace))

        if not isinstance(dspace, FnBase):
            raise TypeError('`dspace` {!r} not an FnBase instance'
                            ''.format(dspace))

        if uspace.field != dspace.field:
            raise ValueError('fields {} and {} of the undiscretized and '
                             'data spaces, resp., are not equal'
                             ''.format(uspace.field, dspace.field))

        if sampling is not None and not sampling.is_linear:
            raise TypeError('`sampling` {!r} is not linear'
                            ''.format(sampling))

        if interpol is not None and not interpol.is_linear:
            raise TypeError('`interpol` {!r} is not linear'
                            ''.format(interpol))

    # Pass-through attributes of the wrapped ``dspace``
    def zero(self):
        """Return the element of all zeros."""
        return self.element_type(self, self.dspace.zero())

    def one(self):
        """Return the element of all ones."""
        return self.element_type(self, self.dspace.one())

    @property
    def weighting(self):
        """This space's weighting scheme."""
        return getattr(self.dspace, 'weighting', None)

    @property
    def is_weighted(self):
        """``True`` if the ``dspace`` is weighted."""
        return getattr(self.dspace, 'is_weighted', False)

    def _lincomb(self, a, x1, b, x2, out):
        """Raw linear combination."""
        self.dspace._lincomb(a, x1.ntuple, b, x2.ntuple, out.ntuple)

    def _dist(self, x1, x2):
        """Raw distance between two elements."""
        return self.dspace._dist(x1.ntuple, x2.ntuple)

    def _norm(self, x):
        """Raw norm of an element."""
        return self.dspace._norm(x.ntuple)

    def _inner(self, x1, x2):
        """Raw inner product of two elements."""
        return self.dspace._inner(x1.ntuple, x2.ntuple)

    def _multiply(self, x1, x2, out):
        """Raw pointwise multiplication of two elements."""
        self.dspace._multiply(x1.ntuple, x2.ntuple, out.ntuple)

    def _divide(self, x1, x2, out):
        """Raw pointwise multiplication of two elements."""
        self.dspace._divide(x1.ntuple, x2.ntuple, out.ntuple)

    @property
    def examples(self):
        """Return example functions in the space.

        These are created by discretizing the examples in the underlying
        `uspace`.

        See Also
        --------
        odl.space.fspace.FunctionSpace.examples
        """
        for name, elem in self.uspace.examples:
            yield (name, self.element(elem))

    @property
    def element_type(self):
        """`DiscretizedSpaceElement`"""
        return DiscretizedSpaceElement


class DiscretizedSpaceElement(DiscretizedSetElement, FnBaseVector):

    """Representation of a `DiscretizedSpace` element."""

    def __init__(self, space, data):
        """Initialize a new instance."""
        assert isinstance(space, DiscretizedSpace)
        DiscretizedSetElement.__init__(self, space, data)

    def __ipow__(self, p):
        """Implement ``self **= p``."""
        # Falls back to `LinearSpaceElement.__ipow__` if `self.ntuple`
        # has no own `__ipow__`. The fallback only works for integer `p`.
        self.ntuple.__ipow__(p)
        return self


def dspace_type(space, impl, dtype=None):
    """Select the correct corresponding n-tuples space.

    Parameters
    ----------
    space : `LinearSpace`
        Template space from which to infer an adequate data space. If
        it has a `LinearSpace.field` attribute, ``dtype`` must be
        consistent with it.
    impl : string
        Implementation backend for the data space
    dtype : `numpy.dtype`, optional
        Data type which the space is supposed to use. If ``None`` is
        given, the space type is purely determined from ``space`` and
        ``impl``. Otherwise, it must be compatible with the
        field of ``space``.

    Returns
    -------
    stype : type
        Space type selected after the space's field, the backend and
        the data type
    """
    spacetype_map = {RealNumbers: FN_IMPLS,
                     ComplexNumbers: FN_IMPLS,
                     type(None): NTUPLES_IMPLS}

    field_type = type(getattr(space, 'field', None))

    if dtype is None:
        pass
    elif is_real_floating_dtype(dtype):
        if field_type is None or field_type == ComplexNumbers:
            raise TypeError('real floating data type {!r} requires space '
                            'field to be of type RealNumbers, got {}'
                            ''.format(dtype, field_type))
    elif is_complex_floating_dtype(dtype):
        if field_type is None or field_type == RealNumbers:
            raise TypeError('complex floating data type {!r} requires space '
                            'field to be of type ComplexNumbers, got {!r}'
                            ''.format(dtype, field_type))
    elif is_scalar_dtype(dtype):
        if field_type == ComplexNumbers:
            raise TypeError('non-floating data type {!r} requires space field '
                            'to be of type RealNumbers, got {!r}'
                            .format(dtype, field_type))
    else:
        raise TypeError('non-scalar data type {!r} cannot be combined with '
                        'a `LinearSpace`'.format(dtype))

    stype = spacetype_map[field_type].get(impl, None)

    if stype is None:
        raise NotImplementedError('no corresponding data space available '
                                  'for space {!r} and implementation {!r}'
                                  ''.format(space, impl))
    return stype


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
