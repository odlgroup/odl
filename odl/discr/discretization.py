# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Base classes for discretization."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from odl.operator import Operator
from odl.set.sets import Set
from odl.space.base_tensors import TensorSpace, Tensor
from odl.space.entry_points import TENSOR_SPACE_IMPLS
from odl.set import RealNumbers, ComplexNumbers
from odl.util import (
    arraynd_repr, arraynd_str, indent_rows,
    is_real_floating_dtype, is_complex_floating_dtype, is_numeric_dtype)


__all__ = ('DiscretizedSpace',)


class DiscretizedSpace(TensorSpace):

    """Abstract discretization class for general sets or spaces.

    A discretization in ODL is a way to encode the transition from
    an arbitrary set to a set of discrete values explicitly representable
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
        uspace : `Set`
            The undiscretized (abstract) set to be discretized.
        dspace : `TensorSpace`
            Data space providing containers for the values of a
            discretized object.
        sampling : `Operator`, optional
            Operator mapping a `uspace` element to a `dspace` element.
            Must satisfy ``sampling.domain == uspace``,
            ``sampling.range == dspace``.
        interpol : `Operator`, optional
            Operator mapping a `dspace` element to a `uspace` element.
            Must satisfy ``interpol.domain == dspace``,
            ``interpol.range == uspace``.
            """
        if not isinstance(uspace, Set):
            raise TypeError('`uspace` must be a `Set` instance, '
                            'got {!r}'.format(uspace))
        if not isinstance(dspace, TensorSpace):
            raise TypeError('`dspace` {!r} not a `TensorSpace` instance'
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

        super().__init__(dspace.shape, dspace.dtype, dspace.order)
        self.__uspace = uspace
        self.__dspace = dspace
        self.__sampling = sampling
        self.__interpolation = interpol

    @property
    def uspace(self):
        """Undiscretized/continuous space of this discretization."""
        return self.__uspace

    @property
    def dspace(self):
        """Space for the coefficients of the elements of this space."""
        return self.__dspace

    @property
    def dspace_type(self):
        """Data space type of this discretization."""
        return type(self.dspace)

    @property
    def sampling(self):
        """Operator mapping a `uspace` element to a tensor."""
        if self.__sampling is not None:
            return self.__sampling
        else:
            raise NotImplementedError('no sampling operator provided')

    @property
    def interpolation(self):
        """Operator mapping a tensor to a `uspace` element."""
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
        element : `DiscretizedSpaceElement`
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
            ``True`` if ``other`` is a `DiscretizedSpace`
            instance and all attributes `uspace`, `dspace`,
            `DiscretizedSpace.sampling` and `DiscretizedSpace.interpolation`
            of ``other`` and this discretization are equal, ``False``
            otherwise.
        """
        # Optimizations for simple cases
        if other is self:
            return True
        elif other is None:
            return False
        else:
            return (TensorSpace.__eq__(self, other) and
                    other.uspace == self.uspace and
                    other.dspace == self.dspace and
                    other.sampling == self.sampling and
                    other.interpolation == self.interpolation)

    def __hash__(self):
        """Return ``hash(self)``."""

        return hash((super().__hash__(), self.uspace, self.dspace,
                     self.sampling, self.interpolation))

    @property
    def impl(self):
        """Underlying implmentation type for the dspace."""
        return self.dspace.impl

    @property
    def domain(self):
        """Domain of the continuous space."""
        return self.uspace.domain

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
        return self.dspace.weighting

    @property
    def is_weighted(self):
        """``True`` if the ``dspace`` is weighted."""
        return getattr(self.dspace, 'is_weighted', False)

    def _lincomb(self, a, x1, b, x2, out):
        """Raw linear combination."""
        self.dspace._lincomb(a, x1.tensor, b, x2.tensor, out.tensor)

    def _dist(self, x1, x2):
        """Raw distance between two elements."""
        return self.dspace._dist(x1.tensor, x2.tensor)

    def _norm(self, x):
        """Raw norm of an element."""
        return self.dspace._norm(x.tensor)

    def _inner(self, x1, x2):
        """Raw inner product of two elements."""
        return self.dspace._inner(x1.tensor, x2.tensor)

    def _multiply(self, x1, x2, out):
        """Raw pointwise multiplication of two elements."""
        self.dspace._multiply(x1.tensor, x2.tensor, out.tensor)

    def _divide(self, x1, x2, out):
        """Raw pointwise multiplication of two elements."""
        self.dspace._divide(x1.tensor, x2.tensor, out.tensor)

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
        """Type of elements in this space: `DiscretizedSpaceElement`."""
        return DiscretizedSpaceElement


class DiscretizedSpaceElement(Tensor):

    """Representation of a `DiscretizedSpace` element.

    Basically only a wrapper class for dspace's element class."""

    def __init__(self, space, tensor):
        """Initialize a new instance."""
        Tensor.__init__(self, space)
        self.__tensor = tensor

    @property
    def tensor(self):
        """Structure for data storage."""
        return self.__tensor

    @property
    def dtype(self):
        """Type of data storage."""
        return self.tensor.dtype

    @property
    def size(self):
        """Size of data storage."""
        return self.tensor.size

    def __len__(self):
        """Return ``len(self)``.

        Size of data storage.
        """
        return self.size

    def copy(self):
        """Create an identical (deep) copy of this element."""
        return self.space.element(self.tensor.copy())

    def asarray(self, out=None):
        """Extract the data of this array as a numpy array.

        Parameters
        ----------
        out : `numpy.ndarray`, optional
            Array in which the result should be written in-place.
            Has to be contiguous and of the correct dtype.
        """
        return self.tensor.asarray(out=out)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if all entries of ``other`` are equal to this
            element's entries, ``False`` otherwise.
        """
        return (other in self.space and
                self.tensor == other.tensor)

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        Parameters
        ----------
        indices : int or `slice`
            The position(s) that should be accessed

        Returns
        -------
        values : `Tensor`
            The value(s) at the index (indices)
        """
        return self.tensor.__getitem__(indices)

    def __setitem__(self, indices, values):
        """Implement ``self[indices] = values``.

        Parameters
        ----------
        indices : int or `slice`
            The position(s) that should be set
        values : scalar, `array-like` or `Tensor`
            The value(s) that are to be assigned.

            If ``index`` is an int, ``value`` must be single value.

            If ``index`` is a slice, ``value`` must be broadcastable
            to the size of the slice (same size, shape (1,)
            or single value).
        """
        input_data = getattr(values, 'tensor', values)
        self.tensor.__setitem__(indices, input_data)

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
        DiscretizedSpace.sampling : For full description
        """
        self.space.sampling(ufunc, out=self.tensor, **kwargs)

    @property
    def interpolation(self):
        """Interpolation operator associated with this element.

        Returns
        -------
        interpolation_op : `FunctionSpaceMapping`
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
        DiscretizedSpace.interpolation : For full description
        """
        return self.space.interpolation(self.tensor)

    def __ipow__(self, p):
        """Implement ``self **= p``."""
        # The concrete `tensor` can specialize `__ipow__` for non-integer
        # `p` so we want to use it here. Otherwise we get the default
        # `LinearSpaceElement.__ipow__` which only works for integer `p`.
        self.tensor.__ipow__(p)
        return self

    def __str__(self):
        """Return ``str(self)``."""
        return arraynd_str(self.asarray())

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_str = arraynd_repr(self.asarray())
        if self.space.ndim == 1:
            return '{!r}.element({})'.format(self.space, inner_str)
        else:
            return '{!r}.element(\n{}\n)'.format(self.space,
                                                 indent_rows(inner_str))


def dspace_type(space, impl, dtype=None):
    """Select the correct corresponding tensor space.

    Parameters
    ----------
    space : `LinearSpace`
        Template space from which to infer an adequate data space. If
        it has a `LinearSpace.field` attribute, ``dtype`` must be
        consistent with it.
    impl : string
        Implementation backend for the data space
    dtype : optional
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
    elif is_numeric_dtype(dtype):
        if field_type == ComplexNumbers:
            raise TypeError('non-floating data type {!r} requires space field '
                            'to be of type RealNumbers, got {!r}'
                            .format(dtype, field_type))
    else:
        raise TypeError('non-numeric data type {!r} cannot be combined with '
                        'a `LinearSpace`'.format(dtype))

    stype = TENSOR_SPACE_IMPLS.get(impl, None)

    if stype is None:
        raise NotImplementedError('no corresponding data space available '
                                  'for space {!r} and implementation {!r}'
                                  ''.format(space, impl))
    return stype


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
