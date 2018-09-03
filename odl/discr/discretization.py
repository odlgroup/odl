# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Base classes for discretization."""

from __future__ import print_function, division, absolute_import

from odl.operator import Operator
from odl.set.sets import Set
from odl.space.base_tensors import TensorSpace, Tensor
from odl.space.entry_points import tensor_space_impl
from odl.set import RealNumbers, ComplexNumbers
from odl.util import (
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
    the set to be discretized ("function space" or ``fspace``), and a
    backend for storage and processing of the discrete values
    ("tensor space" or ``tspace``).
    Since function spaces represent by far the most significant application,
    the non-discretized set is called ``fspace``.

    As additional information, two mappings can be provided.
    The first one is an explicit way to map an (abstract) element from
    ``fspace`` to a tensor in ``tspace``. This mapping is called
    **sampling** in ODL.
    The second one encodes the converse way of mapping a tensor to
    an element of the original set. This mapping is called
    **interpolation**.
    """

    def __init__(self, fspace, tspace, sampling=None, interpol=None):
        """Abstract initialization method.

        Intended to be called by subclasses for proper type checking
        and setting of attributes.

        Parameters
        ----------
        fspace : `Set`
            The non-discretized (abstract) set to be discretized.
        tspace : `TensorSpace`
            Space providing containers for the values/coefficients of a
            discretized object.
        sampling : `Operator`, optional
            Operator mapping an `fspace` element to a `tspace` element.
            Must satisfy ``sampling.domain == fspace``,
            ``sampling.range == tspace``.
        interpol : `Operator`, optional
            Operator mapping a `tspace` element to an `fspace` element.
            Must satisfy ``interpol.domain == tspace``,
            ``interpol.range == fspace``.
            """
        if not isinstance(fspace, Set):
            raise TypeError('`fspace` must be a `Set` instance, '
                            'got {!r}'.format(fspace))
        if not isinstance(tspace, TensorSpace):
            raise TypeError('`tspace` {!r} not a `TensorSpace` instance'
                            ''.format(tspace))

        if sampling is not None:
            if not isinstance(sampling, Operator):
                raise TypeError('`sampling` {!r} not an `Operator` '
                                'instance'.format(sampling))
            if sampling.domain != fspace:
                raise ValueError('`sampling.domain` {!r} not equal to '
                                 '`fspace` {!r}'
                                 ''.format(sampling.domain, tspace))
            if sampling.range != tspace:
                raise ValueError('`sampling.range` {!r} not equal to'
                                 '`tspace` {!r}'
                                 ''.format(sampling.range, tspace))

        if interpol is not None:
            if not isinstance(interpol, Operator):
                raise TypeError('`interpol` {!r} not an Operator '
                                'instance'.format(interpol))
            if interpol.domain != tspace:
                raise ValueError('`interpol.domain` {} not equal to '
                                 '`tspace` {!r}'
                                 ''.format(interpol.domain, tspace))
            if interpol.range != fspace:
                raise ValueError('`interpol.range` {!r} not equal to '
                                 '`fspace` {!r}'
                                 ''.format(interpol.range, fspace))

        super(DiscretizedSpace, self).__init__(tspace.shape, tspace.dtype)
        self.__fspace = fspace
        self.__tspace = tspace
        self.__sampling = sampling
        self.__interpolation = interpol

    @property
    def fspace(self):
        """Non-discretized space of this discretization."""
        return self.__fspace

    @property
    def tspace(self):
        """Space for the coefficients of the elements of this space."""
        return self.__tspace

    @property
    def tspace_type(self):
        """Tensor space type of this discretization."""
        return type(self.tspace)

    @property
    def sampling(self):
        """Operator mapping an `fspace` element to a `Tensor`."""
        if self.__sampling is not None:
            return self.__sampling
        else:
            raise NotImplementedError('no sampling operator provided')

    @property
    def interpolation(self):
        """Operator mapping a `Tensor` to an `fspace` element."""
        if self.__interpolation is not None:
            return self.__interpolation
        else:
            raise NotImplementedError('no interpolation operator provided')

    def element(self, inp=None, order=None, **kwargs):
        """Create an element from ``inp`` or from scratch.

        Parameters
        ----------
        inp : optional
            Input data to create an element from. It needs to be
            understood by either the `sampling` operator of this
            instance or by its ``tspace.element`` method.
        order : {None, 'C', 'F'}, optional
            Storage order of the returned element. For ``'C'`` and ``'F'``,
            contiguous memory in the respective ordering is enforced.
            The default ``None`` enforces no contiguousness.
        kwargs :
            Additional arguments passed on to `sampling` when called
            on ``inp``, in the form ``sampling(inp, **kwargs)``.
            This can be used e.g. for functions with parameters.

        Returns
        -------
        element : `DiscretizedSpaceElement`
            The discretized element, calculated as ``sampling(inp)`` or
            ``tspace.element(inp)``, tried in this order.

        See Also
        --------
        sampling : create a discrete element from a non-discretized one
        """
        if inp is None:
            return self.element_type(self, self.tspace.element(order=order))
        elif inp in self and order is None:
            return inp
        elif callable(inp):
            sampled = self.sampling(inp, **kwargs)
            return self.element_type(self,
                                     self.tspace.element(sampled, order=order))
        else:
            return self.element_type(self,
                                     self.tspace.element(inp, order=order))

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is a `DiscretizedSpace`
            instance and all attributes `fspace`, `tspace`,
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
            return (super(DiscretizedSpace, self).__eq__(other) and
                    other.fspace == self.fspace and
                    other.tspace == self.tspace and
                    (getattr(other, 'sampling', None) ==
                     getattr(self, 'sampling', None)) and
                    (getattr(other, 'interpolation', None) ==
                     getattr(self, 'interpolation', None)))

    def __hash__(self):
        """Return ``hash(self)``."""
        prop_list = [super(DiscretizedSpace, self).__hash__(),
                     self.fspace, self.tspace]
        # May not exist
        try:
            prop_list.append(self.sampling)
        except NotImplementedError:
            pass
        try:
            prop_list.append(self.interpolation)
        except NotImplementedError:
            pass

        return hash(tuple(prop_list))

    @property
    def domain(self):
        """Domain of the continuous space."""
        return self.fspace.domain

    def zero(self):
        """Return the element of all zeros."""
        return self.element_type(self, self.tspace.zero())

    def one(self):
        """Return the element of all ones."""
        return self.element_type(self, self.tspace.one())

    @property
    def weighting(self):
        """This space's weighting scheme."""
        return self.tspace.weighting

    @property
    def is_weighted(self):
        """``True`` if the ``tspace`` is weighted."""
        return getattr(self.tspace, 'is_weighted', False)

    @property
    def impl(self):
        """Name of the implementation back-end."""
        return self.tspace.impl

    def _lincomb(self, a, x1, b, x2, out):
        """Raw linear combination."""
        self.tspace._lincomb(a, x1.tensor, b, x2.tensor, out.tensor)

    def _dist(self, x1, x2):
        """Raw distance between two elements."""
        return self.tspace._dist(x1.tensor, x2.tensor)

    def _norm(self, x):
        """Raw norm of an element."""
        return self.tspace._norm(x.tensor)

    def _inner(self, x1, x2):
        """Raw inner product of two elements."""
        return self.tspace._inner(x1.tensor, x2.tensor)

    def _multiply(self, x1, x2, out):
        """Raw pointwise multiplication of two elements."""
        self.tspace._multiply(x1.tensor, x2.tensor, out.tensor)

    def _divide(self, x1, x2, out):
        """Raw pointwise multiplication of two elements."""
        self.tspace._divide(x1.tensor, x2.tensor, out.tensor)

    @property
    def examples(self):
        """Return example functions in the space.

        These are created by discretizing the examples in the underlying
        `fspace`.

        See Also
        --------
        odl.space.fspace.FunctionSpace.examples
        """
        for name, elem in self.fspace.examples:
            yield (name, self.element(elem))

    @property
    def element_type(self):
        """Type of elements in this space: `DiscretizedSpaceElement`."""
        return DiscretizedSpaceElement


class DiscretizedSpaceElement(Tensor):

    """Representation of a `DiscretizedSpace` element.

    Basically only a wrapper class for tspace's element class."""

    def __init__(self, space, tensor):
        """Initialize a new instance."""
        super(DiscretizedSpaceElement, self).__init__(space)
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

    def astype(self, dtype):
        """Return a copy of this element with new ``dtype``.

        Parameters
        ----------
        dtype :
            Scalar data type of the returned space. Can be provided
            in any way the `numpy.dtype` constructor understands, e.g.
            as built-in type or as a string. Data types with non-trivial
            shapes are not allowed.

        Returns
        -------
        newelem : `DiscretizedSpaceElement`
            Version of this element with given data type.
        """
        return self.space.astype(dtype).element(self.tensor.astype(dtype))

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
            The position(s) that should be accessed.

        Returns
        -------
        values : `Tensor`
            The value(s) at the index (indices).
        """
        if isinstance(indices, type(self)):
            indices = indices.tensor
        return self.tensor[indices]

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
        if isinstance(indices, type(self)):
            indices = indices.tensor
        if isinstance(values, type(self)):
            values = values.tensor
        self.tensor.__setitem__(indices, values)

    def sampling(self, ufunc, **kwargs):
        """Sample a continuous function and assign to this element.

        Parameters
        ----------
        ufunc : ``self.space.fspace`` element
            The continuous function that should be samplingicted.
        kwargs :
            Additional arugments for the sampling operator implementation

        Examples
        --------
        >>> space = odl.uniform_discr(0, 1, 5)
        >>> x = space.element()

        Assign x according to a continuous function:

        >>> x.sampling(lambda t: t)
        >>> x  # Print values at grid points (which are centered)
        uniform_discr(0.0, 1.0, 5).element([ 0.1,  0.3,  0.5,  0.7,  0.9])

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


def tspace_type(space, impl, dtype=None):
    """Select the correct corresponding tensor space.

    Parameters
    ----------
    space : `LinearSpace`
        Template space from which to infer an adequate tensor space. If
        it has a ``field`` attribute, ``dtype`` must be consistent with it.
    impl : string
        Implementation backend for the tensor space.
    dtype : optional
        Data type which the space is supposed to use. If ``None`` is
        given, the space type is purely determined from ``space`` and
        ``impl``. Otherwise, it must be compatible with the
        field of ``space``.

    Returns
    -------
    stype : type
        Space type selected after the space's field, the backend and
        the data type.
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

    try:
        return tensor_space_impl(impl)
    except ValueError:
        raise NotImplementedError('no corresponding tensor space available '
                                  'for space {!r} and implementation {!r}'
                                  ''.format(space, impl))


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
