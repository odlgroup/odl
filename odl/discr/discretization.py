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

# pylint: disable=abstract-method

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import super
from future.utils import with_metaclass

# External
from abc import ABCMeta

# ODL
from odl.util.utility import arraynd_repr, arraynd_str
from odl.operator.operator import Operator, LinearOperator
from odl.space.base_ntuples import NtuplesBase, FnBase, Ntuples, Rn, Cn
from odl.set.sets import Set, RealNumbers, ComplexNumbers
from odl.set.space import LinearSpace
from odl.space import CUDA_AVAILABLE
if CUDA_AVAILABLE:
    from odl.space.cu_ntuples import CudaNtuples, CudaRn
    CudaCn = type(None)  # TODO: add CudaCn to imports once it is implemented
else:
    CudaRn = CudaCn = CudaNtuples = type(None)


__all__ = ('RawDiscretization', 'Discretization')


class RawDiscretization(with_metaclass(ABCMeta, NtuplesBase)):

    """Abstract raw discretization class.

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
    the source set to an `n`-tuple. This mapping is called
    **restriction** in ODL.
    The second one encodes the converse way of mapping an `n`-tuple to
    an element of the original set. This mapping is called
    **extension**.
    """

    def __init__(self, uspace, dspace, restr=None, ext=None, **kwargs):
        """Abstract initialization method.

        Intended to be called by subclasses for proper type checking
        and setting of attributes.

        Parameters
        ----------
        uspace : `Set`
            The undiscretized (abstract) set to be discretized
        dspace : `NtuplesBase`
            Data space providing containers for the values of a
            discretized object
        restr : `Operator`, optional
            Operator mapping a `uspace` element to a `dspace` element.
            Must satisfy `restr.domain == uspace`,
            `restr.range == dspace`.
        ext : `Operator`, optional
            Operator mapping a `dspace` element to a `uspace` element.
            Must satisfy `ext.domain == dspace`,
            `ext.range == uspace.

            """
        if not isinstance(uspace, Set):
            raise TypeError('undiscretized space {} not a `Set` instance.'
                            ''.format(uspace))

        if not isinstance(dspace, NtuplesBase):
            raise TypeError('data space {} not an `NtuplesBase` instance.'
                            ''.format(dspace))

        if restr is not None:
            if not isinstance(restr, Operator):
                raise TypeError('restriction operator {} not an `Operator` '
                                'instance.'.format(restr))

            if restr.domain != uspace:
                raise ValueError('restriction operator domain {} not equal to '
                                 'the undiscretized space {}.'
                                 ''.format(restr.domain, dspace))

            if restr.range != dspace:
                raise ValueError('restriction operator range {} not equal to'
                                 'the data space {}.'
                                 ''.format(restr.range, dspace))

        if ext is not None:
            if not isinstance(ext, Operator):
                raise TypeError('extension operator {} not an `Operator` '
                                'instance.'.format(ext))

            if ext.domain != dspace:
                raise ValueError('extension operator domain {} not equal to'
                                 'the data space {}.'
                                 ''.format(ext.domain, dspace))

            if ext.range != uspace:
                raise ValueError('extension operator range {} not equal to'
                                 'the undiscretized space {}.'
                                 ''.format(ext.range, uspace))

        super().__init__(dspace.size, dspace.dtype)
        self._uspace = uspace
        self._dspace = dspace
        self._restriction = restr
        self._extension = ext

    @property
    def uspace(self):
        """The undiscretized space."""
        return self._uspace

    @property
    def dspace(self):
        """The data space."""
        return self._dspace

    @property
    def dspace_type(self):
        """Data space type of this discretization."""
        return type(self._dspace)

    @property
    def restriction(self):
        """The operator mapping a `uspace` element to an n-tuple."""
        if self._restriction is not None:
            return self._restriction
        else:
            raise NotImplementedError('no restriction operator provided.')

    @property
    def extension(self):
        """The operator mapping an n-tuple to a `uspace` element."""
        if self._extension is not None:
            return self._extension
        else:
            raise NotImplementedError('no extension operator provided.')

    def element(self, inp=None):
        """Create an element from `inp` or from scratch.

        Parameters
        ----------
        inp : `object`, optional
            The input data to create an element from. Must be
            recognizable by the `element()` method of either `dspace`
            or `uspace`.

        Returns
        -------
        element : `RawDiscretization.Vector`
            The discretized element, calculated as
            `dspace.element(inp)` or
            `restriction(uspace.element(inp))`, tried in this order.
        """
        if inp is None:
            return self.Vector(self, self.dspace.element())
        elif inp in self.uspace:
            return self.Vector(
                self, self.restriction(self.uspace.element(inp)))
        else:  # Sequence-type input
            return self.Vector(self, self.dspace.element(inp))

    def __eq__(self, other):
        """`s.__eq__(other) <==> s == other`.

        Returns
        -------
        equals : `bool`
            `True` if `other` is a `RawDiscretization` instance and
            all attributes `uspace`, `dspace`, `restriction` and
            `extension` of `other` and this discretization are equal,
            `False` otherwise.
        """
        if other is self:
            return True

        return (super().__eq__(other) and
                other.uspace == self.uspace and
                other.dspace == self.dspace and
                other.restriction == self.restriction and
                other.extension == self.extension)

    @property
    def domain(self):
        """The domain of the continuous space."""
        return self.uspace.domain

    @property
    def dtype(self):
        """The dtype of the representation space."""
        return self.dspace.dtype

    class Vector(NtuplesBase.Vector):

        """Representation of a `RawDiscretization` element.

        Basically only a wrapper class for dspace's vector class."""

        def __init__(self, space, ntuple):
            """Initialize a new instance."""
            if not isinstance(space, RawDiscretization):
                raise TypeError('space {!r} not a `RawDiscretization` '
                                'instance.'.format(space))

            if not isinstance(ntuple, space.dspace.Vector):
                raise TypeError('n-tuple {!r} not an instance of `{}.Vector`.'
                                ''.format(ntuple,
                                          space.dspace.__class__.__name__))
            super().__init__(space)
            self._ntuple = ntuple

        @property
        def ntuple(self):
            """Structure for data storage."""
            return self._ntuple

        @property
        def dtype(self):
            """type of data storage."""
            return self.ntuple.dtype

        @property
        def size(self):
            """size of data storage."""
            return self.ntuple.size

        def copy(self):
            """Create an identical (deep) copy of this vector."""
            return self.space.element(self.ntuple.copy())

        def asarray(self, out=None):
            """Extract the data of this array as a numpy array.

            Parameters
            ----------
            out : `ndarray`, Optional (default: `None`)
                Array in which the result should be written in-place.
                Has to be contiguous and of the correct dtype.
            """
            return self.ntuple.asarray(out=out)

        def __eq__(self, other):
            """`vec.__eq__(other) <==> vec == other`.

            Returns
            -------
            equals : `bool`
                `True` if all entries of `other` are equal to this
                vector's entries, `False` otherwise.
            """
            return (type(other) == type(self) and
                    self.ntuple == other.ntuple)

        def __getitem__(self, indices):
            """Access values of this vector.

            Parameters
            ----------
            indices : `int` or `slice`
                The position(s) that should be accessed

            Returns
            -------
            values : `dspace_type.Vector`
                The value(s) at the index (indices)
            """
            return self.ntuple.__getitem__(indices)

        def __setitem__(self, indices, values):
            """Set values of this vector.

            Parameters
            ----------
            indices : `int` or `slice`
                The position(s) that should be set
            values : {scalar, array-like, `NtuplesBase.Vector`}
                The value(s) that are to be assigned.

                If `index` is an `int`, `value` must be single value.

                If `index` is a `slice`, `value` must be broadcastable
                to the size of the slice (same size, shape (1,)
                or single value).
            """
            if isinstance(values, RawDiscretization.Vector):
                self.ntuple.__setitem__(indices, values.ntuple)
            else:
                self.ntuple.__setitem__(indices, values)

        def __str__(self):
            """`vec.__str__() <==> str(vec)`."""
            return arraynd_str(self.asarray())

        def __repr__(self):
            """`vec.__repr__() <==> repr(vec)`."""
            return '{!r}.element({})'.format(self.space,
                                             arraynd_repr(self.asarray()))


class Discretization(with_metaclass(ABCMeta, RawDiscretization,
                                    FnBase)):

    """Abstract class for discretizations of linear vector spaces.

    This variant of `RawDiscretization` adds linear structure to all
    its members. The `uspace` is a linear space, the `dspace`
    for the data representation is an implementation of :math:`F^n`,
    where `F` is some field, and both `restriction` and `extension`
    are linear operators.
    """

    def __init__(self, uspace, dspace, restr=None, ext=None, **kwargs):
        """Abstract initialization method.

        Intended to be called by subclasses for proper type checking
        and setting of attributes.

        Parameters
        ----------
        uspace : `LinearSpace`
            The (abstract) space to be discretized
        dspace : `FnBase`
            Data space providing containers for the values of a
            discretized object. Its `field` attribute must be the same
            as `uspace.field`.
        restr : `LinearOperator`, optional
            Operator mapping a `uspace` element to a `dspace` element.
            Must satisfy `restr.domain == uspace`,
            `restr.range == dspace`
        ext : `LinearOperator`, optional
            Operator mapping a `dspace` element to a `uspace` element.
            Must satisfy `ext.domain == dspace`,
            `ext.range == uspace`
        """
        super().__init__(uspace, dspace, restr, ext, **kwargs)
        FnBase.__init__(self, dspace.size, dspace.dtype)

        if not isinstance(uspace, LinearSpace):
            raise TypeError('undiscretized space {} not a `LinearSpace` '
                            'instance.'.format(uspace))

        if not isinstance(dspace, FnBase):
            raise TypeError('data space {} not an `FnBase` instance.'
                            ''.format(dspace))

        if uspace.field != dspace.field:
            raise ValueError('fields {} and {} of the undiscretized and '
                             'data spaces, resp., are not equal.'
                             ''.format(uspace.field, dspace.field))

        if restr is not None:
            if not isinstance(restr, LinearOperator):
                raise TypeError('restriction operator {} is not a '
                                '`LinearOperator` instance.'.format(restr))

        if ext is not None:
            if not isinstance(ext, LinearOperator):
                raise TypeError('extension operator {} is not a '
                                '`LinearOperator` instance.'.format(ext))

    # Pass-through attributes of the wrapped `dspace`
    def zero(self):
        """Create a vector of zeros."""
        return self.Vector(self, self.dspace.zero())

    def _lincomb(self, z, a, x, b, y):
        """Raw linear combination."""
        self.dspace._lincomb(z.ntuple, a, x.ntuple, b, y.ntuple)

    def _dist(self, x, y):
        """Raw distance between two vectors."""
        # TODO: implement inner product according to correspondence
        # principle!
        return self.dspace._dist(x.ntuple, y.ntuple)

    def _norm(self, x):
        """Raw norm of a vector."""
        # TODO: implement inner product according to correspondence
        # principle!
        return self.dspace._norm(x.ntuple)

    def _inner(self, x, y):
        """Raw inner product of two vectors."""
        # TODO: implement inner product according to correspondence
        # principle!
        return self.dspace._inner(x.ntuple, y.ntuple)

    def _multiply(self, z, x, y):
        """Raw pointwise multiplication of two vectors."""
        self.dspace._multiply(z.ntuple, x.ntuple, y.ntuple)

    class Vector(RawDiscretization.Vector, FnBase.Vector):

        """Representation of a `Discretization` element."""

        def __init__(self, space, data):
            """Initialize a new instance."""
            if not isinstance(space, Discretization):
                raise TypeError('space {!r} not a `Discretization` '
                                'instance.'.format(space))
            super().__init__(space, data)


def dspace_type(space, impl):
    """Select the correct corresponding n-tuples space.

    Parameters
    ----------
    space : `LinearSpace`
        The template space
    impl : {'numpy', 'cuda'}
        The backend for the data space

    Returns
    -------
    dtype : type
        Space type selected after the space's field and the chosen
        backend
    """
    if impl not in ('numpy', 'cuda'):
        raise ValueError('implementation type {} not understood.'
                         ''.format(impl))

    if impl == 'cuda' and not CUDA_AVAILABLE:
        raise ValueError('CUDA implementation not available.')

    spacetype_map = {
        'numpy': {RealNumbers: Rn, ComplexNumbers: Cn, None: Ntuples},
        'cuda': {RealNumbers: CudaRn, ComplexNumbers: None, None: CudaNtuples}
    }

    field_type = None if not hasattr(space, 'field') else type(space.field)
    stype = spacetype_map[impl][field_type]
    if stype is None:
        raise NotImplementedError('no corresponding data space available for '
                                  'space {!r} and implementation {!r}.'
                                  ''.format(space, impl))
    return stype

if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
