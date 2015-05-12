# Copyright 2014, 2015 Holger Kohr, Jonas Adler
#
# This file is part of RL.
#
# RL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RL.  If not, see <http://www.gnu.org/licenses/>.


# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
try:
    from builtins import str, zip, range, super
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import str, zip, range, super
from future import standard_library

# 
from itertools import repeat

# RL imports
from RL.space.space import HilbertSpace, NormedSpace, MetricSpace, LinearSpace
from RL.utility.utility import errfmt

standard_library.install_aliases()


class ProductSpace(LinearSpace):
    """Product space (X1 x X2 x ... x Xn)
    Creates a Cartesian product of an arbitrary set of spaces.

    For example:

    `ProductSpace(Reals(), Reals())` is mathematically equivalent to `RN(2)`

    Note that the later is obviously more efficient.

    Args:
        spaces    (multiple) LinearSpace     One or more instances of linear spaces
    """

    def __init__(self, *spaces):        
        if len(spaces) == 0:
            raise TypeError("Empty product not allowed")

        if not all(spaces[0].field == y.field for y in spaces):
            raise TypeError("All spaces must have the same field")

        self.spaces = spaces
        self._nProducts = len(self.spaces)
        self._field = spaces[0].field  # X_n has same field

    def zero(self):
        """ Create a vector whose components is the zero vectors in the underlying spaces
        """
        return self.makeVector(*[space.zero() for space in self.spaces])

    def empty(self):
        """ Create a vector whose components is arbitrary vectors in the underlying spaces
        """
        return self.makeVector(*[space.empty() for space in self.spaces])

    def linCombImpl(self, z, a, x, b, y):
        """ Applies linComb to each component
        """
        for space, zp, xp, yp in zip(self.spaces, z.parts, x.parts, y.parts):
            space.linCombImpl(zp, a, xp, b, yp)

    @property
    def field(self):
        """ Get the underlying field of this product space.
        The field is the same for each underlying space.
        """
        return self._field

    def equals(self, other):
        """ Tests two objects for equality.

        Returns true if other is a ProductSpace with the same subspaces as this space.
        """
        return (isinstance(other, ProductSpace) and
                self._nProducts == other._nProducts and
                all(x.equals(y) for x, y in zip(self.spaces, other.spaces)))

    def makeVector(self, *args):
        """ Creates an element in the product space

        Parameters
        ----------
        The method has two call patter, the first is:

        *args : tuple of LinearSpace.Vector's
                A tuple of vectors in the underlying space.
                This will simply wrap the Vectors (not copy).

        The second pattern is to create a new Vector from scratch, in this case

        *args : Argument for i:th vector

        Returns
        -------
        ProductSpace.Vector instance


        Examples
        --------

        >>> rn = RN(2)
        >>> rm = RN(3)
        >>> prod = ProductSpace(rn,rm)
        >>> x = rn.makeVector([1,2])
        >>> y = rm.makeVector([1,2,3])
        >>> z = prod.makeVector(x, y)
        >>> z
        {[ 1.  2.], [ 1.  2.  3.]}

        """

        if not isinstance(args[0], LinearSpace.Vector):
            # Delegate constructors
            return self.makeVector(*(space.makeVector(arg)
                                   for arg, space in zip(args, self.spaces)))
        else:  # Construct from existing tuple
            if any(part.space != space
                    for part, space in zip(args, self.spaces)):
                raise TypeError(errfmt('''
                The spaces of all parts must correspond to this
                space's parts'''))

            #Use class to allow subclassing
            return self.__class__.Vector(self, *args)

    def __len__(self):
        """ Get the number of parts of this product space
        """
        return self._nProducts

    def __getitem__(self, index):
        """ Access the index:th subspace of this space

        Parameters
        ----------

        index : int
                The position that should be accessed

        Returns
        -------
        LinearSpace, the index:th subspace

        Examples
        --------

        >>> rn = RN(2)
        >>> rm = RN(3)
        >>> prod = ProductSpace(rn, rm)
        >>> prod[0]
        RN(2)
        >>> prod[1]
        RN(3)

        """
        return self.spaces[index]

    def __str__(self):
        return ' x '.join(str(space) for space in self.spaces)

    def __repr__(self):
        return ('ProductSpace(' +
                ', '.join(str(space) for space in self.spaces) + ')')

    class Vector(LinearSpace.Vector):
        def __init__(self, space, *args):
            super().__init__(space)
            self.parts = args

        def __len__(self):
            """ The number of components of this vector
            """
            return self.space._nProducts

        def __getitem__(self, index):
            """ Access the index:th component of this vector

            Parameters
            ----------

            index : int
                    The position that should be accessed

            Returns
            -------
            LinearSpace.Vector of type self.space[index], the index:th sub-vector

            Examples
            --------

            >>> rn = RN(2)
            >>> rm = RN(3)
            >>> prod = ProductSpace(rn, rm)
            >>> z = prod.makeVector([1, 2], [1, 2, 3])
            >>> z[0]
            [1.0, 2.0]
            >>> z[1]
            [1.0, 2.0, 3.0]

            """
            return self.parts[index]

        def __str__(self):
            return ('{' + ', '.join(str(part) for part in self.parts) +
                    '}')

        def __repr__(self):
            return (repr(self.space) + '.makeVector(' +
                    ', '.join(repr(part) for part in self.parts) + ')')

    
class MetricProductSpace(ProductSpace, MetricSpace):
    """ A product space of Metric Spaces (X1 x X2 x ... x Xn)

    Creates a Cartesian product of an arbitrary set of spaces.

    For example:

    `MetricProductSpace(Reals(), Reals())` is mathematically equivalent to `RN(2)` ##TODO MAKE METRIC RN SPACE

    Note that the later is obviously more efficient.

    Parameters
    ----------

    *spaces : MetricSpace's       
              A set of normed spaces
    **kwargs : {'ord'}
                'ord' : string, optional
                        The manner in which to combine the metrics. Default value 2

    The following values for `ord` can be specified. 

    =====  ================================================================
    ord    Definition
    =====  ================================================================
    inf    max(dist(x[0],y[0]), ..., dist(x[n-1],y[n-1]))
    -inf   min(dist(x[0],y[0]), ..., dist(x[n-1],y[n-1]))
    0      (dist(x[0],y[0]) != 0 + ... + dist(x[n-1],y[n-1]) != 0)
    other  (dist(x[0],y[0])**ord + ... + dist(x[n-1],y[n-1])**ord)**(1/ord)
    =====  ================================================================
    """
    
    def __init__(self, *spaces, **kwargs):
        self.ord = kwargs.pop('ord', 2)

        super().__init__(*spaces)

    def distImpl(self, x, y):
        if self.ord == float('inf'):
            return max(space.distImpl(xp, yp) for space, xp, yp in zip(self.spaces, x.parts, y.parts))        
        elif self.ord == -float('inf'):
            return min(space.distImpl(xp, yp) for space, xp, xp, yp in zip(self.spaces, x.parts, y.parts))
        elif self.ord == 0:
            return sum(space.distImpl(xp, yp) != 0 for space, xp, xp, yp in zip(self.spaces, x.parts, y.parts))
        else:
            return sum(space.distImpl(xp, yp)**self.ord for space, xp, xp, yp in zip(self.spaces, x.parts, y.parts))**(1/self.ord)

    def __repr__(self):
        return ('MetricProductSpace(' +
                ', '.join(str(space) for space in self.spaces) + ')')
        
    class Vector(ProductSpace.Vector, MetricSpace.Vector):
        pass

class NormedProductSpace(MetricProductSpace, NormedSpace):
    """ A product space of Normed Spaces (X1 x X2 x ... x Xn)

    Creates a Cartesian product of an arbitrary set of spaces.

    For example:

    `NormedProductSpace(Reals(), Reals())` is mathematically equivalent to `NormedRN(2)`

    Note that the later is obviously more efficient.

    Parameters
    ----------

    *spaces : NormedSpace's       
                A set of normed spaces
    **kwargs : {'ord'}
                'ord' : string
                        The order of the norm.

    The following values for `ord` can be specified. 
    Note that any value of ord < 1 only gives a pseudonorm

    =====  ================================================================
    ord    Definition
    =====  ================================================================
    inf    max(norm(x[0]), ..., norm(x[n-1]))
    -inf   min(norm(x[0]), ..., norm(x[n-1]))
    0      (norm(x[0]) != 0 + ... + norm(x[n-1]) != 0)
    other  (norm(x[0])**ord + ... + norm(x[n-1])**ord)**(1/ord)
    =====  ================================================================
    """
    
    def __init__(self, *spaces, **kwargs):
        self.ord = kwargs.pop('ord',2)

        super().__init__(*spaces)

    def normImpl(self, x):
        if self.ord == float('inf'):
            return max(space.normImpl(xp) for space, xp in zip(self.spaces, x.parts))        
        elif self.ord == -float('inf'):
            return min(space.normImpl(xp) for space, xp in zip(self.spaces, x.parts))
        elif self.ord == 0:
            return sum(space.normImpl(xp) != 0 for space, xp in zip(self.spaces, x.parts))
        else:
            return sum(space.normImpl(xp)**self.ord for space, xp in zip(self.spaces, x.parts))**(1/self.ord)

    def __repr__(self):
        return ('NormedProductSpace(' +
                ', '.join(str(space) for space in self.spaces) + ')')
        
    class Vector(MetricProductSpace.Vector, NormedSpace.Vector):
        pass

class HilbertProductSpace(NormedProductSpace, HilbertSpace):
    """ A product space of HilbertSpaces (X1 x X2 x ... x Xn)

    Creates a Cartesian product of an arbitrary set of spaces.

    For example:

    `HilbertProductSpace(Reals(), Reals())` is mathematically equivalent to `EuclideanSpace(2)`

    Note that the later is obviously more efficient.

    Parameters
    ----------

    *spaces : HilbertSpace's       
              A set of HilbertSpace's
    **kwargs : {'weights'}
                'weights' : Array-Like
                            List of weights, same size as spaces

    The inner product in the HilbertProductSpace is per default defined as

    inner(x, y) = x[0]*y[0] + ... + x[n-1]*y[n-1]

    The optional parameter `weights` changes this behaviour to

    inner(x, y) = weights[0]*x[0]*y[0] + ... + weights[n-1]*x[n-1]*y[n-1]
    """

    def __init__(self, *spaces, **kwargs):
        self.weights = kwargs.pop('weights', None)

        super().__init__(*spaces)

    def innerImpl(self, x, y):
        if self.weights:
            return sum(space.innerImpl(xp, yp)
                       for space, weight, xp, yp in zip(self.spaces, self.weights, x.parts, y.parts))
        else:
            return sum(space.innerImpl(xp, yp)
                       for space, xp, yp in zip(self.spaces, x.parts, y.parts))

    def __repr__(self):
        return ('HilbertProductSpace(' +
                ', '.join(str(space) for space in self.spaces) + ')')

    class Vector(ProductSpace.Vector, HilbertSpace.Vector):
        pass

def makeProductSpace(*spaces):
    """ Creates an appropriate ProductSpace

    Selects the type of product space that has the most structure (Inner product, norm)
    given a set of spaces

    Parameters
    ----------

    *spaces : LinearSpace's       
              A set of LinearSpace's

    Returns
    -------
    ProductSpace, NormedProductSpace or HilbertProductSpace depending on the lowest common denominator in spaces.

    """
    if all(isinstance(space, HilbertSpace) for space in spaces):
        return HilbertProductSpace(*spaces)
    elif all(isinstance(space, NormedSpace) for space in spaces):
        return NormedProductSpace(*spaces)
    else:
        return ProductSpace(*spaces)

def makePowerSpace(underlying_space, nProducts):
    """ Creates a Cartesian "power" of a space. For example,

    `PowerSpace(Reals(),3)` is mathematically the same as `RN(3)` or `R x R x R`
    Note that the later is more efficient.

    Selects the type of product space that has the most structure (Inner product, norm)
    given a set of spaces

    Parameters
    ----------

    underlying_space : LinearSpace     
                       The underlying space that should be repeated
    nProduct         : Int
                       Number of products in the result

    Returns
    -------
    ProductSpace, NormedProductSpace or HilbertProductSpace depending on the lowest common denominator in spaces.

    """
    return makeProductSpace(*([underlying_space]*nProducts))