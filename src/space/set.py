from __future__ import unicode_literals, print_function, division
from __future__ import absolute_import
from future.builtins import object
from future import standard_library
standard_library.install_aliases()
from abc import ABCMeta, abstractmethod, abstractproperty
from numbers import Integral, Real, Complex

class AbstractSet(object):
    """ An arbitrary set
    """

    __metaclass__ = ABCMeta #Set as abstract

    @abstractmethod
    def equals(self, other):
        """ Test two sets for equality
        """

    @abstractmethod
    def isMember(self, other):
        """ Test if other is a member of self
        """

    #Implicitly default implemented methods
    def __eq__(self, other):         
        return self.equals(other)

    def __ne__(self, other):         
        return not self.equals(other)

class EmptySet(AbstractSet):
    """ The empty set has no members (None is considered "no element")
    """
    def equals(self, other):
        return isinstance(other, EmptySet)

    def isMember(self, other):
        return other is None

class ComplexNumbers(AbstractSet):
    """ The set of complex numbers
    """

    def equals(self, other):
        return isinstance(other, ComplexNumbers)

    def isMember(self, other):
        return isinstance(other, Complex)


class RealNumbers(ComplexNumbers):
    """ The set of real numbers
    """

    def equals(self, other):
        return isinstance(other, RealNumbers)

    def isMember(self, other):
        return isinstance(other, Real)

class Integers(RealNumbers):
    """ The set of all non-negative integers
    """

    def equals(self, other):
        return isinstance(other, Integers)

    def isMember(self, other):
        return isinstance(other, Integral)


class Interval(RealNumbers):
    """ The set of real numbers in the interval [begin,end]
    """

    def __init__(self, begin, end):
        self.begin = begin
        self.end = end

    def equals(self, other):
        return isinstance(other, Interval) and self.begin == other.begin and self.end == other.end

    def isMember(self, other):
        return RealNumbers.isMember(self, other) and self.begin <= other <= self.end

class Square(AbstractSet):

    def __init__(self, begin, end):
        self.begin = begin
        self.end = end

    def equals(self, other):
        return isinstance(other, Square) and self.begin == other.begin and self.end == other.end

    def isMember(self, other):
        return RealNumbers.isMember(self, other[0])  and self.begin[0] <= other[0] <= self.end[0] and \
               RealNumbers.isMember(self, other[1])  and self.begin[1] <= other[1] <= self.end[1]