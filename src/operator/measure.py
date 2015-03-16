from __future__ import unicode_literals, print_function, division
from __future__ import absolute_import
from future.builtins import object
from future import standard_library
standard_library.install_aliases()
from abc import ABCMeta, abstractmethod, abstractproperty

class Set(object):
    """ An arbitrary set
    """

    __metaclass__ = ABCMeta #Set as abstract

    @abstractmethod
    def __eq__(self,other):
        """ Test two sets for equality
        """

class Discretization(object):
    """ A discretization of some set
    """

    __metaclass__ = ABCMeta #Set as abstract

    @abstractmethod
    def integrate(self,f):
        """Calculate the integral of f
        """

#Example implementation
class Interval(Set):
    def __init__(self,begin,end):
        self.begin = begin
        self.end = end

    def __eq__(self, other):
        return isinstance(other,Interval) and self.begin == other.begin and self.end == other.end #TODO float?