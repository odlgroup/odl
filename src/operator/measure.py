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


class Measure(object):
    """ A measure on some set
    """

    __metaclass__ = ABCMeta #Set as abstract

    def __call__(self,set):
        return self.measure(set)

    @abstractmethod
    def measure(self,set):
        """Calculate the measure of set
        """


class MeasurableSets(object):
    """ Some measurable sets, subsets of some set
    """

    __metaclass__ = ABCMeta #Set as abstract


class Discretization(MeasurableSets):
    """ A discretization of some measurable sets
    """
    __metaclass__ = ABCMeta #Set as abstract

    @abstractmethod
    def __iter__(self):
        """Discrete spaces can be iterated over
        """

class MeasureSpace(object):
    """ A space where integration is defined
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

    def midpoint(self):
        return (self.end+self.begin)/2.0


class BorelMeasure(Measure):
    def measure(self, interval):
        return interval.end-interval.begin


class UniformDiscretization(Discretization):
    def __init__(self,interval,n):
        self.interval = interval
        self.n = n

    def __iter__(self):
        class intervalIter():
            def __init__(self,begin,step,n):
                self.cur = begin
                self.step = step
                self.n = n

            def next(self):
                if self.n>0:
                    i = Interval(self.cur,self.cur+self.step)
                    self.cur += self.step
                    self.n -= 1
                    return i
                else:
                    raise StopIteration()

        begin = self.interval.begin
        step = (self.interval.end - self.interval.begin)/self.n
        return intervalIter(begin,step,self.n)


class DiscreteMeaureSpace(MeasureSpace):
    def __init__(self,discretization,measure):
        self.discretization = discretization
        self.measure = measure

    def integrate(self,f):
        return sum(f.apply(inter.midpoint()) * self.measure(inter) for inter in self.discretization)