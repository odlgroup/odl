from __future__ import unicode_literals, print_function, division
from __future__ import absolute_import
from future.builtins import object, zip
from future import standard_library
standard_library.install_aliases()

from RL.space.space import HilbertSpace, Algebra

""" An example of a very simple space, the space RN
"""

class Reals(HilbertSpace, Algebra):
    """The real numbers
    """

    def __init__(self):
        self._field = RealNumbers()

    def innerImpl(self, x, y):        
        return x.__val__ * y.__val__

    def linCombImpl(self, z, a, x, b, y):        
        z.__val__ = a*x.__val__ + b*y.__val__

    def multiplyImpl(self, x, y):
        y.__val__ *= x.__val__

    def empty(self):
        return self.makeVector(0.0)

    @property
    def field(self):
        return self._field
    
    def equals(self, other):
        return isinstance(other, Reals)

    def makeVector(self, value):
        return Reals.Vector(self, value)

    class Vector(HilbertSpace.Vector):
        """Real vectors are floats
        """

        __val__ = None
        def __init__(self, space, v):
            HilbertSpace.Vector.__init__(self, space)
            self.__val__ = v

        def __float__(self):        
            return self.__val__.__float__()