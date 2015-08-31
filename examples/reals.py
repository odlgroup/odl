from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from future.builtins import object, zip
from future import standard_library
standard_library.install_aliases()

from odl.space.space import LinearSpace

""" An example of a very simple space, the space Rn
"""

class Reals(LinearSpace):
    """The real numbers
    """

    def __init__(self):
        self._field = RealNumbers()

    def _inner(self, x, y):
        return x.__val__ * y.__val__

    def _lincomb(self, z, a, x, b, y):
        z.__val__ = a*x.__val__ + b*y.__val__

    def _multiply(self, x, y):
        y.__val__ *= x.__val__

    @property
    def field(self):
        return self._field

    def equals(self, other):
        return isinstance(other, Reals)

    def element(self, value=0):
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