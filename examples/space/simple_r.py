"""An example of a very simple space, the real numbers."""

import odl


class Reals(odl.set.LinearSpace):
    """The real numbers."""

    def __init__(self):
        super(Reals, self).__init__(field=odl.RealNumbers())

    def _inner(self, x1, x2):
        return x1.__val__ * x2.__val__

    def _lincomb(self, a, x1, b, x2, out):
        out.__val__ = a * x1.__val__ + b * x2.__val__

    def _multiply(self, x1, x2, out):
        out.__val__ = x1.__val__ * x2.__val__

    def __eq__(self, other):
        return isinstance(other, Reals)

    def element(self, value=0):
        return RealNumber(self, value)


class RealNumber(odl.set.space.LinearSpaceElement):
    """Real vectors are floats."""

    __val__ = None

    def __init__(self, space, v):
        super(RealNumber, self).__init__(space)
        self.__val__ = v

    def __float__(self):
        return self.__val__.__float__()

    def __str__(self):
        return str(self.__val__)


R = Reals()
x = R.element(5.0)
y = R.element(10.0)

print(x)
print(y)
print(x + y)
print(x * y)
print(x - y)
print(3.14 * x)
