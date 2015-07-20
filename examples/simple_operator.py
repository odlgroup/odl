from __future__ import unicode_literals, print_function, division
from __future__ import absolute_import
from future.builtins import object, zip
from future import standard_library
standard_library.install_aliases()

from RL.space.cartesian import Rn
from RL.operator.operator import Operator

""" An example of a very simple operator on Rn
"""

class addOp(Operator):
    def __init__(self, n, x):
        self.x = x
        self.range = Rn(n)
        self.domain = Rn(n)

    def _apply(self, rhs, out):
        out.data[:] = rhs.data[:] + self.x

n = 3
rn = Rn(n)
x = rn.element([1, 2, 3])

op = addOp(n, 10)

print(op(x))