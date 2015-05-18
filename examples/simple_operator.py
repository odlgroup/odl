from __future__ import unicode_literals, print_function, division
from __future__ import absolute_import
from future.builtins import object, zip
from future import standard_library
standard_library.install_aliases()

from RL.space.euclidean import RN
from RL.operator.operator import Operator

""" An example of a very simple operator on RN
"""

class addOp(Operator):
    def __init__(self, n, x):
        self.n = n
        self.x = x
        self.range = RN(n)
        self.domain = RN(n)

    def _apply(self, rhs, out):
        out.data[:] = rhs.data[:] + self.x

n = 3
rn = RN(n)
x = rn.element([1, 2, 3])

op = addOp(n, 10)

print(op(x))