from itertools import chain
from collections import Iterable
import unittest

#Todo move
class RLTestCase(unittest.TestCase):
    def assertAlmostEqual(self,f1,f2):
        unittest.TestCase.assertAlmostEqual(self,float(f1),float(f2))

    def assertAllAlmostEquals(self,iter1,iter2):
        for [i1,i2] in zip(iter1,iter2):
            if isinstance(i1, Iterable):
                self.assertAllAlmostEquals(i1,i2)
            else:
                self.assertAlmostEqual(i1,i2)