from itertools import chain
from collections import Iterable
import unittest

#Todo move
class RLTestCase(unittest.TestCase):
    def assertAlmostEqual(self,f1,f2,*args,**kwargs):
        unittest.TestCase.assertAlmostEqual(self,float(f1),float(f2),*args,**kwargs)

    def assertAllAlmostEquals(self,iter1,iter2,*args,**kwargs):
        for [i1,i2] in zip(iter1,iter2):
            try:
                self.assertAllAlmostEquals(iter(i1),iter(i2),*args,**kwargs)
            except TypeError:
                self.assertAlmostEquals(float(i1),float(i2),*args,**kwargs)