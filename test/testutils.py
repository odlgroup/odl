from itertools import chain
from collections import Iterable
import unittest

#Todo move
class RLTestCase(unittest.TestCase):
    def assertAlmostEqual(self,f1,f2,*args,**kwargs):
        unittest.TestCase.assertAlmostEqual(self,float(f1),float(f2),*args,**kwargs)

    def assertAllAlmostEquals(self,iter1,iter2,*args,**kwargs):
        for [i1,i2] in zip(iter1,iter2):
            if isinstance(i1, Iterable):
                self.assertAllAlmostEquals(i1,i2,*args,**kwargs)
            else:
                self.assertAlmostEqual(i1,i2,*args,**kwargs)