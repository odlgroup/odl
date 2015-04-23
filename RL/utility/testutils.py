from itertools import izip_longest
import unittest
from time import time


#Todo move
class RLTestCase(unittest.TestCase):
    def assertAlmostEqual(self, f1, f2, *args, **kwargs):
        unittest.TestCase.assertAlmostEqual(self, float(f1), float(f2), *args, **kwargs)

    def assertAllAlmostEquals(self, iter1, iter2, *args, **kwargs):
        differentLengthSentinel = object() #Sentinel object used to check that both iterators are the same length.

        if iter1 is None and iter2 is None:
            return

        for [i1, i2] in izip_longest(iter1, iter2, fillvalue=differentLengthSentinel):
            #Verify that none of the lists has ended (then they are not the same size
            self.assertIsNot(i1, differentLengthSentinel)
            self.assertIsNot(i2, differentLengthSentinel)
            try:
                self.assertAllAlmostEquals(iter(i1), iter(i2), *args, **kwargs)
            except TypeError:
                self.assertAlmostEquals(float(i1), float(i2), *args, **kwargs)

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time() - self.tstart)


def consume(iterator):
    """ Consumes an iterator and returns the last value
    """
    for x in iterator:
        pass
    return x