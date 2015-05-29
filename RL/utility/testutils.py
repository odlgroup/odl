# Copyright 2014, 2015 Holger Kohr, Jonas Adler
#
# This file is part of RL.
#
# RL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RL.  If not, see <http://www.gnu.org/licenses/>.

"""
Utilities for use inside the RL project, not for external use.
"""

# Imports for common Python 2/3 codebase

from __future__ import (print_function, unicode_literals, division,
                        absolute_import)
from builtins import object, super
from future import standard_library

# External module imports
from itertools import zip_longest
import unittest
from time import time

standard_library.install_aliases()


# Todo move
class RLTestCase(unittest.TestCase):
    def assertAlmostEqual(self, f1, f2, *args, **kwargs):
        unittest.TestCase.assertAlmostEqual(self, float(f1), float(f2), *args,
                                            **kwargs)

    def assertAllAlmostEquals(self, iter1, iter2, *args, **kwargs):
        """ Assert thaat all elements in iter1 and iter2 are almost equal.

        The iterators may be nestled lists or warying types

        assertAllAlmostEquals([[1,2],[3,4]],np.array([[1,2],[3,4]]) == True
        """
        # Sentinel object used to check that both iterators are the same length
        differentLengthSentinel = object()

        if iter1 is None and iter2 is None:
            return

        for [i1, i2] in zip_longest(iter1, iter2,
                                    fillvalue=differentLengthSentinel):
            # Verify that none of the lists has ended (then they are not the
            # same size)
            self.assertIsNot(i1, differentLengthSentinel)
            self.assertIsNot(i2, differentLengthSentinel)
            try:
                self.assertAllAlmostEquals(iter(i1), iter(i2), *args, **kwargs)
            except TypeError:
                self.assertAlmostEquals(float(i1), float(i2), *args, **kwargs)


def skip_all_tests(reason=None):
    """ Creates a TestCase replacement class where all tests are skipped
    """
    if reason is None:
        reason = ''

    class SkipAllTestsMeta(type):
        def __new__(mcs, name, bases, local):
            for attr in local:
                value = local[attr]
                if attr.startswith('test') and callable(value):
                    local[attr] = unittest.skip(reason)(value)
            return super().__new__(mcs, name, bases, local)

    class SkipAllTestCase(unittest.TestCase):
        __metaclass__ = SkipAllTestsMeta

    return SkipAllTestCase


class Timer(object):
    """ A timer to be used as:

    with Timer("name"):
        Do stuff

    Prints the time stuff took to execute.
    """
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[{}] '.format(self.name))
        print('Elapsed: {}'.format(time() - self.tstart))
