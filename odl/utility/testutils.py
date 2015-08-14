# Copyright 2014, 2015 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""
Utilities for use inside the ODL project, not for external use.
"""

# Imports for common Python 2/3 codebase

from __future__ import (print_function, unicode_literals, division,
                        absolute_import)
from builtins import object, super
from future import standard_library
standard_library.install_aliases()

# External module imports
# pylint: disable=no-name-in-module
from itertools import zip_longest
import unittest
from time import time
from future.utils import with_metaclass


# TODO: move
class ODLTestCase(unittest.TestCase):
    # Use names compatible with unittest
    # pylint: disable=invalid-name
    def assertAlmostEqual(self, f1, f2, *args, **kwargs):
        unittest.TestCase.assertAlmostEqual(self, float(f1), float(f2), *args,
                                            **kwargs)

    # pylint: disable=invalid-name
    def assertAllAlmostEquals(self, iter1, iter2, *args, **kwargs):
        """ Assert thaat all elements in iter1 and iter2 are almost equal.

        The iterators may be nested lists or varying types

        assertAllAlmostEquals([[1,2],[3,4]],np.array([[1,2],[3,4]]) == True
        """
        # Sentinel object used to check that both iterators are the same length
        different_length_sentinel = object()

        if iter1 is None and iter2 is None:
            return

        for [ip1, ip2] in zip_longest(iter1, iter2,
                                      fillvalue=different_length_sentinel):
            # Verify that none of the lists has ended (then they are not the
            # same size)
            self.assertIsNot(ip1, different_length_sentinel)
            self.assertIsNot(ip2, different_length_sentinel)
            try:
                self.assertAllAlmostEquals(iter(ip1), iter(ip2), *args,
                                           **kwargs)
            except TypeError:
                self.assertAlmostEqual(ip1, ip2, *args, **kwargs)

    def assertAllEquals(self, iter1, iter2, *args, **kwargs):
        """ Assert thaat all elements in iter1 and iter2 are equal.

        The iterators may be nested lists or varying types
        """
        kwargs['delta'] = 0
        self.assertAllAlmostEquals(iter1, iter2, *args, **kwargs)


def skip_all(reason=None):
    """ Create a TestCase replacement class where all tests are skipped
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

    class SkipAllTestCase(with_metaclass(SkipAllTestsMeta, unittest.TestCase)):
        pass

    return SkipAllTestCase


class Timer(object):
    """ A timer to be used as:

    with Timer("name"):
        Do stuff

    Prints the time stuff took to execute.
    """
    def __init__(self, name=None):
        self.name = name
        self.tstart = None

    def __enter__(self):
        self.tstart = time()

    def __exit__(self, type, value, traceback):
        if self.name is not None:
            print('[{}] '.format(self.name))
        print('Elapsed: {}'.format(time() - self.tstart))
