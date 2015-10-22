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

"""Utilities for internal use."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from builtins import object
import pytest
from future import standard_library
standard_library.install_aliases()

# External module imports
# pylint: disable=no-name-in-module
from numpy import ravel_multi_index, prod
from itertools import zip_longest
import sys
from time import time

__all__ = ('almost_equal', 'all_equal', 'all_almost_equal', 'skip_if_no_cuda', 
           'Timer', 'timeit', 'ProgressBar', 'ProgressRange')

def almost_equal(a, b, places=7):    
    if a is None and b is None:
        return True
        
    try:
        a = complex(a)
        b = complex(b) 
    except TypeError:
        return False
        
    if abs(a) == float('inf') and abs(b) == float('inf'):
        return a == b
        
    if round(abs(complex(a) - complex(b)), places) == 0:
        return True
    else:
        print(complex(a), complex(b))
        return False

def all_equal(iter1, iter2):
    # Sentinel object used to check that both iterators are the same length
    different_length_sentinel = object()

    if iter1 is None and iter2 is None:
        return True

    try:
        i1 = iter(iter1)
        i2 = iter(iter2)
    except TypeError:
        return iter1 == iter2

    for [ip1, ip2] in zip_longest(i1, i2,
                                  fillvalue=different_length_sentinel):
        # Verify that none of the lists has ended (then they are not the
        # same size)
        if ip1 is different_length_sentinel or ip2 is different_length_sentinel:
            return False
            
        if not all_equal(ip1, ip2):
            return False
    
    return True

def all_almost_equal(iter1, iter2, places=7):
    # Sentinel object used to check that both iterators are the same length
    different_length_sentinel = object()

    if iter1 is None and iter2 is None:
        return True

    try:
        i1 = iter(iter1)
        i2 = iter(iter2)
    except TypeError:
        return almost_equal(iter1, iter2, places)

    for [ip1, ip2] in zip_longest(iter1, iter2,
                                  fillvalue=different_length_sentinel):
        # Verify that none of the lists has ended (then they are not the
        # same size)
        if ip1 is different_length_sentinel or ip2 is different_length_sentinel:
            return False
            
        if not all_almost_equal(ip1, ip2, places):
            return False
    
    return True
    
    
skip_if_no_cuda = pytest.mark.skipif("not odl.CUDA_AVAILABLE", reason='CUDA not available')

class FailCounter(object):
    """ Used to count the number of failures of something

    Usage
    -----

    with FailCounter() as counter:
        # Do stuff
    
        counter.fail()
        
    #when done
        
    *** FAILED 1 TEST CASE(S) ***
    
        

    Prints the time stuff took to execute.
    """

    def __init__(self, err_msg=None):
        self.num_failed = 0
        self.err_msg = err_msg

    def __enter__(self):
        return self
        
    def fail(self, string=None):
        self.num_failed += 1
        
        #Todo: possibly limit number of printed strings
        if string is not None:
            print(string)

    def __exit__(self, type, value, traceback):
        if self.num_failed == 0:
            print('Completed all test cases')
        else:
            if self.err_msg is not None:
                print(self.err_msg)
            print('*** FAILED {} TEST CASE(S) ***'.format(self.num_failed))

class Timer(object):

    """A timer context manager.

    Usage
    -----

    with Timer("name"):
        # Do stuff

    Prints the time stuff took to execute.
    """

    def __init__(self, name=None):
        if name is not None:
            self.name = name
        else:
            self.name = 'Elapsed'
        self.tstart = None

    def __enter__(self):
        self.tstart = time()

    def __exit__(self, type, value, traceback):
        time_str = '{:.3f}'.format(time() - self.tstart)
        print('{:>30s} : {:>10s} '.format(self.name, time_str))


def timeit(arg):

    """A timer decorator.

    Usage
    -----

    @timeit
    def myfunction(...):
        ...

    @timeit("info string")
    def myfunction(...):
        ...
    """

    if callable(arg):
        def timed_function(*args, **kwargs):
            with Timer(str(arg)):
                return arg(*args, **kwargs)
        return timed_function
    else:
        def _timeit_helper(fn):
            def timed_function(*args, **kwargs):
                with Timer(arg):
                    return fn(*args, **kwargs)
            return timed_function
        return _timeit_helper


class ProgressBar(object):

    """A simple command-line progress bar.

    Usage
    -----

    >>> progress = ProgressBar('Reading data', 10)
    \rReading data: [                              ] Starting
    >>> progress.update(4) #halfway, zero indexing
    \rReading data: [###############               ] 50.0%   

    Also supports multiple index, from slowest varying to fastest

    >>> progress = ProgressBar('Reading data', 10, 10)
    \rReading data: [                              ] Starting
    >>> progress.update(9, 8)
    \rReading data: [############################# ] 99.0%   

    Also supports simply calling update, which moves the counter forward

    >>> progress = ProgressBar('Reading data', 10, 10)
    \rReading data: [                              ] Starting
    >>> progress.update()
    \rReading data: [                              ]  1.0%   
    """

    def __init__(self, text='progress', *njobs):
        self.text = str(text)
        if len(njobs) == 0:
            raise ValueError('Need to provide at least one job len')
        self.njobs = njobs
        self.current_progress = 0.0
        self.index = 0
        self.done = False
        self.start()

    def start(self):
        sys.stdout.write('\r{0}: [{1:30s}] Starting'.format(self.text,
                                                            ' '*30))

        sys.stdout.flush()

    def update(self, *indices):
        if indices:
            if len(indices) != len(self.njobs):
                raise ValueError('number of indices not correct')
            self.index = ravel_multi_index(indices, self.njobs) + 1
        else:
            self.index += 1

        # Find progress as ratio between 0 and 1
        # offset by 1 for zero indexing
        progress = self.index / prod(self.njobs)

        # Write a progressbar and percent
        if progress < 1.0:
            # Only update on 0.1% intervals
            if progress > self.current_progress+0.001:
                sys.stdout.write('\r{0}: [{1:30s}] {2:4.1f}%   '.format(
                    self.text, '#'*int(30*progress), 100*progress))
                self.current_progress = progress
        else:  # Special message when done
            if not self.done:
                sys.stdout.write('\r{0}: [{1:30s}] Done      \n'.format(
                    self.text, '#'*30))
                self.done = True

        sys.stdout.flush()


class ProgressRange(object):
    def __init__(self, text, n):
        self.current = 0
        self.n = n
        self.bar = ProgressBar(text, n)

    def __iter__(self):
        return self

    def next(self):
        if self.current < self.n:
            val = self.current
            self.current += 1
            self.bar.update()
            return val
        else:
            raise StopIteration()
