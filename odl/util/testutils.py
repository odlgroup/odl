# Copyright 2014-2016 The ODL development group
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
from future import standard_library
standard_library.install_aliases()
from builtins import int, object

from itertools import zip_longest
import numpy as np
import sys
import os
from time import time


__all__ = ('almost_equal', 'all_equal', 'all_almost_equal', 'never_skip',
           'skip_if_no_stir', 'skip_if_no_pywavelets',
           'skip_if_no_pyfftw', 'skip_if_no_largescale',
           'noise_array', 'noise_element', 'noise_elements',
           'Timer', 'timeit', 'ProgressBar', 'ProgressRange',
           'test', 'run_doctests')


def _places(a, b, default=None):
    """Return number of expected correct digits between ``a`` and ``b``.

    Returned numbers if one of ``a.dtype`` and ``b.dtype`` is as below:
        2 -- for ``np.float16``

        3 -- for ``np.float32`` or ``np.complex64``

        5 -- for all other cases
    """
    dtype1 = getattr(a, 'dtype', object)
    dtype2 = getattr(b, 'dtype', object)
    return min(dtype_places(dtype1, default), dtype_places(dtype2, default))


def dtype_places(dtype, default=None):
    """Return number of correct digits expected for given dtype.

    Returned numbers:
        1 -- for ``np.float16``

        3 -- for ``np.float32`` or ``np.complex64``

        5 -- for all other cases
    """
    small_dtypes = [np.float32, np.complex64]
    tiny_dtypes = [np.float16]

    if dtype in tiny_dtypes:
        return 1
    elif dtype in small_dtypes:
        return 3
    else:
        return default if default is not None else 5


def almost_equal(a, b, places=None):
    """Return ``True`` if the scalars ``a`` and ``b`` are almost equal."""
    if a is None and b is None:
        return True

    if places is None:
        places = _places(a, b)

    eps = 10 ** -places

    try:
        complex(a)
        complex(b)
    except TypeError:
        return False

    if np.isnan(a) and np.isnan(b):
        return True

    if np.isinf(a) and np.isinf(b):
        return a == b

    if abs(complex(b)) < eps:
        return abs(complex(a) - complex(b)) < eps
    else:
        return abs(a / b - 1) < eps


def all_equal(iter1, iter2):
    """Return ``True`` if all elements in ``a`` and ``b`` are equal."""
    # Direct comparison for scalars, tuples or lists
    try:
        if iter1 == iter2:
            return True
    except ValueError:  # Raised by NumPy when comparing arrays
        pass

    # Special case for None
    if iter1 is None and iter2 is None:
        return True

    # If one nested iterator is exhausted, go to direct comparison
    try:
        it1 = iter(iter1)
        it2 = iter(iter2)
    except TypeError:
        try:
            return iter1 == iter2
        except ValueError:  # Raised by NumPy when comparing arrays
            return False

    diff_length_sentinel = object()

    # Compare element by element and return False if the sequences have
    # different lengths
    for [ip1, ip2] in zip_longest(it1, it2,
                                  fillvalue=diff_length_sentinel):
        # Verify that none of the lists has ended (then they are not the
        # same size)
        if ip1 is diff_length_sentinel or ip2 is diff_length_sentinel:
            return False

        if not all_equal(ip1, ip2):
            return False

    return True


def all_almost_equal_array(v1, v2, places):
    # Ravel if has order, only DiscreteLpElement has an order
    if hasattr(v1, 'order'):
        v1 = v1.__array__().ravel(v1.order)
    else:
        v1 = v1.__array__()

    if hasattr(v2, 'order'):
        v2 = v2.__array__().ravel(v2.order)
    else:
        v2 = v2.__array__()

    return np.all(np.isclose(v1, v2,
                             rtol=10 ** (-places), atol=10 ** (-places),
                             equal_nan=True))


def all_almost_equal(iter1, iter2, places=None):
    """Return ``True`` if all elements in ``a`` and ``b`` are almost equal."""
    try:
        if iter1 is iter2 or iter1 == iter2:
            return True
    except ValueError:
        pass

    if iter1 is None and iter2 is None:
        return True

    if places is None:
        places = _places(iter1, iter2, None)

    if hasattr(iter1, '__array__') and hasattr(iter2, '__array__'):
        return all_almost_equal_array(iter1, iter2, places)

    try:
        it1 = iter(iter1)
        it2 = iter(iter2)
    except TypeError:
        return almost_equal(iter1, iter2, places)

    diff_length_sentinel = object()
    for [ip1, ip2] in zip_longest(it1, it2,
                                  fillvalue=diff_length_sentinel):
        # Verify that none of the lists has ended (then they are not the
        # same size)
        if ip1 is diff_length_sentinel or ip2 is diff_length_sentinel:
            return False

        if not all_almost_equal(ip1, ip2, places):
            return False

    return True


def is_subdict(subdict, dictionary):
    """Return ``True`` if all items of ``subdict`` are in ``dictionary``."""
    return all(item in dictionary.items() for item in subdict.items())


try:
    # Try catch in case user does not have pytest
    import pytest

    # Used in lists where the elements should all be skipifs
    never_skip = pytest.mark.skipif(
        "False",
        reason='Fill in, never skips'
    )

    skip_if_no_stir = pytest.mark.skipif(
        "not odl.tomo.backends.stir_bindings.STIR_AVAILABLE",
        reason='STIR not available'
    )

    skip_if_no_pywavelets = pytest.mark.skipif(
        "not odl.trafos.PYWT_AVAILABLE",
        reason='PyWavelets not available'
    )

    skip_if_no_pyfftw = pytest.mark.skipif(
        "not odl.trafos.PYFFTW_AVAILABLE",
        reason='pyFFTW not available')

    skip_if_no_largescale = pytest.mark.skipif(
        "not pytest.config.getoption('--largescale')",
        reason='Need --largescale option to run'
    )

    skip_if_no_benchmark = pytest.mark.skipif(
        "not pytest.config.getoption('--benchmark')",
        reason='Need --benchmark option to run'
    )

except ImportError:
    def _pass(function):
        """Trivial decorator used if pytest marks are not available."""
        return function

    never_skip = _pass
    skip_if_no_stir = _pass
    skip_if_no_pywavelets = _pass
    skip_if_no_pyfftw = _pass
    skip_if_no_largescale = _pass
    skip_if_no_benchmark = _pass


# Helpers to generate data
def noise_array(space):
    """Generate a white noise array that is compatible with ``space``.

    The array contains white noise with standard deviation 1 in the case of
    floating point dtypes and uniformly spaced values between -10 and 10 in
    the case of integer dtypes.

    For product spaces the method is called recursively for all sub-spaces.

    Notes
    -----
    This method is intended for internal testing purposes. For more explicit
    example elements see ``odl.phantoms`` and ``LinearSpaceElement.examples``.

    Parameters
    ----------
    space : `LinearSpace`
        Space from which to derive the array data type and size.

    Returns
    -------
    noise_array : `numpy.ndarray` element
        Array with white noise such that ``space.element``'s can be created
        from it.

    See Also
    --------
    noise_element
    noise_elements
    odl.set.space.LinearSpace.examples : Examples of elements
        typical to the space.
    """
    from odl.space import ProductSpace
    if isinstance(space, ProductSpace):
        return np.array([noise_array(si) for si in space])
    else:
        # Generate numpy space elements, real or complex or int
        if np.issubdtype(space.dtype, np.floating):
            arr = np.random.randn(space.size)
        elif np.issubdtype(space.dtype, np.integer):
            arr = np.random.randint(-10, 10, space.size)
        else:
            arr = (np.random.randn(space.size) +
                   1j * np.random.randn(space.size)) / np.sqrt(2.0)

        return arr.astype(space.dtype, copy=False)


def noise_element(space):
    """Create a white noise element in ``space``.

    The element contains white noise with standard deviation 1 in the case of
    floating point dtypes and uniformly spaced values between -10 and 10 in
    the case of integer dtypes.

    For product spaces the method is called recursively for all sub-spaces.

    Notes
    -----
    This method is intended for internal testing purposes. For more explicit
    example elements see ``odl.phantoms`` and ``LinearSpaceElement.examples``.

    Parameters
    ----------
    space : `LinearSpace`
        Space in which to create an element. The
        `odl.set.space.LinearSpace.element` method of the space needs to
        accept input of `numpy.ndarray` type.

    Returns
    -------
    noise_element : ``space`` element

    See Also
    --------
    noise_array
    noise_elements
    odl.set.space.LinearSpace.examples : Examples of elements typical
        to the space.
    """
    return space.element(noise_array(space))


def noise_elements(space, n=1):
    """Create a list of ``n`` noise arrays and elements in ``space``.

    The arrays contain white noise with standard deviation 1 in the case of
    floating point dtypes and uniformly spaced values between -10 and 10 in
    the case of integer dtypes.

    The returned elements wrap the arrays.

    For product spaces the method is called recursively for all sub-spaces.

    Notes
    -----
    This method is intended for internal testing purposes. For more explicit
    example elements see ``odl.phantoms`` and ``LinearSpaceElement.examples``.

    Parameters
    ----------
    space : `LinearSpace`
        Space in which to create an element. The
        `odl.set.space.LinearSpace.element` method of the space needs to
        accept input of `numpy.ndarray` type.
    n : int
        Number of elements to create.

    Returns
    -------
    arrays : `numpy.ndarray` or tuple of `numpy.ndarray`
        A single array if ``n == 1``, otherwise a tuple of arrays.
    elements : ``space`` element or tuple of ``space`` elements
        A single element if ``n == 1``, otherwise a tuple of elements.

    See Also
    --------
    noise_array
    noise_element
    """
    arrs = tuple(noise_array(space) for _ in range(n))

    # Make space elements from arrays
    elems = tuple(space.element(arr.copy()) for arr in arrs)

    if n == 1:
        return tuple(arrs + elems)
    else:
        return arrs, elems


class FailCounter(object):

    """Used to count the number of failures of something

    Usage::

        with FailCounter() as counter:
            # Do stuff

            counter.fail()

    When done, it prints

    ``*** FAILED 1 TEST CASE(S) ***``
    """

    def __init__(self, test_name, err_msg=None, logger=print):
        self.num_failed = 0
        self.test_name = test_name
        self.err_msg = err_msg
        self.fail_strings = []
        self.log = logger

    def __enter__(self):
        return self

    def fail(self, string=None):
        """Add failure with reason as string."""
        self.num_failed += 1

        # TODO: possibly limit number of printed strings
        if string is not None:
            self.fail_strings += [str(string)]

    def __exit__(self, type, value, traceback):
        if self.num_failed == 0:
            self.log('{:<70}: Completed all test cases.'
                     ''.format(self.test_name))
        else:
            print(self.test_name)

            for fail_string in self.fail_strings:
                print(fail_string)

            if self.err_msg is not None:
                print(self.err_msg)
            print('*** FAILED {} TEST CASE(S) ***'.format(self.num_failed))


class Timer(object):

    """A timer context manager.

    Usage::

        with Timer('name'):
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

    Usage::

        @timeit
        def myfunction(...):
            ...

        @timeit('info string')
        def myfunction(...):
            ...
    """

    if callable(arg):
        def timed_function(*args, **kwargs):
            with Timer(str(arg)):
                return arg(*args, **kwargs)
        return timed_function
    else:
        def _timeit_helper(func):
            def timed_function(*args, **kwargs):
                with Timer(arg):
                    return func(*args, **kwargs)
            return timed_function
        return _timeit_helper


class ProgressBar(object):

    """A simple command-line progress bar.

    Usage:

    >>> progress = ProgressBar('Reading data', 10)
    \rReading data: [                              ] Starting
    >>> progress.update(4) #halfway, zero indexing
    \rReading data: [###############               ] 50.0%

    Multi-indices, from slowest to fastest:

    >>> progress = ProgressBar('Reading data', 10, 10)
    \rReading data: [                              ] Starting
    >>> progress.update(9, 8)
    \rReading data: [############################# ] 99.0%

    Supports simply calling update, which moves the counter forward:

    >>> progress = ProgressBar('Reading data', 10, 10)
    \rReading data: [                              ] Starting
    >>> progress.update()
    \rReading data: [                              ]  1.0%
    """

    def __init__(self, text='progress', *njobs):
        """Initialize a new instance."""
        self.text = str(text)
        if len(njobs) == 0:
            raise ValueError('need to provide at least one job')
        self.njobs = njobs
        self.current_progress = 0.0
        self.index = 0
        self.done = False
        self.start()

    def start(self):
        """Print the initial bar."""
        sys.stdout.write('\r{0}: [{1:30s}] Starting'.format(self.text,
                                                            ' ' * 30))

        sys.stdout.flush()

    def update(self, *indices):
        """Update the bar according to ``indices``."""
        if indices:
            if len(indices) != len(self.njobs):
                raise ValueError('number of indices not correct')
            self.index = np.ravel_multi_index(indices, self.njobs) + 1
        else:
            self.index += 1

        # Find progress as ratio between 0 and 1
        # offset by 1 for zero indexing
        progress = self.index / np.prod(self.njobs)

        # Write a progressbar and percent
        if progress < 1.0:
            # Only update on 0.1% intervals
            if progress > self.current_progress + 0.001:
                sys.stdout.write('\r{0}: [{1:30s}] {2:4.1f}%   '.format(
                    self.text, '#' * int(30 * progress), 100 * progress))
                self.current_progress = progress
        else:  # Special message when done
            if not self.done:
                sys.stdout.write('\r{0}: [{1:30s}] Done      \n'.format(
                    self.text, '#' * 30))
                self.done = True

        sys.stdout.flush()


class ProgressRange(object):

    """Simple range sequence with progress bar output"""

    def __init__(self, text, n):
        """Initialize a new instance."""
        self.current = 0
        self.n = n
        self.bar = ProgressBar(text, n)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.n:
            val = self.current
            self.current += 1
            self.bar.update()
            return val
        else:
            raise StopIteration()


def test(arguments=None):
    """Run ODL tests given by arguments."""
    try:
        import pytest
    except ImportError:
        raise ImportError('ODL tests cannot be run without `pytest` installed.'
                          '\nRun `$ pip install [--user] odl[testing]` in '
                          'order to install `pytest`.')

    from .pytest_plugins import collect_ignore

    this_dir = os.path.dirname(__file__)
    odl_root = os.path.abspath(os.path.join(this_dir, os.pardir, os.pardir))

    args = ['-x', '{root}/odl'.format(root=odl_root)]

    ignores = ['--ignore={}'.format(file) for file in collect_ignore]
    args.extend(ignores)

    if arguments is not None:
        args.extend(arguments)

    pytest.main(args)


def run_doctests(skip_if=False):
    """Run all doctests in the current module.

    For ``skip_if=True``, the tests in the module are skipped.
    """
    from doctest import testmod, NORMALIZE_WHITESPACE, SKIP
    optionflags = NORMALIZE_WHITESPACE
    if skip_if:
        optionflags |= SKIP
    testmod(optionflags=optionflags)


if __name__ == '__main__':
    run_doctests()
