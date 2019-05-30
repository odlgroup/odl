# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Testing utilities."""

from __future__ import absolute_import, division, print_function
from future.moves.itertools import zip_longest

import os
import sys
import warnings
from builtins import object
from time import time

import numpy as np

from odl.util.utility import is_string, run_from_ipython


__all__ = (
    'dtype_ndigits',
    'dtype_tol',
    'all_equal',
    'all_almost_equal',
    'is_subdict',
    'skip_if_no_pyfftw',
    'skip_if_no_pywavelets',
    'skip_if_no_largescale',
    'skip_if_no_benchmark',
    'simple_fixture',
    'noise_array',
    'noise_element',
    'noise_elements',
    'FailCounter',
    'Timer',
    'timeit',
    'ProgressBar',
    'ProgressRange',
    'test',
    'run_doctests',
    'test_file',
)


def _ndigits(a, b, default=None):
    """Return number of expected correct digits comparing ``a`` and ``b``.

    The returned number is the minimum `dtype_ndigits` of the two objects.

    See Also
    --------
    dtype_ndigits
    """
    dtype1 = getattr(a, 'dtype', object)
    dtype2 = getattr(b, 'dtype', object)
    return min(dtype_ndigits(dtype1, default), dtype_ndigits(dtype2, default))


def dtype_ndigits(dtype, default=None):
    """Return the number of correct digits expected for a given dtype.

    This is intended as a somewhat generous default (relative) precision for
    results of more or less stable computations.

    Returned numbers:

    - ``np.float16``: ``1``
    - ``np.float32`` or ``np.complex64``: ``3``
    - Others: ``default`` if given, otherwise ``5``

    See Also
    --------
    dtype_tol : Same precision expressed as tolerance
    """
    small_dtypes = [np.float32, np.complex64]
    tiny_dtypes = [np.float16]

    if dtype in tiny_dtypes:
        return 1
    elif dtype in small_dtypes:
        return 3
    else:
        return default if default is not None else 5


def dtype_tol(dtype, default=None):
    """Return a tolerance for a given dtype.

    This is intended as a somewhat generous default (relative) tolerance for
    results of more or less stable computations.

    Returned numbers:

    - ``np.float16``: ``1e-1``
    - ``np.float32`` or ``np.complex64``: ``1e-3``
    - Others: ``default`` if given, otherwise ``1e-5``

    See Also
    --------
    dtype_ndigits : Same tolerance expressed in number of digits.
    """
    return 10 ** -dtype_ndigits(dtype, default)


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


def all_almost_equal_array(v1, v2, ndigits):
    return np.allclose(v1, v2,
                       rtol=10 ** -ndigits, atol=10 ** -ndigits,
                       equal_nan=True)


def all_almost_equal(iter1, iter2, ndigits=None):
    """Return ``True`` if all elements in ``a`` and ``b`` are almost equal."""
    try:
        if iter1 is iter2 or iter1 == iter2:
            return True
    except ValueError:
        pass

    if iter1 is None and iter2 is None:
        return True

    if hasattr(iter1, '__array__') and hasattr(iter2, '__array__'):
        # Only get default ndigits if comparing arrays, need to keep `None`
        # otherwise for recursive calls.
        if ndigits is None:
            ndigits = _ndigits(iter1, iter2, None)
        return all_almost_equal_array(iter1, iter2, ndigits)

    try:
        it1 = iter(iter1)
        it2 = iter(iter2)
    except TypeError:
        if ndigits is None:
            ndigits = _ndigits(iter1, iter2, None)
        return np.isclose(iter1, iter2,
                          atol=10 ** -ndigits, rtol=10 ** -ndigits,
                          equal_nan=True)

    diff_length_sentinel = object()
    for [ip1, ip2] in zip_longest(it1, it2,
                                  fillvalue=diff_length_sentinel):
        # Verify that none of the lists has ended (then they are not the
        # same size)
        if ip1 is diff_length_sentinel or ip2 is diff_length_sentinel:
            return False

        if not all_almost_equal(ip1, ip2, ndigits):
            return False

    return True


def is_subdict(subdict, dictionary):
    """Return ``True`` if all items of ``subdict`` are in ``dictionary``."""
    return all(item in dictionary.items() for item in subdict.items())


try:
    # Try catch in case user does not have pytest
    import pytest

except ImportError:

    def identity(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        else:
            return identity

    skip_if_no_pyfftw = identity
    skip_if_no_pywavelets = identity
    skip_if_no_largescale = identity
    skip_if_no_benchmark = identity

else:
    # Mark decorators for test parameters
    skip_if_no_pyfftw = pytest.mark.skipif(
        'not odl.trafos.PYFFTW_AVAILABLE',
        reason='pyFFTW not available',
    )
    skip_if_no_pywavelets = pytest.mark.skipif(
        'not odl.trafos.PYWT_AVAILABLE',
        reason='PyWavelets not available',
    )
    skip_if_no_largescale = pytest.mark.skipif(
        "not pytest.config.getoption('--largescale')",
        reason='--largescale option not given',
    )
    skip_if_no_benchmark = pytest.mark.skipif(
        "not pytest.config.getoption('--benchmark')",
        reason='--benchmark option not given',
    )


def simple_fixture(name, params, fmt=None):
    """Helper to create a pytest fixture using only name and params.

    Parameters
    ----------
    name : str
        Name of the parameters used for the ``ids`` argument
        to `pytest.fixture`.
    params : sequence
        Values to be taken as parameters in the fixture. They are
        used as ``params`` argument to `_pytest.fixtures.fixture`.
        Arguments wrapped in a ``pytest.skipif`` decorator are
        unwrapped for the generation of the test IDs.
    fmt : str, optional
        Use this format string for the generation of the ``ids``.
        For each value, the id string is generated as ::

            fmt.format(name=name, value=value)

        hence the format string must use ``{name}`` and ``{value}``.
        Default format strings are:

            - ``" {name}='{value}' "`` for string parameters,
            - ``" {name}={value} "`` for other types.
    """
    import _pytest

    if fmt is None:
        # Use some intelligence to make good format strings
        fmt_str = " {name}='{value}' "
        fmt_default = " {name}={value} "

        ids = []
        for p in params:
            # TODO: other types of decorators?
            if (
                isinstance(p, _pytest.mark.MarkDecorator)
                and p.name == 'skipif'
            ):
                # Unwrap the wrapped object in the decorator
                if is_string(p.args[1]):
                    ids.append(fmt_str.format(name=name, value=p.args[1]))
                else:
                    ids.append(fmt_default.format(name=name, value=p.args[1]))
            else:
                if is_string(p):
                    ids.append(fmt_str.format(name=name, value=p))
                else:
                    ids.append(fmt_default.format(name=name, value=p))
    else:
        # Use provided `fmt` for everything
        ids = [fmt.format(name=name, value=p) for p in params]

    wrapper = pytest.fixture(scope='module', ids=ids, params=params)
    return wrapper(lambda request: request.param)


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

    Examples
    --------
    Create single noise array:

    >>> space = odl.rn(3)
    >>> array = noise_array(space)

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
        if space.dtype == bool:
            # TODO(kohr-h): use `randint(..., dtype=bool)` from Numpy 1.11 on
            arr = np.random.randint(0, 2, size=space.shape).astype(bool)
        elif np.issubdtype(space.dtype, np.unsignedinteger):
            arr = np.random.randint(0, 10, space.shape)
        elif np.issubdtype(space.dtype, np.signedinteger):
            arr = np.random.randint(-10, 10, space.shape)
        elif np.issubdtype(space.dtype, np.floating):
            arr = np.random.randn(*space.shape)
        elif np.issubdtype(space.dtype, np.complexfloating):
            arr = (
                np.random.randn(*space.shape)
                + 1j * np.random.randn(*space.shape)
            ) / np.sqrt(2.0)
        else:
            raise ValueError('bad dtype {}'.format(space.dtype))

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

    Examples
    --------
    Create single noise element:

    >>> space = odl.rn(3)
    >>> vector = noise_element(space)

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

    The returned elements have the same values as the arrays.

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
    n : int, optional
        Number of elements to create.

    Returns
    -------
    arrays : `numpy.ndarray` or tuple of `numpy.ndarray`
        A single array if ``n == 1``, otherwise a tuple of arrays.
    elements : ``space`` element or tuple of ``space`` elements
        A single element if ``n == 1``, otherwise a tuple of elements.

    Examples
    --------
    Create single noise element:

    >>> space = odl.rn(3)
    >>> arr, vector = noise_elements(space)

    Create multiple noise elements:

    >>> [arr1, arr2], [vector1, vector2] = noise_elements(space, n=2)

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
        raise ImportError(
            'ODL tests cannot be run without `pytest` installed.\n'
            'Run `$ pip install [--user] odl[testing]` in order to install '
            '`pytest`.'
        )

    from .pytest_config import collect_ignore

    this_dir = os.path.dirname(__file__)
    odl_root = os.path.abspath(os.path.join(this_dir, os.pardir, os.pardir))

    args = ['{root}/odl'.format(root=odl_root)]

    ignores = ['--ignore={}'.format(file) for file in collect_ignore]
    args.extend(ignores)

    if arguments is not None:
        args.extend(arguments)

    pytest.main(args)


def run_doctests(skip_if=False, **kwargs):
    """Run all doctests in the current module.

    This function calls ``doctest.testmod()``, by default with the options
    ``optionflags=doctest.NORMALIZE_WHITESPACE`` and
    ``extraglobs={'odl': odl, 'np': np}``. This can be changed with
    keyword arguments.

    Parameters
    ----------
    skip_if : bool
        For ``True``, skip the doctests in this module.
    kwargs :
        Extra keyword arguments passed on to the ``doctest.testmod``
        function.
    """
    from doctest import testmod, NORMALIZE_WHITESPACE, SKIP
    from packaging.version import parse as parse_version
    import odl
    import numpy as np

    optionflags = kwargs.pop('optionflags', NORMALIZE_WHITESPACE)
    if skip_if:
        optionflags |= SKIP

    extraglobs = kwargs.pop('extraglobs', {'odl': odl, 'np': np})

    if run_from_ipython():
        try:
            import spyder
        except ImportError:
            pass
        else:
            if parse_version(spyder.__version__) < parse_version('3.1.4'):
                warnings.warn('A bug with IPython and Spyder < 3.1.4 '
                              'sometimes causes doctests to fail to run. '
                              'Please upgrade Spyder or use another '
                              'interpreter if the doctests do not work.',
                              RuntimeWarning)

    testmod(optionflags=optionflags, extraglobs=extraglobs, **kwargs)


def test_file(file, args=None):
    """Run tests in file with proper default arguments."""
    try:
        import pytest
    except ImportError:
        raise ImportError('ODL tests cannot be run without `pytest` installed.'
                          '\nRun `$ pip install [--user] odl[testing]` in '
                          'order to install `pytest`.')

    if args is None:
        args = []

    args.extend([str(file.replace('\\', '/')), '-v', '--capture=sys'])

    pytest.main(args)


if __name__ == '__main__':
    run_doctests()
