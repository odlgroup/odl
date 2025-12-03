# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities mainly for internal use."""

import contextlib
from collections import OrderedDict
from contextlib import contextmanager
from itertools import product
from packaging.requirements import Requirement

import numpy as np

from odl.core.util.print_utils import is_string

__all__ = (
    "nd_iterator",
    "conj_exponent",
    "nullcontext",
    "writable_array",
    "run_from_ipython",
    "npy_random_seed",
    "unique",
)


def nd_iterator(shape):
    """Iterator over n-d cube with shape.

    Parameters
    ----------
    shape : sequence of int
        The number of points per axis

    Returns
    -------
    nd_iterator : generator
        Generator returning tuples of integers of length ``len(shape)``.

    Examples
    --------
    >>> for pt in nd_iterator([2, 2]):
    ...     print(pt)
    (0, 0)
    (0, 1)
    (1, 0)
    (1, 1)
    """
    return product(*map(range, shape))


def conj_exponent(exp):
    """Conjugate exponent ``exp / (exp - 1)``.

    Parameters
    ----------
    exp : positive float or inf
        Exponent for which to calculate the conjugate. Must be
        at least 1.0.

    Returns
    -------
    conj : positive float or inf
        Conjugate exponent. For ``exp=1``, return ``float('inf')``,
        for ``exp=float('inf')`` return 1. In all other cases, return
        ``exp / (exp - 1)``.
    """
    if exp == 1.0:
        return float("inf")
    if exp == float("inf"):
        return 1.0

    return exp / (exp - 1.0)


@contextmanager
def nullcontext(enter_result=None):
    """Backport of the Python >=3.7 trivial context manager.

    See `the Python documentation
    <https://docs.python.org/3/library/contextlib.html#contextlib.nullcontext>`_
    for details.
    """
    try:
        yield enter_result
    finally:
        pass


try:
    nullcontext = contextlib.nullcontext
except AttributeError:
    pass


@contextmanager
def writable_array(obj, must_be_contiguous: bool = False):
    """Context manager that casts `obj` to a backend-specific array and saves changes
    made on that array back into `obj`.

    Parameters
    ----------
    obj : `array-like`
        Object that should be made available as writable array.
        It must be valid as input to `numpy.asarray` and needs to
        support the syntax ``obj[:] = arr``.
    must_be_contiguous : bool
        Whether the writable array should guarantee standard C order.

    Examples
    --------
    Usage with ODL vectors:

    >>> space = odl.uniform_discr(0, 1, 3)
    >>> x = space.element([1, 2, 3])
    >>> with writable_array(x) as arr:
    ...     arr += [1, 1, 1]
    >>> x
    uniform_discr(0.0, 1.0, 3).element([ 2.,  3.,  4.])

    Note that the changes are in general only saved upon exiting the
    context manager. Before, the input object may remain unchanged.
    """
    if isinstance(obj, np.ndarray):
        if must_be_contiguous and not obj.data.c_contiguous:
            # Needs to convert to contiguous array
            arr = np.ascontiguousarray(obj)
            try:
                yield arr
            finally:
                obj[:] = arr
        else:
            try:
                yield obj
            finally:
                pass
    else:
        with obj.writable_array(must_be_contiguous=must_be_contiguous) as arr:
            yield arr


def run_from_ipython():
    """If the process is run from IPython."""
    return "__IPYTHON__" in globals()


def pkg_supports(feature, pkg_version, pkg_feat_dict):
    """Return bool indicating whether a package supports ``feature``.

    Parameters
    ----------
    feature : str
        Name of a potential feature of a package.
    pkg_version : str
        Version of the package that should be checked for presence of the
        feature.
    pkg_feat_dict : dict
        Specification of features of a package. Each item has the
        following form::

            feature_name: version_specification

        Here, ``feature_name`` is a string that is matched against
        ``feature``, and ``version_specification`` is a string or a
        sequence of strings that specifies version sets. These
        specifications are the same as for ``setuptools`` requirements,
        just without the package name.
        A ``None`` entry signals "no support in any version", i.e.,
        always ``False``.
        If a sequence of requirements are given, they are OR-ed together.
        See ``Examples`` for details.

    Returns
    -------
    supports : bool
        ``True`` if ``pkg_version`` of the package in question supports
        ``feature``, ``False`` otherwise.

    Examples
    --------
    >>> feat_dict = {
    ...     'feat1': '==0.5.1',
    ...     'feat2': '>0.6, <=0.9',  # both required simultaneously
    ...     'feat3': ['>0.6', '<=0.9'],  # only one required, i.e. always True
    ...     'feat4': ['==0.5.1', '>0.6, <=0.9'],
    ...     'feat5': None
    ... }
    >>> pkg_supports('feat1', '0.5.1', feat_dict)
    True
    >>> pkg_supports('feat1', '0.4', feat_dict)
    False
    >>> pkg_supports('feat2', '0.5.1', feat_dict)
    False
    >>> pkg_supports('feat2', '0.6.1', feat_dict)
    True
    >>> pkg_supports('feat2', '0.9', feat_dict)
    True
    >>> pkg_supports('feat2', '1.0', feat_dict)
    False
    >>> pkg_supports('feat3', '0.4', feat_dict)
    True
    >>> pkg_supports('feat3', '1.0', feat_dict)
    True
    >>> pkg_supports('feat4', '0.5.1', feat_dict)
    True
    >>> pkg_supports('feat4', '0.6', feat_dict)
    False
    >>> pkg_supports('feat4', '0.6.1', feat_dict)
    True
    >>> pkg_supports('feat4', '1.0', feat_dict)
    False
    >>> pkg_supports('feat5', '0.6.1', feat_dict)
    False
    >>> pkg_supports('feat5', '1.0', feat_dict)
    False
    """
    # This is an ugly workaround for the future deprecation of pkg_resources

    def parse_requirements(s):
        return (
            Requirement(line)
            for line in s.splitlines()
            if line.strip() and not line.startswith("#")
        )

    feature = str(feature)
    pkg_version = str(pkg_version)
    supp_versions = pkg_feat_dict.get(feature, None)
    if supp_versions is None:
        return False

    # Make sequence from single string
    if is_string(supp_versions):
        supp_versions = [supp_versions]

    # Make valid package requirements
    ver_specs = ["pkg" + supp_ver for supp_ver in supp_versions]
    # Each parse_requirements list contains only one entry since we specify
    # only one package
    ver_reqs = [list(parse_requirements(ver_spec))[0] for ver_spec in ver_specs]

    # If one of the requirements in the list is met, return True
    for req in ver_reqs:
        if req.specifier.contains(pkg_version, prereleases=True):
            return True

    # No match
    return False


@contextmanager
def npy_random_seed(seed):
    """Context manager to temporarily set the NumPy random generator seed.

    Parameters
    ----------
    seed : int or None
        Seed value for the random number generator.
        ``None`` is interpreted as keeping the current seed.

    Examples
    --------
    Use this to make drawing pseudo-random numbers repeatable:

    >>> with npy_random_seed(42):
    ...     rand_int = np.random.randint(10)
    >>> with npy_random_seed(42):
    ...     same_rand_int = np.random.randint(10)
    >>> rand_int == same_rand_int
    True
    """
    do_seed = seed is not None
    orig_rng_state = None
    try:
        if do_seed:
            orig_rng_state = np.random.get_state()
            np.random.seed(seed)
        yield

    finally:
        if do_seed and orig_rng_state is not None:
            np.random.set_state(orig_rng_state)


def unique(seq):
    """Return the unique values in a sequence.

    Parameters
    ----------
    seq : sequence
        Sequence with (possibly duplicate) elements.

    Returns
    -------
    unique : list
        Unique elements of ``seq``.
        Order is guaranteed to be the same as in seq.

    Examples
    --------
    Determine unique elements in list

    >>> unique([1, 2, 3, 3])
    [1, 2, 3]

    >>> unique((1, 'str', 'str'))
    [1, 'str']

    The utility also works with unhashable types:

    >>> unique((1, [1], [1]))
    [1, [1]]
    """
    # First check if all elements are hashable, if so O(n) can be done
    try:
        return list(OrderedDict.fromkeys(seq))
    except TypeError:
        # Non-hashable, resort to O(n^2)
        unique_values = []
        for i in seq:
            if i not in unique_values:
                unique_values.append(i)
        return unique_values


if __name__ == "__main__":
    from odl.core.util.testutils import run_doctests

    run_doctests()
