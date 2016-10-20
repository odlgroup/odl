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

"""Bindings to the ``pyFFTW`` back-end for Fourier transforms.

The `pyFFTW <https://hgomersall.github.io/pyFFTW/>`_  package is a Python
wrapper around the well-known `FFTW <http://fftw.org/>`_ library for fast
Fourier transforms.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range
from future.utils import raise_from

from multiprocessing import cpu_count
import numpy as np
try:
    import pyfftw
    PYFFTW_AVAILABLE = True
except ImportError:
    PYFFTW_AVAILABLE = False

from odl.util import (
    is_real_dtype, dtype_repr, complex_dtype, normalized_axes_tuple)


__all__ = ('pyfftw_call', 'PYFFTW_AVAILABLE')


def pyfftw_call(array_in, array_out, direction='forward', axes=None,
                halfcomplex=False, **kwargs):
    """Calculate the DFT with pyfftw.

    The discrete Fourier (forward) transform calcuates the sum::

        f_hat[k] = sum_j( f[j] * exp(-2*pi*1j * j*k/N) )

    where the summation is taken over all indices
    ``j = (j[0], ..., j[d-1])`` in the range ``0 <= j < N``
    (component-wise), with ``N`` being the shape of the input array.

    The output indices ``k`` lie in the same range, except
    for half-complex transforms, where the last axis ``i`` in ``axes``
    is shortened to ``0 <= k[i] < floor(N[i]/2) + 1``.

    In the backward transform, sign of the the exponential argument
    is flipped.

    Parameters
    ----------
    array_in : `numpy.ndarray`
        Array to be transformed
    array_out : `numpy.ndarray`
        Output array storing the transformed values, may be aliased
        with ``array_in``.
    direction : {'forward', 'backward'}
        Direction of the transform
    axes : int or sequence of ints, optional
        Dimensions along which to take the transform. ``None`` means
        using all axes and is equivalent to ``np.arange(ndim)``.
    halfcomplex : bool, optional
        If ``True``, calculate only the negative frequency part along the
        last axis. If ``False``, calculate the full complex FFT.
        This option can only be used with real input data.

    Other Parameters
    ----------------
    fftw_plan : ``pyfftw.FFTW``, optional
        Use this plan instead of calculating a new one. If specified,
        the options ``planning_effort``, ``planning_timelimit`` and
        ``threads`` have no effect.
    planning_effort : {'estimate', 'measure', 'patient', 'exhaustive'}
        Flag for the amount of effort put into finding an optimal
        FFTW plan. See the `FFTW doc on planner flags
        <http://www.fftw.org/fftw3_doc/Planner-Flags.html>`_.
        Default: 'estimate'
    planning_timelimit : float or ``None``, optional
        Limit planning time to roughly this many seconds.
        Default: ``None`` (no limit)
    threads : int, optional
        Number of threads to use.
        Default: Number of CPUs if the number of data points is larger
        than 4096, else 1.
    normalise_idft : bool, optional
        If ``True``, the result of the backward transform is divided by
        ``1 / N``, where ``N`` is the total number of points in
        ``array_in[axes]``. This ensures that the IDFT is the true
        inverse of the forward DFT.
        Default: ``False``
    import_wisdom : filename or file handle, optional
        File to load FFTW wisdom from. If the file does not exist,
        it is ignored.
    export_wisdom : filename or file handle, optional
        File to append the accumulated FFTW wisdom to

    Returns
    -------
    fftw_plan : ``pyfftw.FFTW``
        The plan object created from the input arguments. It can be
        reused for transforms of the same size with the same data types.
        Note that reuse only gives a speedup if the initial plan
        used a planner flag other than ``'estimate'``.
        If ``fftw_plan`` was specified, the returned object is a
        reference to it.

    Notes
    -----
    * The planning and direction flags can also be specified as
      capitalized and prepended by ``'FFTW_'``, i.e. in the original
      FFTW form.
    * For a ``halfcomplex`` forward transform, the arrays must fulfill
      ``array_out.shape[axes[-1]] == array_in.shape[axes[-1]] // 2 + 1``,
      and vice versa for backward transforms.
    * All planning schemes except ``'estimate'`` require an internal copy
      of the input array but are often several times faster after the
      first call (measuring results are cached). Typically,
      'measure' is a good compromise. If you cannot afford the copy,
      use ``'estimate'``.
    * If a plan is provided via the ``fftw_plan`` parameter, no copy
      is needed internally.
    """
    import pickle

    if not array_in.flags.aligned:
        raise ValueError('input array not aligned')

    if not array_out.flags.aligned:
        raise ValueError('output array not aligned')

    if axes is None:
        axes = tuple(range(array_in.ndim))

    axes = normalized_axes_tuple(axes, array_in.ndim)

    direction = _pyfftw_to_local(direction)
    fftw_plan_in = kwargs.pop('fftw_plan', None)
    planning_effort = _pyfftw_to_local(kwargs.pop('planning_effort',
                                                  'estimate'))
    planning_timelimit = kwargs.pop('planning_timelimit', None)
    threads = kwargs.pop('threads', None)
    normalise_idft = kwargs.pop('normalise_idft', False)
    wimport = kwargs.pop('import_wisdom', '')
    wexport = kwargs.pop('export_wisdom', '')

    # Cast input to complex if necessary
    array_in_copied = False
    if is_real_dtype(array_in.dtype) and not halfcomplex:
        # Need to cast array_in to complex dtype
        array_in = array_in.astype(complex_dtype(array_in.dtype))
        array_in_copied = True

    # Do consistency checks on the arguments
    _pyfftw_check_args(array_in, array_out, axes, halfcomplex, direction)

    # Import wisdom if possible
    if wimport:
        try:
            with open(wimport, 'rb') as wfile:
                wisdom = pickle.load(wfile)
        except IOError:
            wisdom = []
        except TypeError:  # Got file handle
            wisdom = pickle.load(wimport)

        if wisdom:
            pyfftw.import_wisdom(wisdom)

    # Copy input array if it hasn't been done yet and the planner is likely
    # to destroy it. If we already have a plan, we don't have to worry.
    planner_destroys = _pyfftw_destroys_input(
        [planning_effort], direction, halfcomplex, array_in.ndim)
    must_copy_array_in = fftw_plan_in is None and planner_destroys

    if must_copy_array_in and not array_in_copied:
        plan_arr_in = np.empty_like(array_in)
        flags = [_local_to_pyfftw(planning_effort), 'FFTW_DESTROY_INPUT']
    else:
        plan_arr_in = array_in
        flags = [_local_to_pyfftw(planning_effort)]

    if fftw_plan_in is None:
        if threads is None:
            if plan_arr_in.size <= 4096:  # Trade-off wrt threading overhead
                threads = 1
            else:
                threads = cpu_count()

        fftw_plan = pyfftw.FFTW(
            plan_arr_in, array_out, direction=_local_to_pyfftw(direction),
            flags=flags, planning_timelimit=planning_timelimit,
            threads=threads, axes=axes)
    else:
        fftw_plan = fftw_plan_in

    fftw_plan(array_in, array_out, normalise_idft=normalise_idft)

    if wexport:
        try:
            with open(wexport, 'ab') as wfile:
                pickle.dump(pyfftw.export_wisdom(), wfile)
        except TypeError:  # Got file handle
            pickle.dump(pyfftw.export_wisdom(), wexport)

    return fftw_plan


def _pyfftw_to_local(flag):
    return flag.lstrip('FFTW_').lower()


def _local_to_pyfftw(flag):
    return 'FFTW_' + flag.upper()


def _pyfftw_destroys_input(flags, direction, halfcomplex, ndim):
    """Return ``True`` if FFTW destroys an input array, ``False`` otherwise."""
    if any(flag in flags or _pyfftw_to_local(flag) in flags
           for flag in ('FFTW_MEASURE', 'FFTW_PATIENT', 'FFTW_EXHAUSTIVE',
                        'FFTW_DESTROY_INPUT')):
        return True
    elif (direction in ('backward', 'FFTW_BACKWARD') and halfcomplex and
          ndim != 1):
        return True
    else:
        return False


def _pyfftw_check_args(arr_in, arr_out, axes, halfcomplex, direction):
    """Raise an error if anything is not ok with in and out."""
    if len(set(axes)) != len(axes):
        raise ValueError('duplicate axes are not allowed')

    if direction == 'forward':
        out_shape = list(arr_in.shape)
        if halfcomplex:
            try:
                out_shape[axes[-1]] = arr_in.shape[axes[-1]] // 2 + 1
            except IndexError as err:
                raise_from(IndexError('axis index {} out of range for array '
                                      'with {} axes'
                                      ''.format(axes[-1], arr_in.ndim)),
                           err)

        if arr_out.shape != tuple(out_shape):
            raise ValueError('expected output shape {}, got {}'
                             ''.format(tuple(out_shape), arr_out.shape))

        if is_real_dtype(arr_in.dtype):
            out_dtype = complex_dtype(arr_in.dtype)
        elif halfcomplex:
            raise ValueError('cannot combine halfcomplex forward transform '
                             'with complex input')
        else:
            out_dtype = arr_in.dtype

        if arr_out.dtype != out_dtype:
            raise ValueError('expected output dtype {}, got {}'
                             ''.format(dtype_repr(out_dtype),
                                       dtype_repr(arr_out.dtype)))

    elif direction == 'backward':
        in_shape = list(arr_out.shape)
        if halfcomplex:
            try:
                in_shape[axes[-1]] = arr_out.shape[axes[-1]] // 2 + 1
            except IndexError as err:
                raise_from(IndexError('axis index {} out of range for array '
                                      'with {} axes'
                                      ''.format(axes[-1], arr_out.ndim)),
                           err)

        if arr_in.shape != tuple(in_shape):
            raise ValueError('expected input shape {}, got {}'
                             ''.format(tuple(in_shape), arr_in.shape))

        if is_real_dtype(arr_out.dtype):
            in_dtype = complex_dtype(arr_out.dtype)
        elif halfcomplex:
            raise ValueError('cannot combine halfcomplex backward transform '
                             'with complex output')
        else:
            in_dtype = arr_out.dtype

        if arr_in.dtype != in_dtype:
            raise ValueError('expected input dtype {}, got {}'
                             ''.format(dtype_repr(in_dtype),
                                       dtype_repr(arr_in.dtype)))

    else:  # Shouldn't happen
        raise RuntimeError


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests(skip_if=not PYFFTW_AVAILABLE)
