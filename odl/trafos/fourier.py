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

"""Discretized Fourier transform on L^p spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range, super
from future.utils import raise_from

# External
from math import pi
from multiprocessing import cpu_count
import numpy as np
try:
    import pyfftw
    PYFFTW_AVAILABLE = True
except ImportError:
    PYFFTW_AVAILABLE = False

# Internal
from odl.discr.grid import RegularGrid
from odl.discr.lp_discr import (
    DiscreteLp, dspace_type, conj_exponent, discr_sequence_space)
from odl.discr.partition import uniform_partition_fromgrid
from odl.operator.operator import Operator
from odl.set.sets import RealNumbers, ComplexNumbers
from odl.space.base_ntuples import _TYPE_MAP_R2C
from odl.space.cu_ntuples import CudaNtuples
from odl.space.ntuples import Ntuples
from odl.space.fspace import FunctionSpace
from odl.util.numerics import fast_1d_tensor_mult
from odl.util.utility import (
    is_real_dtype, is_scalar_dtype, is_real_floating_dtype,
    is_complex_floating_dtype)


__all__ = ('FourierTransform', 'FourierTransformInverse',
           'DiscreteFourierTransform', 'DiscreteFourierTransformInverse',
           'pyfftw_call', 'dft_preprocess_data', 'dft_postprocess_data',
           'PYFFTW_AVAILABLE')


def reciprocal(grid, shift=True, axes=None, halfcomplex=False):
    """Return the reciprocal of the given regular grid.

    This function calculates the reciprocal (Fourier/frequency space)
    grid for a given regular grid defined by the nodes::

        x[k] = x[0] + k * s,

    where ``k = (k[0], ..., k[d-1])`` is a ``d``-dimensional index in
    the range ``0 <= k < N`` (component-wise). The multi-index
    ``N`` is the shape of the input grid.
    This grid's reciprocal is then given by the nodes::

        xi[j] = xi[0] + j * sigma,

    with the reciprocal grid stride ``sigma = 2*pi / (s * N)``.
    The minimum frequency ``xi[0]`` can in principle be chosen
    freely, but usually it is chosen in a such a way that the reciprocal
    grid is centered around zero. For this, there are two possibilities:

    1. Make the grid point-symmetric around 0.

    2. Make the grid "almost" point-symmetric around zero by shifting
       it to the left by half a reciprocal stride.

    In the first case, the minimum frequency (per axis) is given as::

        xi_1[0] = -pi/s + pi/(s*n) = -pi/s + sigma/2.

    For the second case, it is::

        xi_1[0] = -pi / s.

    Note that the zero frequency is contained in case 1 for an odd
    number of points, while for an even size, the second option
    guarantees that 0 is contained.

    If a real-to-complex (half-complex) transform is to be computed,
    the reciprocal grid has the shape ``M[i] = floor(N[i]/2) + 1``
    in the last transform axis ``i``.

    Parameters
    ----------
    grid : `odl.RegularGrid`
        Original sampling grid
    shift : `bool` or sequence of `bool`, optional
        If `True`, the grid is shifted by half a stride in the negative
        direction. With a sequence, this option is applied separately on
        each axis.
    axes : sequence of `int`, optional
        Dimensions in which to calculate the reciprocal. The sequence
        must have the same length as ``shift`` if the latter is given
        as a sequence. `None` means all axes in ``grid``.
    halfcomplex : `bool`, optional
        If `True`, return the half of the grid with last coordinate
        less than zero. This is related to the fact that for real-valued
        functions, the other half is the mirrored complex conjugate of
        the given half and therefore needs not be stored.

    Returns
    -------
    recip : `RegularGrid`
        The reciprocal grid
    """
    if axes is None:
        axes = list(range(grid.ndim))
    else:
        axes = list(axes)

    # List indicating shift or not per "active" axis, same length as axes
    shift_list = _shift_list(shift, len(axes))

    # Full-length vectors
    stride = grid.stride
    shape = np.array(grid.shape)
    rmin = grid.min_pt.copy()
    rmax = grid.max_pt.copy()
    rshape = list(shape)

    # Shifted axes (full length to avoid ugly double indexing)
    shifted = np.zeros(grid.ndim, dtype=bool)
    shifted[axes] = shift_list
    rmin[shifted] = -pi / stride[shifted]
    # Length min->max increases by double the shift, so we
    # have to compensate by a full stride
    rmax[shifted] = (-rmin[shifted] -
                     2 * pi / (stride[shifted] * shape[shifted]))

    # Non-shifted axes
    not_shifted = np.zeros(grid.ndim, dtype=bool)
    not_shifted[axes] = np.logical_not(shift_list)
    rmin[not_shifted] = ((-1.0 + 1.0 / shape[not_shifted]) *
                         pi / stride[not_shifted])
    rmax[not_shifted] = -rmin[not_shifted]

    # Change last axis shape and max if halfcomplex
    if halfcomplex:
        rshape[axes[-1]] = shape[axes[-1]] // 2 + 1

        # - Odd and not shifted or even and shifted -> 0
        # - Odd and shifted -> - stride / 2
        # - Even and not shifted -> + stride / 2
        last_odd = shape[axes[-1]] % 2 == 1
        last_shifted = shift_list[-1]
        half_rstride = pi / (shape[axes[-1]] * stride[axes[-1]])

        if last_odd:
            if last_shifted:
                rmax[axes[-1]] = -half_rstride
            else:
                rmax[axes[-1]] = 0
        else:
            if last_shifted:
                rmax[axes[-1]] = 0
            else:
                rmax[axes[-1]] = half_rstride

    return RegularGrid(rmin, rmax, rshape)


def inverse_reciprocal(grid, x0, axes=None, halfcomplex=False,
                       halfcx_parity='even'):
    """Return the inverse reciprocal of the given regular grid.

    Given a reciprocal grid::

        xi[j] = xi[0] + j * sigma,

    with a multi-index ``j = (j[0], ..., j[d-1])`` in the range
    ``0 <= j < M``, this function calculates the original grid::

        x[k] = x[0] + k * s

    by using a provided ``x[0]`` and calculating the stride ``s``.

    If the reciprocal grid is interpreted as coming from a usual
    complex-to-complex FFT, it is ``N == M``, and the stride is::

        s = 2*pi / (sigma * N)

    For a reciprocal grid from a real-to-complex (half-complex) FFT,
    it is ``M[i] = floor(N[i]/2) + 1`` in the last transform axis ``i``.
    To resolve the ambiguity regarding the parity of ``N[i]``, the
    it must be specified if the output shape should be even or odd,
    resulting in::

        odd : N[i] = 2 * M[i] - 1
        even: N[i] = 2 * M[i] - 2

    The output stride is calculated with this ``N`` as above in this
    case.

    Parameters
    ----------
    grid : `odl.RegularGrid`
        Original sampling grid
    x0 : array-like
        Minimal point of the inverse reciprocal grid
    axes : sequence of `int`, optional
        Dimensions in which to calculate the reciprocal. The sequence
        must have the same length as ``shift`` if the latter is given
        as a sequence. `None` means all axes in ``grid``.
    halfcomplex : `bool`, optional
        If `True`, interpret the given grid as the reciprocal as used
        in a half-complex FFT (see above). Otherwise, the grid is
        regarded as being used in a complex-to-complex transform.
    halfcx_parity : {'even', 'odd'}
        Use this parity for the shape of the returned grid in the
        last axis of ``axes`` in the case ``halfcomplex=True``

    Returns
    -------
    irecip : `RegularGrid`
        The inverse reciprocal grid
    """
    if axes is None:
        axes = list(range(grid.ndim))
    else:
        axes = list(axes)

    rstride = grid.stride
    rshape = grid.shape

    # Calculate shape of the output grid by adjusting in axes[-1]
    irshape = list(rshape)
    if halfcomplex:
        if str(halfcx_parity).lower() == 'even':
            irshape[axes[-1]] = 2 * rshape[axes[-1]] - 2
        elif str(halfcx_parity).lower() == 'odd':
            irshape[axes[-1]] = 2 * rshape[axes[-1]] - 1
        else:
            raise ValueError("halfcomplex parity '{}' not understood."
                             "".format(halfcx_parity))

    irmin = np.asarray(x0)
    irshape = np.asarray(irshape)
    irstride = np.copy(rstride)
    irstride[axes] = 2 * pi / (irshape[axes] * rstride[axes])
    irmax = irmin + (irshape - 1) * irstride

    return RegularGrid(irmin, irmax, irshape)


def _pyfftw_to_local(flag):
    return flag.lstrip('FFTW_').lower()


def _local_to_pyfftw(flag):
    return 'FFTW_' + flag.upper()


def _pyfftw_destroys_input(flags, direction, halfcomplex, ndim):
    """Return `True` if FFTW destroys an input array, `False` otherwise."""
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
        raise ValueError('Duplicate axes are not allowed.')

    if direction == 'forward':
        out_shape = list(arr_in.shape)
        if halfcomplex:
            try:
                out_shape[axes[-1]] = arr_in.shape[axes[-1]] // 2 + 1
            except IndexError as err:
                raise_from(IndexError('axis index {} out of range for array '
                                      'with {} axes.'
                                      ''.format(axes[-1], arr_in.ndim)),
                           err)

        if arr_out.shape != tuple(out_shape):
            raise ValueError('Expected output shape {}, got {}.'
                             ''.format(tuple(out_shape), arr_out.shape))

        if is_real_dtype(arr_in.dtype):
            out_dtype = _TYPE_MAP_R2C[arr_in.dtype]
        elif halfcomplex:
            raise ValueError('Cannot combine halfcomplex forward transform '
                             'with complex input.')
        else:
            out_dtype = arr_in.dtype

        if arr_out.dtype != out_dtype:
            raise ValueError('Expected output dtype {}, got {}.'
                             ''.format(out_dtype, arr_out.dtype))

    elif direction == 'backward':
        in_shape = list(arr_out.shape)
        if halfcomplex:
            try:
                in_shape[axes[-1]] = arr_out.shape[axes[-1]] // 2 + 1
            except IndexError as err:
                raise_from(IndexError('axis index {} out of range for array '
                                      'with {} axes.'
                                      ''.format(axes[-1], arr_out.ndim)),
                           err)

        if arr_in.shape != tuple(in_shape):
            raise ValueError('Expected input shape {}, got {}.'
                             ''.format(tuple(in_shape), arr_in.shape))

        if is_real_dtype(arr_out.dtype):
            in_dtype = _TYPE_MAP_R2C[arr_out.dtype]
        elif halfcomplex:
            raise ValueError('Cannot combine halfcomplex backward transform '
                             'with complex output.')
        else:
            in_dtype = arr_out.dtype

        if arr_in.dtype != in_dtype:
            raise ValueError('Expected input dtype {}, got {}.'
                             ''.format(in_dtype, arr_in.dtype))

    else:  # Shouldn't happen
        raise RuntimeError


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
        Output array storing the transformed values
    direction : {'forward', 'backward'}
        Direction of the transform
    axes : sequence of `int`, optional
        Dimensions along which to take the transform. `None` means
        using all axis and is equivalent to ``np.arange(ndim)``.
    halfcomplex : `bool`, optional
        If `True`, calculate only the negative frequency part along the
        last axis. If `False`, calculate the full complex FFT.
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
        Default: 'estimate'.
    planning_timelimit : `float`, optional
        Limit planning time to roughly this amount of seconds.
        Default: `None` (no limit)
    threads : `int`, optional
        Number of threads to use.
        Default: Number of CPUs if the number of data points is larger
        than 1000, else 1.
    normalise_idft : `bool`, optional
        If `True`, the backward transform is normalized by
        ``1 / N``, where ``N`` is the total number of points in
        ``array_in[axes]``. This ensures that the IDFT is the true
        inverse of the forward DFT.
        Default: `False`
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
      use 'estimate'.
    """
    import pickle

    if not array_in.flags.aligned:
        raise ValueError('Input array not aligned.')

    if not array_out.flags.aligned:
        raise ValueError('Output array not aligned.')

    # We can use _fftw_to_local here since it strigifies and converts to
    # lowercase
    if axes is None:
        axes = tuple(range(array_in.ndim))
    else:
        axes = tuple(axes)

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
        array_in = array_in.astype(_TYPE_MAP_R2C[array_in.dtype])
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
            if plan_arr_in.size < 1000:  # Somewhat arbitrary
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


class DiscreteFourierTransform(Operator):

    """Plain forward DFT, only evaluating the trigonometric sum.

    This operator calculates the forward DFT::

        f_hat[k] = sum_j( f[j] * exp(-+ 1j*2*pi * j*k/N) )

    without any further shifting or scaling compensation. See the
    `Numpy FFT documentation`_, the `pyfftw API documentation`_ or
    `What FFTW really computes`_ for further information.

    See also
    --------
    numpy.fftn : n-dimensional FFT routine
    numpy.rfftn : n-dimensional half-complex FFT
    pyfftw_call : apply an FFTW transform

    References
    ----------
    .. _Numpy FFT documentation:
        http://docs.scipy.org/doc/numpy/reference/routines.fft.html
    .. _pyfftw API documentation:
       http://hgomersall.github.io/pyFFTW/pyfftw/pyfftw.html
    .. _What FFTW really computes:
       http://www.fftw.org/fftw3_doc/What-FFTW-Really-Computes.html
    """

    def __init__(self, dom, ran=None, axes=None, sign='-', halfcomplex=False,
                 impl='numpy'):
        """Initialize a new instance.

        Parameters
        ----------
        dom : `DiscreteLp`
            Domain of the Fourier transform. If its
            `DiscreteLp.exponent` is equal to 2.0, this operator has
            an adjoint which is equal to the inverse.
        ran : `DiscreteLp`, optional
            Range of the Fourier transform. If not given, the range
            is determined from ``dom`` and the other parameters as
            a `discr_sequence_space` with exponent ``p / (p - 1)``
            (read as 'inf' for p=1 and 1 for p='inf').
        axes : sequence of `int`, optional
            Dimensions in which a transform is to be calculated. `None`
            means all axes.
        sign : {'-', '+'}, optional
            Sign of the complex exponent. Default: '-'
        halfcomplex : `bool`, optional
            If `True`, calculate only the negative frequency part
            along the last axis in ``axes`` for real input. This
            reduces the size of the range to ``floor(N[i]/2) + 1`` in
            this axis ``i``, where ``N`` is the shape of the input
            arrays.
            Otherwise, calculate the full complex FFT. If ``dom_dtype``
            is a complex type, this option has no effect.
        impl : {'numpy', 'pyfftw'}
            Backend for the FFT implementation. The 'pyfftw' backend
            is faster but requires the ``pyfftw`` package.

        Examples
        --------
        Complex-to-complex (default) transforms have the same grids
        in domain and range:

        >>> domain = discr_sequence_space((2, 4))
        >>> fft = DiscreteFourierTransform(domain)
        >>> fft.domain.shape
        (2, 4)
        >>> fft.range.shape
        (2, 4)

        Real-to-complex transforms have a range grid with shape
        ``n // 2 + 1`` in the last tranform axis:

        >>> domain = discr_sequence_space((2, 3, 4), dtype='float')
        >>> axes = (0, 1)
        >>> fft = DiscreteFourierTransform(
        ...     domain, halfcomplex=True, axes=axes)
        >>> fft.range.shape   # shortened in the second axis
        (2, 2, 4)
        >>> fft.domain.shape
        (2, 3, 4)
        """
        if not isinstance(dom, DiscreteLp):
            raise TypeError('domain {!r} is not a DiscreteLp instance.'
                            ''.format(dom))
        if ran is not None and not isinstance(ran, DiscreteLp):
            raise TypeError('range {!r} is not a DiscreteLp instance.'
                            ''.format(ran))

        # Implementation
        self._impl = str(impl).lower()
        if self.impl not in ('numpy', 'pyfftw'):
            raise ValueError("implementation '{}' not understood."
                             "".format(impl))
        if self.impl == 'pyfftw' and not PYFFTW_AVAILABLE:
            raise ValueError('pyfftw backend not available.')

        # Axes
        if axes is None:
            axes_ = np.arange(dom.ndim)
        else:
            axes_ = np.atleast_1d(axes)
            axes_[axes_ < 0] += dom.ndim
            if not all(0 <= ax < dom.ndim for ax in axes_):
                raise ValueError('invalid entries in axes.')
        self._axes = list(axes_)

        # Half-complex
        if dom.field == ComplexNumbers():
            self._halfcomplex = False
            ran_dtype = dom.dtype
        else:
            self._halfcomplex = bool(halfcomplex)
            ran_dtype = _TYPE_MAP_R2C[dom.dtype]

        # Sign of the transform
        if sign not in ('+', '-'):
            raise ValueError("sign '{}' not understood.".format(sign))
        if sign == '+' and self.halfcomplex:
            raise ValueError("cannot combine sign '+' with a half-complex "
                             "transform.")
        self._sign = sign

        # Calculate the range
        ran_shape = reciprocal(
            dom.grid, shift=False, halfcomplex=halfcomplex, axes=axes).shape

        if ran is None:
            impl = 'cuda' if isinstance(dom.dspace, CudaNtuples) else 'numpy'
            ran = discr_sequence_space(
                ran_shape, conj_exponent(dom.exponent), impl=impl,
                dtype=ran_dtype, order=dom.order)
        else:
            if ran.shape != ran_shape:
                raise ValueError('expected range shape {}, got {}.'
                                 ''.format(ran_shape, ran.shape))
            if ran.dtype != ran_dtype:
                raise ValueError('expected range data type {}, got {}.'
                                 ''.format(ran_dtype, ran.dtype))

        super().__init__(dom, ran, linear=True)
        self._fftw_plan = None

    def _call(self, x, out, **kwargs):
        """Implement ``self(x, out[, **kwargs])``.

        Parameters
        ----------
        x : domain `element`
            Discretized function to be transformed
        out : range `element`
            Element to which the output is written

        Notes
        -----
        See the `pyfftw_call` function for ``**kwargs`` options.
        The parameters ``axes`` and ``halfcomplex`` cannot be
        overridden.

        See also
        --------
        pyfftw_call : Call pyfftw backend directly
        """
        # TODO: Implement zero padding
        if self.impl == 'numpy':
            out[:] = self._call_numpy(x.asarray())
        else:
            out[:] = self._call_pyfftw(x.asarray(), out.asarray(), **kwargs)

    def _call_numpy(self, x):
        """Return ``self(x)`` using numpy.

        Parameters
        ----------
        x : `numpy.ndarray`
            Input array to be transformed

        Returns
        -------
        out : `numpy.ndarray`
            Result of the transform
        """
        assert isinstance(x, np.ndarray)

        if self.halfcomplex:
            return np.fft.rfftn(x, axes=self.axes)
        else:
            if self.sign == '-':
                return np.fft.fftn(x, axes=self.axes)
            else:
                # Need to undo Numpy IFFT scaling
                return (np.prod(np.take(self.domain.shape, self.axes)) *
                        np.fft.ifftn(x, axes=self.axes))

    def _call_pyfftw(self, x, out, **kwargs):
        """Implement ``self(x[, out, **kwargs])`` using pyfftw.

        Parameters
        ----------
        x : `numpy.ndarray`
            Input array to be transformed
        out : `numpy.ndarray`
            Output array storing the result
        flags : sequence of `str`, optional
            Flags for the transform. ``'FFTW_UNALIGNED'`` is not
            supported, and ``'FFTW_DESTROY_INPUT'`` is enabled by
            default. See the `pyfftw API documentation`_
            for futher information.
            Default: ``('FFTW_MEASURE',)``
        threads : positive `int`, optional
            Number of threads to use. Default: 1
        planning_timelimit : `float` or `None`, optional
            Rough upper limit in seconds for the planning step of the
            transform. `None` means no limit. See the
            `pyfftw API documentation`_ for futher information.

        Returns
        -------
        out : `numpy.ndarray`
            Result of the transform. If ``out`` was given, the returned
            object is a reference to it.

        References
        ----------
        .. _pyfftw API documentation:
           http://hgomersall.github.io/pyFFTW/pyfftw/pyfftw.html
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(out, np.ndarray)

        kwargs.pop('normalise_idft', None)  # Using 'False' here
        kwargs.pop('axes', None)
        kwargs.pop('halfcomplex', None)
        flags = list(_pyfftw_to_local(flag) for flag in
                     kwargs.pop('flags', ('FFTW_MEASURE',)))
        try:
            flags.remove('unaligned')
        except ValueError:
            pass
        try:
            flags.remove('destroy_input')
        except ValueError:
            pass
        effort = flags[0] if flags else 'measure'

        direction = 'forward' if self.sign == '-' else 'backward'
        self._fftw_plan = pyfftw_call(
            x, out, direction=direction, axes=self.axes,
            halfcomplex=self.halfcomplex, planning_effort=effort,
            fftw_plan=self._fftw_plan, normalise_idft=False)

        return out

    @property
    def impl(self):
        """Backend for the FFT implementation."""
        return self._impl

    @property
    def axes(self):
        """Axes along the FT is calculated by this operator."""
        return self._axes

    @property
    def sign(self):
        """Sign of the complex exponent in the transform."""
        return self._sign

    @property
    def halfcomplex(self):
        """Return `True` if the last transform axis is halved."""
        return self._halfcomplex

    @property
    def adjoint(self):
        """Adjoint transform, equal to the inverse."""
        if self.domain.field == RealNumbers():
            raise NotImplementedError(
                'Fourier transform from real to complex space has no adjoint.')
        if self.domain.exponent == 2.0 and self.range.exponent == 2.0:
            return self.inverse
        else:
            raise NotImplementedError(
                'no adjoint defined for exponents ({}, {}) != (2, 2).'
                ''.format(self.domain.exponent, self.range.exponent))

    @property
    def inverse(self):
        """Inverse Fourier transform."""
        sign = '+' if self.sign == '-' else '-'
        return DiscreteFourierTransformInverse(
            dom=self.range, ran=self.domain, axes=self.axes,
            halfcomplex=self.halfcomplex, sign=sign)

    def init_fftw_plan(self, planning_effort='measure', **kwargs):
        """Initialize the FFTW plan for this transform for later use.

        If the implementation of this operator is not 'pyfftw', this
        method has no effect.

        Parameters
        ----------
        planning_effort : {'estimate', 'measure', 'patient', 'exhaustive'}
            Flag for the amount of effort put into finding an optimal
            FFTW plan. See the `FFTW doc on planner flags
            <http://www.fftw.org/fftw3_doc/Planner-Flags.html>`_.
        planning_timelimit : `float`, optional
            Limit planning time to roughly this amount of seconds.
            Default: `None` (no limit)
        threads : `int`, optional
            Number of threads to use. Default: 1
        """
        if self.impl != 'pyfftw':
            return

        x = self.domain.element()
        y = self.range.element()
        kwargs.pop('planning_timelimit', None)

        direction = 'forward' if self.sign == '-' else 'backward'
        self._fftw_plan = pyfftw_call(
            x.asarray(), y.asarray(), direction=direction,
            halfcomplex=self.halfcomplex, axes=self.axes,
            planning_effort=planning_effort, **kwargs)

    def clear_fftw_plan(self):
        """Delete the FFTW plan of this transform."""
        self._fftw_plan = None


class DiscreteFourierTransformInverse(DiscreteFourierTransform):

    """Plain backward DFT, only evaluating the trigonometric sum.

    This operator calculates the inverse DFT::

        f[k] = 1/prod(N) * sum_j( f_hat[j] * exp(+- 1j*2*pi * j*k/N) )

    without any further shifting or scaling compensation. See the
    `Numpy FFT documentation`_, the `pyfftw API documentation`_ or
    `What FFTW really computes`_ for further information.

    See also
    --------
    numpy.ifftn : n-dimensional inverse FFT routine
    numpy.irfftn : n-dimensional half-complex inverse FFT
    pyfftw_call : apply an FFTW transform

    References
    ----------
    .. _Numpy FFT documentation:
        http://docs.scipy.org/doc/numpy/reference/routines.fft.html
    .. _pyfftw API documentation:
       http://hgomersall.github.io/pyFFTW/pyfftw/pyfftw.html
    .. _What FFTW really computes:
       http://www.fftw.org/fftw3_doc/What-FFTW-Really-Computes.html
    """
    def __init__(self, ran, dom=None, axes=None, sign='+', halfcomplex=False,
                 impl='numpy'):
        """Initialize a new instance.

        Parameters
        ----------
        ran : `DiscreteLp`
            Range of the inverse Fourier transform. If its
            `DiscreteLp.exponent` is equal to 2.0, this operator has
            an adjoint which is equal to the inverse.
        dom : `DiscreteLp`, optional
            Domain of the inverse Fourier transform. If not given, the
            domain is determined from ``ran`` and the other parameters
            as a `discr_sequence_space` with exponent ``p / (p - 1)``
            (read as 'inf' for p=1 and 1 for p='inf').
        axes : sequence of `int`, optional
            Dimensions in which a transform is to be calculated. `None`
            means all axes.
        sign : {'-', '+'}, optional
            Sign of the complex exponent. Default: '-'
        halfcomplex : `bool`, optional
            If `True`, interpret the last axis in ``axes`` as the
            negative frequency part of the transform of a real signal
            and calculate a "half-complex-to-real" inverse FFT. In this
            case, the domain has by default the shape
            ``floor(N[i]/2) + 1`` in this axis ``i``.
            Otherwise, domain and range have the same shape. If
            ``ran`` is a complex space, this option has no effect.
        impl : {'numpy', 'pyfftw'}
            Backend for the FFT implementation. The 'pyfftw' backend
            is faster but requires the ``pyfftw`` package.

        Examples
        --------
        Complex-to-complex (default) transforms have the same grids
        in domain and range:

        >>> range_ = discr_sequence_space((2, 4))
        >>> ifft = DiscreteFourierTransformInverse(range_)
        >>> ifft.domain.shape
        (2, 4)
        >>> ifft.range.shape
        (2, 4)

        Complex-to-real transforms have a domain grid with shape
        ``n // 2 + 1`` in the last tranform axis:

        >>> range_ = discr_sequence_space((2, 3, 4), dtype='float')
        >>> axes = (0, 1)
        >>> ifft = DiscreteFourierTransformInverse(
        ...     range_, halfcomplex=True, axes=axes)
        >>> ifft.domain.shape   # shortened in the second axis
        (2, 2, 4)
        >>> ifft.range.shape
        (2, 3, 4)
        """
        # Use the checks and init code from the parent class and then simply
        # switch domain and range.
        # Need to switch sign back and forth to check for the correct error
        # scenarios.
        bwd_sign = sign
        fwd_sign = '-' if sign == '+' else '+'
        super().__init__(dom=ran, ran=dom, axes=axes, sign=fwd_sign,
                         halfcomplex=halfcomplex, impl=impl)
        self._domain, self._range = self._range, self._domain
        self._sign = bwd_sign

    def _call_numpy(self, x):
        """Return ``self(x)`` using numpy.

        Parameters
        ----------
        x : `numpy.ndarray`
            Input array to be transformed

        Returns
        -------
        out : `numpy.ndarray`
            Result of the transform
        """
        if self.halfcomplex:
            return np.fft.irfftn(x, axes=self.axes)
        else:
            if self.sign == '+':
                return np.fft.ifftn(x, axes=self.axes)
            else:
                return (np.fft.fftn(x, axes=self.axes) /
                        np.prod(np.take(self.domain.shape, self.axes)))

    def _call_pyfftw(self, x, out, **kwargs):
        """Implement ``self(x[, out, **kwargs])``using pyfftw.

        Parameters
        ----------
        x : domain element
            Input vector to be transformed
        out : range element
            Output vector storing the result
        flags : sequence of `str`, optional
            Flags for the transform. ``'FFTW_UNALIGNED'`` is not
            supported, and ``'FFTW_DESTROY_INPUT'`` is enabled by
            default. See the `pyfftw API documentation`_
            for futher information.
            Default: ``('FFTW_MEASURE',)``
        threads : positive `int`, optional
            Number of threads to use. Default: 1
        planning_timelimit : `float` or `None`, optional
            Rough upper limit in seconds for the planning step of the
            transform. `None` means no limit. See the
            `pyfftw API documentation`_ for futher information.

        Returns
        -------
        out : `DiscreteLpVector`
            Result of the transform. If ``out`` was given, the returned
            object is a reference to it.

        References
        ----------
        .. _pyfftw API documentation:
           http://hgomersall.github.io/pyFFTW/pyfftw/pyfftw.html
        """
        kwargs.pop('normalise_idft', None)  # Using 'True' here
        kwargs.pop('axes', None)
        kwargs.pop('halfcomplex', None)
        flags = list(_pyfftw_to_local(flag) for flag in
                     kwargs.pop('flags', ('FFTW_MEASURE',)))
        try:
            flags.remove('unaligned')
        except ValueError:
            pass
        try:
            flags.remove('destroy_input')
        except ValueError:
            pass
        effort = flags[0] if flags else 'measure'

        direction = 'forward' if self.sign == '-' else 'backward'
        self._fftw_plan = pyfftw_call(
            x, out, direction=direction, axes=self.axes,
            halfcomplex=self.halfcomplex, planning_effort=effort,
            fftw_plan=self._fftw_plan, normalise_idft=True)

        # Need to normalize for 'forward', no way to force pyfftw
        if self.sign == '-':
            out /= np.prod(np.take(self.domain.shape, self.axes))

        return out

    @property
    def inverse(self):
        """Inverse Fourier transform."""
        sign = '-' if self.sign == '+' else '+'
        return DiscreteFourierTransform(
            dom=self.range, ran=self.domain, axes=self.axes,
            halfcomplex=self.halfcomplex, sign=sign)


def dft_preprocess_data(arr, shift=True, axes=None, sign='-', out=None):
    """Pre-process the real-space data before DFT.

    This function multiplies the given data with the separable
    function::

        p(x) = exp(+- 1j * dot(x - x[0], xi[0]))

    where ``x[0]`` and ``xi[0]`` are the minimum coodinates of
    the real-space and reciprocal grids, respectively. The sign of
    the exponent depends on the choice of ``sign``. In discretized
    form, this function becomes an array::

        p[k] = exp(+- 1j * k * s * xi[0])

    If the reciprocal grid is not shifted, i.e. symmetric around 0,
    it is ``xi[0] =  pi/s * (-1 + 1/N)``, hence::

        p[k] = exp(-+ 1j * pi * k * (1 - 1/N))

    For a shifted grid, we have :math:``xi[0] =  -pi/s``, thus the
    array is given by::

        p[k] = (-1)**k

    Parameters
    ----------
    arr : `array-like`
        Array to be pre-processed. If its data type is a real
        non-floating type, it is converted to 'float64'.
    shift : `bool` or sequence of `bool`, optional
        If `True`, the grid is shifted by half a stride in the negative
        direction. With a sequence, this option is applied separately on
        each axis.
    axes : sequence of `int`, optional
        Dimensions in which to calculate the reciprocal. The sequence
        must have the same length as ``shift`` if the latter is given
        as a sequence. `None` means all axes in ``dfunc``.
    sign : {'-', '+'}, optional
        Sign of the complex exponent
    out : `numpy.ndarray`, optional
        Array in which the result is stored. If ``out is arr``,
        an in-place modification is performed. For real data type,
        this is only possible for ``shift=True`` since the factors are
        complex otherwise.

    Returns
    -------
    out : `numpy.ndarray`
        Result of the pre-processing. If ``out`` was given, the returned
        object is a reference to it.
    """
    arr = np.asarray(arr)
    if not is_scalar_dtype(arr.dtype):
        raise ValueError('array has non-scalar data type {}.'
                         ''.format(arr.dtype))
    elif is_real_dtype(arr.dtype) and not is_real_floating_dtype(arr.dtype):
        arr = arr.astype('float64')

    if axes is None:
        axes = list(range(arr.ndim))
    else:
        axes = list(axes)

    shape = arr.shape
    shift_list = _shift_list(shift, len(axes))

    # Make a copy of arr with correct data type if necessary, or copy values.
    if out is None:
        if is_real_dtype(arr.dtype) and not all(shift_list):
            out = np.array(arr, dtype=_TYPE_MAP_R2C[arr.dtype], copy=True)
        else:
            out = arr.copy()
    elif out is arr:
        pass
    else:
        out[:] = arr

    if is_real_dtype(out.dtype) and not shift:
        raise ValueError('cannot pre-process real input in place without '
                         'shift.')

    if sign == '-':
        imag = -1j
    elif sign == '+':
        imag = 1j
    else:
        raise ValueError("sign '{}' not understood.".format(sign))

    def _onedim_arr(length, shift):
        if shift:
            # (-1)^indices
            indices = np.arange(length, dtype='int8')
            arr = -2 * np.mod(indices, 2) + 1.0
        else:
            indices = np.arange(length, dtype='float64')
            arr = np.exp(-imag * pi * indices * (1 - 1.0 / length))
        return arr

    onedim_arrs = []
    for axis, shift in zip(axes, shift_list):
        length = shape[axis]
        onedim_arrs.append(_onedim_arr(length, shift))

    fast_1d_tensor_mult(out, onedim_arrs, axes=axes, out=out)
    return out


def _interp_kernel_ft(norm_freqs, interp):
    """Scaled FT of a one-dimensional interpolation kernel.

    For normalized frequencies ``-1/2 <= xi <= 1/2``, this
    function returns::

        sinc(pi * xi)**k / sqrt(2 * pi)

    where ``k=1`` for 'nearest' and ``k=2`` for 'linear' interpolation.

    Parameters
    ----------
    norm_freqs : `numpy.ndarray`
        Normalized frequencies between -1/2 and 1/2
    interp : {'nearest', 'linear'}
        Type of interpolation kernel

    Returns
    -------
    ker_ft : `numpy.ndarray`
        Values of the kernel FT at the given frequencies
    """
    # Numpy's sinc(x) is equal to the 'math' sinc(pi * x)
    interp_ = str(interp).lower()
    if interp_ == 'nearest':
        return np.sinc(norm_freqs) / np.sqrt(2 * np.pi)
    elif interp_ == 'linear':
        return np.sinc(norm_freqs) ** 2 / np.sqrt(2 * np.pi)
    else:
        raise ValueError("interpolation '{}' not understood.".format(interp))


def dft_postprocess_data(arr, real_grid, recip_grid, shifts, axes,
                         interp, sign='-', op='multiply', out=None):
    """Post-process the Fourier-space data after DFT.

    This function multiplies the given data with the separable
    function::

        q(xi) = exp(+- 1j * dot(x[0], xi)) * s * phi_hat(xi_bar)

    where ``x[0]`` and ``s`` are the minimum point and the stride of
    the real-space grid, respectively, and ``phi_hat(xi_bar)`` is the FT
    of the interpolation kernel. The sign of the exponent depends on the
    choice of ``sign``. Note that for ``op='divide'`` the
    multiplication with ``s * phi_hat(xi_bar)`` is replaced by a
    division with the same array.

    In discretized form on the reciprocal grid, the exponential part
    of this function becomes an array::

        q[k] = exp(+- 1j * dot(x[0], xi[k]))

    and the arguments ``xi_bar`` to the interpolation kernel
    are the normalized frequencies::

        for 'shift=True'  : xi_bar[k] = -pi + pi * (2*k) / N
        for 'shift=False' : xi_bar[k] = -pi + pi * (2*k+1) / N

    See [Pre+2007]_, Section 13.9 "Computing Fourier Integrals Using
    the FFT" for a similar approach.

    Parameters
    ----------
    arr : `array-like`
        Array to be pre-processed. It is converted to data type
        'complex128'.
    real_grid : `RegularGrid`
        Real space grid in the transform
    recip_grid : `RegularGrid`
        Reciprocal grid in the transform
    shifts : sequence of `bool`
        If `True`, the grid is shifted by half a stride in the negative
        direction in the corresponding axes. The sequence must have the
        same length as ``axes``.
    axes : sequence of `int`
        Dimensions along which to take the transform. The sequence must
        have the same length as ``shifts``.
    interp : `str`
        Interpolation scheme used in the real-space
    sign : {'-', '+'}, optional
        Sign of the complex exponent
    op : {'multiply', 'divide'}
        Operation to perform with the stride times the interpolation
        kernel FT
    out : `numpy.ndarray`, optional
        Array in which the result is stored. If ``out is arr``, an
        in-place modification is performed.

    Returns
    -------
    out : `numpy.ndarray`
        Result of the post-processing. If ``out`` was given, the returned
        object is a reference to it.
    """
    arr = np.asarray(arr)
    if not is_complex_floating_dtype(arr.dtype):
        arr = arr.astype('complex128')

    if out is None:
        out = arr.copy()
    elif out is not arr:
        out[:] = arr

    shift_list = list(shifts)
    axes = list(axes)

    if sign == '-':
        imag = -1j
    elif sign == '+':
        imag = 1j
    else:
        raise ValueError("sign '{}' not understood.".format(sign))

    op, op_in = str(op).lower(), op
    if op not in ('multiply', 'divide'):
        raise ValueError("kernel op '{}' not understood.".format(op_in))

    onedim_arrs = []
    for ax, shift in zip(axes, shift_list):
        x = real_grid.min_pt[ax]
        xi = recip_grid.coord_vectors[ax]

        # First part: exponential array
        onedim_arr = (np.exp(imag * x * xi))

        # Second part: interpolation kernel
        len_dft = recip_grid.shape[ax]
        len_orig = real_grid.shape[ax]
        halfcomplex = (len_dft < len_orig)
        odd = len_orig % 2

        if shift:
            # f_k = -0.5 + k / N
            fmin = -0.5
            if halfcomplex:
                if odd:
                    fmax = - 1.0 / (2 * len_orig)
                else:
                    fmax = 0.0
            else:
                # Always -0.5 + (N-1)/N = 0.5 - 1/N
                fmax = 0.5 - 1.0 / len_orig

        else:
            # f_k = -0.5 + 1/(2*N) + k / N
            fmin = -0.5 + 1.0 / (2 * len_orig)
            if halfcomplex:
                if odd:
                    fmax = 0.0
                else:
                    fmax = 1.0 / (2 * len_orig)
            else:
                # Always -0.5 + 1/(2*N) + (N-1)/N = 0.5 - 1/(2*N)
                fmax = 0.5 - 1.0 / (2 * len_orig)

        freqs = np.linspace(fmin, fmax, num=len_dft)
        stride = real_grid.stride[ax]

        if op == 'multiply':
            onedim_arr *= stride * _interp_kernel_ft(freqs, interp)
        else:
            onedim_arr /= stride * _interp_kernel_ft(freqs, interp)

        onedim_arrs.append(onedim_arr)

    fast_1d_tensor_mult(out, onedim_arrs, axes=axes, out=out)
    return out


def _recip_space(space, shifts, halfcomplex, axes):
    """Return the reciprocal space of ``space``."""
    # Calculate range
    recip_grid = reciprocal(space.grid, shift=shifts, halfcomplex=halfcomplex,
                            axes=axes)

    # Make a partition with nodes on the boundary in the transform axes.
    begin = {i: recip_grid.min_pt[i] for i in axes}
    end = {i: recip_grid.max_pt[i] for i in axes}
    part = uniform_partition_fromgrid(recip_grid, begin=begin, end=end)
    # Always complex space
    ran_fspace = FunctionSpace(part.set, field=ComplexNumbers())

    if is_real_dtype(space.dtype):
        ran_dtype = _TYPE_MAP_R2C[space.dtype]
    else:
        ran_dtype = space.dtype

    conj_exp = conj_exponent(space.exponent)
    ran_dspace_type = dspace_type(ran_fspace, impl='numpy', dtype=ran_dtype)
    ran_dspace = ran_dspace_type(part.size, dtype=ran_dtype,
                                 weight=part.cell_volume, exponent=conj_exp)

    recip_spc = DiscreteLp(ran_fspace, part, ran_dspace,
                           exponent=conj_exp)

    return recip_spc


class FourierTransform(Operator):

    """Discretized Fourier transform between discrete L^p spaces.

    This operator is the discretized variant of the continuous
    `Fourier Transform
    <https://en.wikipedia.org/wiki/Fourier_Transform>`_ between
    Lebesgue L^p spaces. It applies a three-step procedure consisting
    of a pre-processing step of the data, an FFT evaluation and
    a post-processing step. Pre- and post-processing account for
    the shift and scaling of the real-space and Fourier-space grids.

    The sign convention ('-' vs. '+') can be changed with the ``sign``
    parameter.

    See also
    --------
    dft_preprocess_data
    pyfftw_call
    dft_postprocess_data
    """

    def __init__(self, dom, ran=None, impl='numpy', **kwargs):
        """
        Parameters
        ----------
        dom : `DiscreteLp`
            Domain of the Fourier transform. If the
            `DiscreteLp.exponent` of ``dom`` and ``ran`` are equal
            to 2.0, this operator has an adjoint which is equal to its
            inverse.
        ran : `DiscreteLp`, optional
            Range of the Fourier transform. If not given, the range
            is determined from ``dom`` and the other parameters. The
            exponent is chosen to be the conjugate ``p / (p - 1)``,
            which reads as 'inf' for p=1 and 1 for p='inf'.
        impl : {'numpy', 'pyfftw'}
            Backend for the FFT implementation. The 'pyfftw' backend
            is faster but requires the ``pyfftw`` package.
        axes : sequence of `int`, optional
            Dimensions along which to take the transform.
            Default: all axes
        sign : {'-', '+'}, optional
            Sign of the complex exponent. Default: '-'
        halfcomplex : `bool`, optional
            If `True`, calculate only the negative frequency part
            along the last axis for real input. If `False`,
            calculate the full complex FFT.
            For complex ``dom``, it has no effect.
            Default: `True`
        shift : `bool` or sequence of `bool`, optional
            If `True`, the reciprocal grid is shifted by half a stride in
            the negative direction. With a boolean sequence, this option
            is applied separately to each axis.
            If a sequence is provided, it must have the same length as
            ``axes`` if supplied. Note that this must be set to `True`
            in the halved axis in half-complex transforms.
            Default: `True`

        Notes
        -----
        The `Operator.range` of this operator always has the
        `ComplexNumbers` as its `LinearSpace.field`, i.e. if the
        field of ``dom`` is the `RealNumbers`, this operator has no
        `Operator.adjoint`.
        """
        # TODO: variants wrt placement of 2*pi

        if not isinstance(dom, DiscreteLp):
            raise TypeError('domain {!r} is not a `DiscreteLp` instance.'
                            ''.format(dom))
        if not isinstance(dom.dspace, Ntuples):
            raise NotImplementedError(
                'Only Numpy-based data spaces are supported, got {}.'
                ''.format(dom.dspace))

        self._axes = list(kwargs.pop('axes', range(dom.ndim)))

        self._impl = str(impl).lower()
        if self.impl not in ('numpy', 'pyfftw'):
            raise ValueError("implementation '{}' not understood."
                             "".format(impl))
        if self.impl == 'pyfftw' and not PYFFTW_AVAILABLE:
            raise ValueError('pyfftw backend not available.')

        # Handle half-complex yes/no and shifts
        if isinstance(dom.grid, RegularGrid):
            if dom.field == ComplexNumbers():
                self._halfcomplex = False
            else:
                self._halfcomplex = bool(kwargs.pop('halfcomplex', True))

            self._shifts = _shift_list(kwargs.pop('shift', True),
                                       length=len(self.axes))
        else:
            raise NotImplementedError('irregular grids not yet supported.')

        sign = kwargs.pop('sign', '-')
        if sign not in ('+', '-'):
            raise ValueError("sign '{}' not understood.".format(sign))
        if sign == '+' and self.halfcomplex:
            raise ValueError("cannot combine sign '+' with a half-complex "
                             "transform.")
        self._sign = sign

        # Need to filter out this situation since the pre-processing step
        # casts to complex otherwise, and then no half-complex transform
        # is possible.
        if dom.field == RealNumbers() and not self.shifts[-1]:
            raise ValueError('shift must be True in the halved (last) axis '
                             'in half-complex transforms.')

        if ran is None:
            # self._halfcomplex and self._axes need to be set for this
            ran = _recip_space(dom, self.shifts, self.halfcomplex, self.axes)

        super().__init__(dom, ran, linear=True)
        self._fftw_plan = None

    def _call(self, x, out, **kwargs):
        """Implement ``self(x, out[, **kwargs])``.

        Parameters
        ----------
        x : domain `element`
            Discretized function to be transformed
        out : range `element`
            Element to which the output is written

        Notes
        -----
        See the `pyfftw_call` function for ``**kwargs`` options.
        The parameters ``axes`` and ``halfcomplex`` cannot be
        overridden.

        See also
        --------
        pyfftw_call : Call pyfftw backend directly
        """
        # TODO: Implement zero padding
        if self.impl == 'numpy':
            out[:] = self._call_numpy(x.asarray())
        else:
            # 0-overhead assignment if asarray() does not copy
            out[:] = self._call_pyfftw(x.asarray(), out.asarray(), **kwargs)

    def _call_numpy(self, x):
        """Return ``self(x)`` for numpy back-end.

        Parameters
        ----------
        x : `numpy.ndarray`
            Array representing the function to be transformed

        Returns
        -------
        out : `numpy.ndarray`
            Result of the transform. If ``out`` was given, the returned
            object is a reference to it.
        """
        assert isinstance(x, np.ndarray)

        # Pre-processing before calculating the sums
        preproc = dft_preprocess_data(
            x, shift=self.shifts, axes=self.axes, sign=self.sign)

        # The actual call to the FFT library
        if self.halfcomplex:
            out = np.fft.rfftn(preproc, axes=self.axes)
        else:
            if self.sign == '-':
                out = np.fft.fftn(preproc, axes=self.axes)
            elif self.sign == '+':
                out = np.fft.ifftn(preproc, axes=self.axes)
                # Numpy's FFT normalizes by 1 / prod(shape[axes]), we
                # need to undo that
                out *= np.prod(np.take(self.domain.shape, self.axes))
            else:
                raise RuntimeError('bad sign {}.'.format(self.sign))

        # Post-processing accounting for shift, scaling and interpolation
        dft_postprocess_data(
            out, real_grid=self.domain.grid, recip_grid=self.range.grid,
            shifts=self.shifts, axes=self.axes, sign=self.sign,
            interp=self.domain.interp, op='multiply', out=out)

        return out

    def _call_pyfftw(self, x, out, **kwargs):
        """Implement ``self(x[, out, **kwargs])`` for pyfftw back-end.

        Parameters
        ----------
        x : `numpy.ndarray`
            Array representing the function to be transformed
        out : `numpy.ndarray`
            Array to which the output is written
        planning_effort : {'estimate', 'measure', 'patient', 'exhaustive'}
            Flag for the amount of effort put into finding an optimal
            FFTW plan. See the `FFTW doc on planner flags
            <http://www.fftw.org/fftw3_doc/Planner-Flags.html>`_.
        planning_timelimit : `float`, optional
            Limit planning time to roughly this amount of seconds.
            Default: `None` (no limit)
        threads : `int`, optional
            Number of threads to use. Default: 1

        Returns
        -------
        out : `numpy.ndarray`
            Result of the transform. If ``out`` was given, the returned
            object is a reference to it.
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(out, np.ndarray)

        # We pop some kwargs options here so that we always use the ones
        # given during init or implicitly assumed.
        kwargs.pop('axes', None)
        kwargs.pop('halfcomplex', None)
        kwargs.pop('normalise_idft', None)  # We use 'False'

        # Pre-processing before calculating the sums
        preproc = dft_preprocess_data(
            x, shift=self.shifts, axes=self.axes, sign=self.sign)

        # The actual call to the FFT library. We store the plan for re-use.
        direction = 'forward' if self.sign == '-' else 'backward'
        self._fftw_plan = pyfftw_call(
            preproc, out, direction=direction, halfcomplex=self.halfcomplex,
            axes=self.axes, normalise_idft=False, **kwargs)

        # Post-processing accounting for shift, scaling and interpolation
        dft_postprocess_data(
            out, real_grid=self.domain.grid, recip_grid=self.range.grid,
            shifts=self.shifts, axes=self.axes, sign=self.sign,
            interp=self.domain.interp, op='multiply', out=out)

        return out

    @property
    def impl(self):
        """Backend for the FFT implementation."""
        return self._impl

    @property
    def axes(self):
        """Axes along the FT is calculated by this operator."""
        return self._axes

    @property
    def sign(self):
        """Sign of the complex exponent in the transform."""
        return self._sign

    @property
    def halfcomplex(self):
        """Return `True` if the last transform axis is halved."""
        return self._halfcomplex

    @property
    def shifts(self):
        """Return the boolean list indicating shifting per axis."""
        return self._shifts

    @property
    def adjoint(self):
        """The adjoint Fourier transform."""
        if self.domain.field == RealNumbers():
            raise NotImplementedError(
                'Fourier transform from real to complex space has no adjoint.')
        elif self.domain.exponent == 2.0 and self.range.exponent == 2.0:
            return self.inverse
        else:
            raise NotImplementedError(
                'no adjoint defined for exponents ({}, {}) != (2, 2).'
                ''.format(self.domain.exponent, self.range.exponent))

    @property
    def inverse(self):
        """The inverse Fourier transform."""
        sign = '+' if self.sign == '-' else '-'
        return FourierTransformInverse(
            dom=self.range, ran=self.domain, impl=self.impl, axes=self.axes,
            halfcomplex=self.halfcomplex, shift=self.shifts, sign=sign)

    def init_fftw_plan(self, planning_effort='measure', **kwargs):
        """Initialize the FFTW plan for this transform for later use.

        If the implementation of this operator is not 'pyfftw', this
        method has no effect.

        Parameters
        ----------
        planning_effort : {'estimate', 'measure', 'patient', 'exhaustive'}
            Flag for the amount of effort put into finding an optimal
            FFTW plan. See the `FFTW doc on planner flags
            <http://www.fftw.org/fftw3_doc/Planner-Flags.html>`_.
        planning_timelimit : `float`, optional
            Limit planning time to roughly this amount of seconds.
            Default: `None` (no limit)
        threads : `int`, optional
            Number of threads to use. Default: 1
        """
        if self.impl != 'pyfftw':
            return

        x = self.domain.element()
        y = self.range.element()
        kwargs.pop('planning_timelimit', None)

        direction = 'forward' if self.sign == '-' else 'backward'
        self._fftw_plan = pyfftw_call(
            x.asarray(), y.asarray(), direction=direction,
            halfcomplex=self.halfcomplex, axes=self.axes,
            planning_effort=planning_effort, **kwargs)

    def clear_fftw_plan(self):
        """Delete the FFTW plan of this transform."""
        self._fftw_plan = None


class FourierTransformInverse(FourierTransform):

    """Inverse of the discretized Fourier transform between L^p spaces.

    This operator is the exact inverse of the `FourierTransform`, and
    **not** a discretization of the Fourier integral with "+" sign in
    the complex exponent. For the latter, use the ``sign`` parameter
    of the forward transform.

    See also
    --------
    FourierTransform
    """

    def __init__(self, ran, dom=None, impl='numpy', **kwargs):
        """
        Parameters
        ----------
        ran : `DiscreteLp`
            Range of the inverse Fourier transform. If the
            `DiscreteLp.exponent` of ``dom`` and ``ran`` are equal
            to 2.0, this operator has an adjoint which is equal to its
            inverse.
        dom : `DiscreteLp`, optional
            Domain of the inverse Fourier transform. If not given, the
            domain is determined from ``ran`` and the other parameters.
            The exponent is chosen to be the conjugate ``p / (p - 1)``,
            which reads as 'inf' for p=1 and 1 for p='inf'.
        impl : {'numpy', 'pyfftw'}
            Backend for the FFT implementation. The 'pyfftw' backend
            is faster but requires the ``pyfftw`` package.
        axes : sequence of `int`, optional
            Dimensions along which to take the transform.
            Default: all axes
        sign : {'-', '+'}, optional
            Sign of the complex exponent. Default: '+'
        halfcomplex : `bool`, optional
            If `True`, calculate only the negative frequency part
            along the last axis for real input. If `False`,
            calculate the full complex FFT.
            For complex ``dom``, it has no effect.
            Default: `True`
        shift : `bool` or sequence of `bool`, optional
            If `True`, the reciprocal grid is shifted by half a stride in
            the negative direction. With a boolean sequence, this option
            is applied separately to each axis.
            If a sequence is provided, it must have the same length as
            ``axes`` if supplied. Note that this must be set to `True`
            in the halved axis in half-complex transforms.
            Default: `True`

        Notes
        -----
        The `Operator.range` of this operator always has the
        `ComplexNumbers` as its `LinearSpace.field`, i.e. if the
        field of ``dom`` is the `RealNumbers`, this operator has no
        `Operator.adjoint`.
        """
        # TODO: variants wrt placement of 2*pi

        # Use initializer of parent and switch roles of domain and range.
        # Need to switch sign back and forth to check for the correct error
        # scenarios.
        if 'sign' in kwargs:
            bwd_sign = kwargs['sign']
            fwd_sign = '-' if kwargs['sign'] == '+' else '+'
        else:
            bwd_sign = '+'
            fwd_sign = '-'
        kwargs['sign'] = fwd_sign
        super().__init__(dom=ran, ran=dom, impl=impl, **kwargs)
        self._domain, self._range = self._range, self._domain
        self._sign = bwd_sign

    def _call_numpy(self, x):
        """Return ``self(x)`` for numpy back-end.

        Parameters
        ----------
        x : `numpy.ndarray`
            Array representing the function to be transformed

        Returns
        -------
        out : `numpy.ndarray`
            Result of the transform. If ``out`` was given, the returned
            object is a reference to it.
        """

        # Pre-processing in IFT = post-processing in FT, but with division
        # instead of multiplication and switched grids
        preproc = dft_postprocess_data(
            x, real_grid=self.range.grid, recip_grid=self.domain.grid,
            shifts=self.shifts, axes=self.axes, sign=self.sign,
            interp=self.domain.interp, op='divide')

        # The actual call to the FFT library
        # Normalization by 1 / prod(shape[axes]) is done by Numpy's FFT if
        # one of the "i" functions is used. For sign='-' we need to do it
        # ourselves.
        if self.halfcomplex:
            out = np.fft.irfftn(preproc, axes=self.axes)
        else:
            if self.sign == '-':
                out = np.fft.fftn(preproc, axes=self.axes)
                out /= np.prod(np.take(self.domain.shape, self.axes))
            elif self.sign == '+':
                out = np.fft.ifftn(preproc, axes=self.axes)
            else:
                raise RuntimeError('bad sign {}.'.format(self.sign))

        # Post-processing in IFT = pre-processing in FT
        preproc = dft_preprocess_data(
            out, shift=self.shifts, axes=self.axes, sign=self.sign, out=out)

        return out

    def _call_pyfftw(self, x, out, **kwargs):
        """Implement ``self(x[, out, **kwargs])`` for pyfftw back-end.

        Parameters
        ----------
        x : `numpy.ndarray`
            Array representing the function to be transformed
        out : `numpy.ndarray`
            Array to which the output is written
        planning_effort : {'estimate', 'measure', 'patient', 'exhaustive'}
            Flag for the amount of effort put into finding an optimal
            FFTW plan. See the `FFTW doc on planner flags
            <http://www.fftw.org/fftw3_doc/Planner-Flags.html>`_.
        planning_timelimit : `float`, optional
            Limit planning time to roughly this amount of seconds.
            Default: `None` (no limit)
        threads : `int`, optional
            Number of threads to use. Default: 1

        Returns
        -------
        out : `numpy.ndarray`
            Result of the transform. If ``out`` was given, the returned
            object is a reference to it.
        """

        # We pop some kwargs options here so that we always use the ones
        # given during init or implicitly assumed.
        kwargs.pop('axes', None)
        kwargs.pop('halfcomplex', None)
        kwargs.pop('normalise_idft', None)  # We use 'True'

        # Pre-processing in IFT = post-processing in FT, but with division
        # instead of multiplication and switched grids
        preproc = dft_postprocess_data(
            x, real_grid=self.range.grid, recip_grid=self.domain.grid,
            shifts=self.shifts, axes=self.axes, sign=self.sign,
            interp=self.domain.interp, op='divide')

        # The actual call to the FFT library. We store the plan for re-use.
        direction = 'forward' if self.sign == '-' else 'backward'
        self._fftw_plan = pyfftw_call(
            preproc, out, direction=direction, halfcomplex=self.halfcomplex,
            axes=self.axes, normalise_idft=True, **kwargs)

        # Normalization is only done for 'backward', we need it for 'forward',
        # too.
        if self.sign == '-':
            out /= np.prod(np.take(self.domain.shape, self.axes))

        # Post-processing in IFT = pre-processing in FT
        preproc = dft_preprocess_data(
            out, shift=self.shifts, axes=self.axes, sign=self.sign, out=out)

        return out

    @property
    def inverse(self):
        """Inverse of the inverse, the forward FT."""
        sign = '+' if self.sign == '-' else '-'
        return FourierTransform(
            dom=self.range, ran=self.domain, impl=self.impl, axes=self.axes,
            halfcomplex=self.halfcomplex, shift=self.shifts, sign=sign)


def _shift_list(shift, length):
    """Turn a single boolean or sequence into a list of given length."""
    try:
        shift_list = [bool(s) for s in shift]
        if len(shift_list) != length:
            raise ValueError('Expected {} entries in shift sequence, got {}.'
                             ''.format(length, len(shift_list)))
    except TypeError:
        shift_list = [bool(shift)] * length

    return shift_list


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
