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
    DiscreteLp, dspace_type, conj_exponent, uniform_discr)
from odl.operator.operator import Operator
from odl.set.sets import RealNumbers, ComplexNumbers
from odl.space.ntuples import Ntuples
from odl.space.fspace import FunctionSpace
from odl.util.utility import is_real_dtype, fast_1d_tensor_mult


__all__ = ('FourierTransform', 'InverseFourierTransform',
           'pyfftw_call', 'dft_preprocess_data', 'dft_postprocess_data',
           'PYFFTW_AVAILABLE')


_TYPE_MAP_R2C = {np.dtype(dtype): np.result_type(dtype, 1j)
                 for dtype in np.sctypes['float']}


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


def reciprocal(grid, shift=True, axes=None, halfcomplex=False):
    """Return the reciprocal of the given regular grid.

    This function calculates the reciprocal (Fourier/frequency space)
    grid for a given regular grid defined by the nodes
    ::
        x[k] = x[0] + k * s,

    where ``k = (k[0], ..., k[d-1])`` is a ``d``-dimensional index in
    the range ``0 <= k < N`` (component-wise). The multi-index
    ``N`` is the shape of the input grid.
    This grid's reciprocal is then given by the nodes
    ::
        xi[j] = xi[0] + j * sigma,

    with the reciprocal grid stride ``sigma = 2*pi / (s * N)``.
    The minimum frequency ``xi[0]`` can in principle be chosen
    freely, but usually it is chosen in a such a way that the reciprocal
    grid is centered around zero. For this, there are two possibilities:

    1. Make the grid point-symmetric around 0.

    2. Make the grid "almost" point-symmetric around zero by shifting
       it to the left by half a reciprocal stride.

    In the first case, the minimum frequency (per axis) is given as
    ::
        xi_1[0] = -pi/s + pi/(s*n) = -pi/s + sigma/2.

    For the second case, it is
    ::
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
    recip : `odl.RegularGrid`
        The reciprocal grid
    """
    if axes is None:
        axes = list(range(grid.ndim))

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

    # TODO: specify as_midp per axis, not supported currently
    return RegularGrid(rmin, rmax, rshape, as_midp=False)


def inverse_reciprocal(grid, x0, axes=None, halfcomplex=False,
                       halfcx_parity='even'):
    """Return the inverse reciprocal of the given regular grid.

    Given a reciprocal grid
    ::
        xi[j] = xi[0] + j * sigma,

    with a multi-index ``j = (j[0], ..., j[d-1])`` in the range
    ``0 <= j < M``, this function calculates the original grid
    ::
        x[k] = x[0] + k * s

    by using a provided ``x[0]`` and calculating the stride ``s``.

    If the reciprocal grid is interpreted as coming from a usual
    complex-to-complex FFT, it is ``N == M``, and the stride is
    ::
        s = 2*pi / (sigma * N)

    For a reciprocal grid from a real-to-complex (half-complex) FFT,
    it is ``M[i] = floor(N[i]/2) + 1`` in the last transform axis ``i``.
    To resolve the ambiguity regarding the parity of ``N[i]``, the
    it must be specified if the output shape should be even or odd,
    resulting in
    ::
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
    irecip : `odl.RegularGrid`
        The inverse reciprocal grid
    """
    if axes is None:
        axes = list(range(grid.ndim))

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

    # TODO: specify as_midp per axis, not supported currently
    return RegularGrid(irmin, irmax, irshape, as_midp=True)


def dft_preprocess_data(dfunc, shift=True, axes=None):
    """Pre-process the real-space data before DFT.

    This function multiplies the given data with the separable
    function
    ::
        p(x) = exp(-1j * dot(x - x[0], xi[0]))

    where ``x[0]`` and ``xi[0]`` are the minimum coodinates of
    the real space and reciprocal grids, respectively. In discretized
    form, this function becomes for an array
    ::
        p[k] = exp(-1j * k * s * xi[0])

    If the reciprocal grid is not shifted, i.e. symmetric around 0,
    it is ``xi[0] =  pi/s * (-1 + 1/N)``, hence
    ::
        p[k] = exp(1j * pi * k * (1 - 1/N))

    For a shifted grid, we have :math:``xi[0] =  -pi/s``, thus the
    array is given by
    ::
        p[k] = (-1)**k

    Parameters
    ----------
    dfunc : `DiscreteLpVector`
        Discrete function to be pre-processed. Changes are made
        in place. For real input data, this is only possible if
        ``shift=True`` since the factors :math:`p_k` are real only
        in this case.
    shift : `bool` or sequence of `bool`, optional
        If `True`, the grid is shifted by half a stride in the negative
        direction. With a sequence, this option is applied separately on
        each axis.
    axes : sequence of `int`, optional
        Dimensions in which to calculate the reciprocal. The sequence
        must have the same length as ``shift`` if the latter is given
        as a sequence. `None` means all axes in ``dfunc``.
    """
    if dfunc.space.field == RealNumbers() and not shift:
        raise ValueError('cannot pre-process in-place without shift.')

    if axes is None:
        axes = list(range(dfunc.ndim))

    shape = dfunc.space.grid.shape
    shift_list = _shift_list(shift, dfunc.ndim)

    def _onedim_arr(length, shift):
        if shift:
            # (-1)^indices
            indices = np.arange(length, dtype='int8')
            arr = -2 * np.mod(indices, 2) + 1
        else:
            indices = np.arange(length, dtype='float64')
            arr = np.exp(1j * pi * indices * (1 - 1.0 / length))
        return arr

    onedim_arrs = []
    for axis in axes:
        shift = shift_list[axis]
        length = shape[axis]
        onedim_arrs.append(_onedim_arr(length, shift))

    fast_1d_tensor_mult(dfunc.asarray(), onedim_arrs, axes=axes)


def _interp_kernel_ft(norm_freqs, interp):
    """Scaled FT of a one-dimensional interpolation kernel.

    For normalized frequencies ``-1/2 <= xi <= 1/2``, this
    function returns
    ::
        sinc(pi * xi)**k / sqrt(2 * pi)

    where ``k=1`` for 'nearest', ``k=2`` for 'linear' and ``k=3``
    for 'cubic' interpolation.

    Parameters
    ----------
    norm_freqs : `numpy.ndarray`
        Normalized frequencies between -1/2 and 1/2
    interp : {'nearest', 'linear', 'cubic'}
        Type of interpolation kernel

    Returns
    -------
    ker_ft : `numpy.ndarray`
        Values of the kernel FT at the given frequencies
    """
    # Numpy's sinc(x) is equal to the 'math' sinc(pi * x)
    if interp == 'nearest':
        return np.sinc(norm_freqs) / np.sqrt(2 * np.pi)
    elif interp == 'linear':
        return np.sinc(norm_freqs) ** 2 / np.sqrt(2 * np.pi)
    elif interp == 'cubic':
        return np.sinc(norm_freqs) ** 3 / np.sqrt(2 * np.pi)
    else:  # Shouldn't happen
        raise RuntimeError


def dft_postprocess_data(dfunc, x0, shifts, axes, orig_shape, orig_stride,
                         interp):
    """Post-process the Fourier-space data after DFT.

    This function multiplies the given data with the separable
    function
    ::
        q(xi) = exp(-1j * dot(x[0], xi)) * s * phi_hat(xi_bar)

    where ``x[0]`` and ``s`` are the minimum point and the stride of
    the real space grid, respectively, and ``phi_hat(xi_bar)`` is the FT
    of the interpolation kernel. In discretized form, the exponential
    part of this function becomes an array
    ::
        q[k] = exp(-1j * dot(x[0], xi[k]))

    and the arguments ``xi_bar`` to the interpolation kernel
    are the normalized frequencies
    ::
        for ``shift=True`` : xi_bar[k] = -pi + pi * (2*k) / N
        for ``shift=False``: xi_bar[k] = -pi + pi * (2*k+1) / N

    See [1]_, Section 13.9 "Computing Fourier Integrals Using the FFT"
    for a similar approach.

    Parameters
    ----------
    dfunc : `DiscreteLpVector`
        Discrete function to be post-processed. Its grid is assumed
        to be the reciprocal grid. Changes are made in place.
    x0 : array-like
        Minimal grid point of the spatial grid before transform
    shifts : sequence of `bool`
        If `True`, the grid is shifted by half a stride in the negative
        direction in the corresponding axes. The sequence must have the
        same length as ``axes``.
    axes : sequence of `int`
        Dimensions along which to take the transform. The sequence must
        have the same length as ``shifts``.
    orig_shape : sequence of positive `int`
        Shape of the original array
    orig_stride : sequence of positive `float`
        Stride of the original array
    interp : `str`
        Interpolation scheme used in real space

    References
    ----------
    .. [1] Press, William H, Teukolsky, Saul A, Vetterling, William T,
       and Flannery, Brian P. *Numerical Recipes in C - The Art of
       Scientific Computing* (Volume 3). Cambridge University Press,
       2007.
    """
    # Reciprocal grid
    rgrid = dfunc.space.grid
    shift_list = list(shifts)
    axes = list(axes)

    onedim_arrs = []
    for ax in axes:
        x = x0[ax]
        xi = rgrid.coord_vectors[ax]

        # First part: exponential array
        onedim_arr = (np.exp(-1j * x * xi))

        # Second part: interpolation kernel
        len_dft = rgrid.shape[ax]
        len_orig = orig_shape[ax]
        halfcomplex = (len_dft < len_orig)
        odd = len_orig % 2
        shift = shift_list[ax]

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
                # Always -0.5 + (N-1)/N = 0.5 - 1/N
                fmax = 0.5 - 1.0 / (2 * len_orig)

        freqs = np.linspace(fmin, fmax, num=len_dft)

        stride = orig_stride[ax]
        onedim_arr *= stride * _interp_kernel_ft(freqs, interp)
        onedim_arrs.append(onedim_arr)

    if dfunc.space.order == 'C':
        mult_axes = axes
    else:
        mult_axes = list(reversed(axes))

    fast_1d_tensor_mult(dfunc.asarray(), onedim_arrs, axes=mult_axes)


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
            out_shape[axes[-1]] = arr_in.shape[axes[-1]] // 2 + 1

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
            raise TypeError('Expected output dtype {}, got {}.'
                            ''.format(out_dtype, arr_out.dtype))

    elif direction == 'backward':
        in_shape = list(arr_out.shape)
        if halfcomplex:
            in_shape[axes[-1]] = arr_out.shape[axes[-1]] // 2 + 1

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
            raise TypeError('Expected input dtype {}, got {}.'
                            ''.format(in_dtype, arr_in.dtype))

    else:  # Shouldn't happen
        raise RuntimeError


def pyfftw_call(array_in, array_out, direction='forward', axes=None,
                halfcomplex=False, **kwargs):
    """Calculate the DFT with pyfftw.

    The discrete Fourier (forward) transform calcuates the sum
    ::
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
    fftw_plan = kwargs.pop('fftw_plan', None)
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
    must_copy_array_in = fftw_plan is None and planner_destroys

    if must_copy_array_in and not array_in_copied:
        plan_arr_in = np.empty_like(array_in)
        flags = [_local_to_pyfftw(planning_effort), 'FFTW_DESTROY_INPUT']
    else:
        plan_arr_in = array_in
        flags = [_local_to_pyfftw(planning_effort)]

    if fftw_plan is None:
        if threads is None:
            if plan_arr_in.size < 1000:
                threads = 1
            else:
                threads = cpu_count()

        fft_plan = pyfftw.FFTW(
            plan_arr_in, array_out, direction=_local_to_pyfftw(direction),
            flags=flags, planning_timelimit=planning_timelimit,
            threads=threads, axes=axes)

    fft_plan(array_in, array_out, normalise_idft=normalise_idft)

    if wexport:
        try:
            with open(wexport, 'ab') as wfile:
                pickle.dump(pyfftw.export_wisdom(), wfile)
        except TypeError:  # Got file handle
            pickle.dump(pyfftw.export_wisdom(), wexport)

    return fftw_plan


def _recip_space(spc, axes, halfcomplex):
    """Return the reciprocal space of ``spc`` with unit stride."""
    # Calculate reciprocal space with a grid with stride 1 and min (0, ..., 0)

    # Just to get the shape right
    shape = reciprocal(
        spc.grid, shift=False, halfcomplex=halfcomplex, axes=axes).shape

    if is_real_dtype(spc.dtype):
        rspc_dtype = _TYPE_MAP_R2C[spc.dtype]
    else:
        rspc_dtype = spc.dtype

    rspc = uniform_discr([0] * spc.ndim, shape, shape, dtype=rspc_dtype,
                         as_midp=False)
    return rspc


class PyfftwTransform(Operator):

    """Plain forward DFT as implemented in ``pyfftw``.

    This operator calculates the forward DFT
    ::
        f_hat[k] = sum_j( f[j] * exp(-2*pi*1j * j*k/N) )

    without any shifting or scaling compensation. See the
    `pyfftw API documentation`_ and `What FFTW really computes`_
    for further information.

    The domain and range of this operator are both `DiscreteLp`
    spaces with exponent 2 and sampling grid stride 1.

    See also
    --------
    pyfftw_call : apply an FFTW transform

    References
    ----------
    .. _pyfftw API documentation:
       http://hgomersall.github.io/pyFFTW/pyfftw/pyfftw.html
    .. _What FFTW really computes:
       http://www.fftw.org/fftw3_doc/What-FFTW-Really-Computes.html
    """
    def __init__(self, dom_shape, dom_dtype='complex',
                 axes=(-1,), halfcomplex=False):
        """Initialize a new instance.

        Parameters
        ----------
        dom_shape : sequence of positive `int`
            Number of sampling points per axis. This determines the
            domain of the operator.
        dom_dtype : optional
            Data type of the input arrays
        axes : sequence of `int`, optional
            Dimensions in which a transform is to be calculated
        halfcomplex : `bool`, optional
            If `True`, calculate only the negative frequency part
            along the last axis in ``axes`` for real input. This
            reduces the size of the range to ``floor(N[i]/2) + 1`` in
            this axis ``i``, where ``N`` is the shape of the input
            arrays.
            Otherwise, calculate the full complex FFT. If ``dom_dtype``
            is a complex type, this option has no effect.

        Examples
        --------
        Complex-to-complex (default) transforms have the same grids
        in domain and range:

        >>> dom_shape = (2, 4)
        >>> fft = PyfftwTransform(dom_shape)
        >>> fft.domain.shape
        (2, 4)
        >>> fft.range.shape
        (2, 4)

        Real-to-complex transforms have a range grid with shape
        ``n // 2 + 1`` in the last tranform axis:

        >>> dom_shape = (2, 3, 4)
        >>> axes = (0, 1)
        >>> fft = PyfftwTransform(
        ...     dom_shape, dom_dtype='float', halfcomplex=True,
        ...     axes=axes)
        >>> fft.domain.shape   # shortened in the second axis
        (2, 2, 4)
        >>> fft.range.shape
        (2, 3, 4)
        """
        # TODO: add option ran_shape to allow zero-padding
        if not np.all(np.array(dom_shape) > 0):
            raise ValueError('invalid entries in dom_shape {}.'
                             ''.format(dom_shape))

        # Domain is a DiscreteLp with stride (1, ..., 1)
        dom_shape = tuple(np.atleast_1d(dom_shape))
        # TODO: as_midp is deprecated, use partition instead
        dom = uniform_discr([0] * len(dom_shape), dom_shape, dom_shape,
                            dtype=dom_dtype, as_midp=False)

        axes_ = np.atleast_1d(axes)
        axes_[axes_ < 0] += len(dom_shape)
        if not all(0 <= ax < len(dom_shape) for ax in axes_):
            raise ValueError('invalid entries in axes.')
        self._axes = list(axes_)

        if dom.field == ComplexNumbers():
            self._halfcomplex = False
        else:
            self._halfcomplex = bool(halfcomplex)

        ran = _recip_space(dom, self.axes, self.halfcomplex)
        super().__init__(dom, ran, linear=True)
        self._fftw_plan = None

    def _call(self, x, out, **kwargs):
        """Implement ``self(x[, out, **kwargs])``.

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
        kwargs.pop('normalise_idft', None)  # Not used here, filtering out
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

        self._fftw_plan = pyfftw_call(
            x.asarray(), out.asarray(), direction='forward', axes=self.axes,
            halfcomplex=self.halfcomplex, planning_effort=effort,
            fftw_plan=self._fftw_plan, **kwargs)

        # TODO: Implement zero padding

    @property
    def axes(self):
        """Axes along the FT is calculated by this operator."""
        return self._axes

    @property
    def halfcomplex(self):
        """Return `True` if the last transform axis is halved."""
        return self._halfcomplex

    @property
    def adjoint(self):
        """Adjoint transform, equal to the inverse."""
        return self.inverse

    @property
    def inverse(self):
        """Inverse Fourier transform."""
        return PyfftwTransformInverse(
            self.domain.shape, self.domain.dtype,
            axes=self.axes, halfcomplex=self.halfcomplex)


class PyfftwTransformInverse(Operator):

    """Plain backward DFT as implemented in ``pyfftw``.

    This operator calculates the inverse DFT
    ::
        f[k] = 1/prod(N) * sum_j( f_hat[j] * exp(2*pi*1j * j*k/N) )

    without any further shifting or scaling compensation. See the
    `pyfftw API documentation`_ and `What FFTW really computes`_
    for further information.

    The domain and range of this operator are both `DiscreteLp`
    spaces with exponent 2 and sampling grid stride 1.

    See also
    --------
    pyfftw_call : apply an FFTW transform

    References
    ----------
    .. _pyfftw API documentation:
       http://hgomersall.github.io/pyFFTW/pyfftw/pyfftw.html
    .. _What FFTW really computes:
       http://www.fftw.org/fftw3_doc/What-FFTW-Really-Computes.html
    """
    def __init__(self, ran_shape, ran_dtype='complex', axes=(-1,),
                 halfcomplex=False):
        """Initialize a new instance.

        Parameters
        ----------
        ran_shape : sequence of positive `int`
            Number of sampling points per axis. This determines the
            range of the operator.
        ran_dtype : optional
            Data type of the output arrays
        axes : sequence of `int`, optional
            Dimensions in which a transform is to be calculated
        halfcomplex : `bool`, optional
            If `True`, interpret the last axis in ``axes`` as the
            negative frequency part of the transform of a real signal
            and calculate a "half-complex-to-real" inverse FFT. In this
            case, the domain has shape ``floor(N[i]/2) + 1`` in this
            axis ``i``.
            Otherwise, domain and range have the same shape. If
            ``ran_dtype`` is a complex type, this option has no effect.

        Examples
        --------
        Complex-to-complex (default) transforms have the same grids
        in domain and range:

        >>> ran_shape = (2, 4)
        >>> ifft = PyfftwTransformInverse(ran_shape)
        >>> ifft.domain.shape
        (2, 4)
        >>> ifft.range.shape
        (2, 4)

        Complex-to-real transforms have a domain grid with shape
        ``n // 2 + 1`` in the last tranform axis:

        >>> ran_shape = (2, 3, 4)
        >>> axes = (0, 1)
        >>> ifft = PyfftwTransformInverse(
        ...     ran_shape, ran_dtype='float', halfcomplex=True,
        ...     axes=axes)
        >>> ifft.domain.shape   # shortened in the second axis
        (2, 2, 4)
        >>> ifft.range.shape
        (2, 3, 4)
        """
        # TODO: add option dom_shape to allow zero-padding
        if not np.all(np.array(ran_shape) > 0):
            raise ValueError('invalid entries in dom_shape {}.'
                             ''.format(ran_shape))

        # Range is a DiscreteLp with stride (1, ..., 1)
        ran_shape = tuple(np.atleast_1d(ran_shape))
        # TODO: as_midp is deprecated, use partition instead
        ran = uniform_discr([0] * len(ran_shape), ran_shape, ran_shape,
                            dtype=ran_dtype, as_midp=False)

        axes_ = np.atleast_1d(axes)
        axes_[axes_ < 0] += len(ran_shape)
        if not all(0 <= ax < len(ran_shape) for ax in axes_):
            raise ValueError('invalid entries in axes.')
        self._axes = list(axes_)

        if ran.field == ComplexNumbers():
            self._halfcomplex = False
        else:
            self._halfcomplex = bool(halfcomplex)

        # self._halfcomplex and self._axes need to be set for this
        dom = _recip_space(ran, self.axes, self.halfcomplex)
        super().__init__(dom, ran, linear=True)
        self._fftw_plan = None

    def _call(self, x, out, **kwargs):
        """Implement ``self(x[, out, **kwargs])``.

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
        kwargs.pop('normalise_idft', None)  # Always using True here
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

        self._fftw_plan = pyfftw_call(
            x.asarray(), out.asarray(), direction='backward', axes=self.axes,
            halfcomplex=self.halfcomplex, planning_effort=effort,
            normalise_idft=True, fftw_plan=self._fftw_plan, **kwargs)

        # TODO: Implement zero padding

    @property
    def axes(self):
        """Axes along the FT is calculated by this operator."""
        return self._axes

    @property
    def halfcomplex(self):
        """Return `True` if the last transform axis is halved."""
        return self._halfcomplex

    @property
    def adjoint(self):
        """Adjoint transform, equal to the inverse."""
        return self.inverse

    @property
    def inverse(self):
        """Inverse Fourier transform."""
        return PyfftwTransform(
            self.range.shape, self.range.dtype,
            axes=self.axes, halfcomplex=self.halfcomplex)


class FourierTransform(Operator):

    """Discretized Fourier transform between discrete Lp spaces.

    The Fourier transform is defined as the linear operator

        :math:`\mathcal{F}: L^p(\mathbb{R}^d) \\to L^q(\mathbb{R}^d)`,

        :math:`\mathcal{F}(\phi)(\\xi) = \widehat{\phi}(\\xi) =
        (2\pi)^{-\\frac{d}{2}}
        \int_{\mathbb{R}^d} \phi(x)\ e^{-i x^{\mathrm{T}}\\xi}
        \ \mathrm{d}x`,

    where :math:`p \geq 1` and :math:`q = p / (p-1)`. The Fourier
    transform is bounded for :math:`1 \\leq p \\leq 2` and can be
    reasonably defined for :math:`p > 2` in the distributional sense
    [1]_.
    Its inverse is given by the formula

        :math:`\mathcal{F^{-1}}(\phi)(x) = \widetilde{\phi}(\\xi) =
        (2\pi)^{-\\frac{d}{2}}
        \int_{\mathbb{R}^d} \phi(\\xi)\ e^{i \\xi^{\mathrm{T}}x}
        \ \mathrm{d}\\xi`,

    For :math:`p = 2`, it is :math:`q = 2`, and the inverse Fourier
    transform is the adjoint operator,
    :math:`\mathcal{F}^* = \mathcal{F}^{-1}`. Note that

        :math:`\mathcal{F^{-1}}(\phi) = \mathcal{F}(\check\phi)
        = \mathcal{F}(\phi)(-\cdot)
        = \overline{\mathcal{F}(\overline{\phi})}
        = \mathcal{F}^3(\phi), \quad \check\phi(x) = \phi(-x)`.

    This implies in particular that for real-valued :math:`\phi`,
    it is :math:`\overline{\mathcal{F}(\phi)}(\\xi) =
    \mathcal{F}(\phi)(-\\xi)`, i.e. the Fourier transform is completely
    known already from the its values in a half-space only. This
    property is used in the `halfcomplex storage format
    <http://fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html>`_.

    Further properties are summarized in `the Wikipedia article on
    the Fourier transform
    <https://en.wikipedia.org/wiki/Fourier_transform>`_.

    References
    ----------
    .. [1] Stein, Elias and Weiss, Guido (1971). Introduction to
       Fourier Analysis on Euclidean Spaces. Princeton, N.J.:
       Princeton University Press. ISBN 978-0-691-08078-9
    """

    def __init__(self, dom, ran=None, impl='numpy', **kwargs):
        """
        Parameters
        ----------
        dom : `DiscreteLp`
            Domain of the Fourier transform. Its
            :attr:`odl.DiscreteLp.exponent` must be at least 1.0;
            if it is equal to 2.0, this operator has an adjoint which
            is equal to the inverse.
        ran : `DiscreteLp`, optional
            Range of the Fourier transform. If not given, the range
            is determined from ``dom`` and the other parameters. The
            exponent is chosen to be the conjugate ``p / (p - 1)``,
            which is read as 'inf' for p=1 and 1 for p='inf'.
        axes : sequence of `int`, optional
            Dimensions along which to take the transform.
            Default: all axes
        halfcomplex : `bool`, optional
            If `True`, calculate only the negative frequency part
            along the last axis for real input. If `False`,
            calculate the full complex FFT.
            For complex ``dom``, it has no effect.
            Default: `True`

            This option only applies to 'uniform-to-uniform' transforms.

        shift : `bool` or sequence of `bool`, optional
            If `True`, the reciprocal grid is shifted by half a stride in
            the negative direction. With a boolean sequence, this option
            is applied separately to each axis.
            If a sequence is provided, it must have the same length as
            ``axes`` if supplied.
            Default: `True`

            This option only applies to 'uniform-to-uniform' transforms.

        Notes
        -----
        The `Operator.range` of this operator always has the
        `ComplexNumbers` as its `LinearSpace.field`, i.e. if the
        field of ``dom`` is the `RealNumbers`, this operator has no
        `Operator.adjoint`.
        """
        # TODO: variants wrt placement of 2*pi
        # TODO: impl flag for Numpy vs. PyFFTW

        if not isinstance(dom, DiscreteLp):
            raise TypeError('domain {!r} is not a `DiscreteLp` instance.'
                            ''.format(dom))
        if not isinstance(dom.dspace, Ntuples):
            raise NotImplementedError(
                'Only Numpy-based data spaces are supported, got {}.'
                ''.format(dom.dspace))

        self._axes = list(kwargs.pop('axes', range(dom.ndim)))

        # Check exponents
        if dom.exponent < 1:
            raise ValueError('domain exponent {} < 1 not allowed.'
                             ''.format(dom.exponent))

        # Half-complex yes/no and shifts
        if isinstance(dom.grid, RegularGrid):
            if dom.field == ComplexNumbers():
                self._halfcomplex = False
            else:
                self._halfcomplex = bool(kwargs.pop('halfcomplex', True))

            self._shifts = _shift_list(kwargs.pop('shift', True),
                                       length=len(self.axes))
        else:
            raise NotImplementedError('irregular grids not yet supported.')

        if ran is None:
            # self._halfcomplex and self._axes need to be set for this
            ran = self._conj_range(dom)

        super().__init__(dom, ran, linear=True)
        self._fftw_plan = None

    def _conj_range(self, dom):
        """Returned the conjugate range determined from ``dom``."""
        # Calculate range
        recip_grid = reciprocal(
            dom.grid, shift=self.shifts, halfcomplex=self.halfcomplex,
            axes=self.axes)

        # Always complex space
        ran_fspace = FunctionSpace(recip_grid.convex_hull(), ComplexNumbers())

        if is_real_dtype(dom.dtype):
            ran_dtype = _TYPE_MAP_R2C[dom.dtype]
        else:
            ran_dtype = dom.dtype

        conj_exp = conj_exponent(dom.exponent)
        ran_dspace_type = dspace_type(ran_fspace, impl='numpy',
                                      dtype=ran_dtype)
        ran_dspace = ran_dspace_type(recip_grid.size, dtype=ran_dtype,
                                     exponent=conj_exp)

        ran = DiscreteLp(ran_fspace, recip_grid, ran_dspace,
                         exponent=conj_exp)

        return ran

    def _call(self, x, out, **kwargs):
        """Implement ``self(x[, out, **kwargs])``.

        Parameters
        ----------
        x : domain element
            Discretized function to be transformed
        out : range element
            Element to which the output is written

        Notes
        -----
        See the `pyfftw_call` function for ``**kwargs`` options.

        See also
        --------
        pyfftw_call : Call pyfftw backend directly
        """
        self._call_pyfftw(x, out, **kwargs)

    def _call_numpy(self, x, out):
        """Implement ``self(x[, out, **kwargs])`` for numpy back-end."""
        # TODO: Implement this
        raise NotImplementedError

    def _call_pyfftw(self, x, out, **kwargs):
        """Implement ``self(x[, out, **kwargs])`` for pyfftw back-end.

        Parameters
        ----------
        x : domain element
            Discretized function to be transformed
        out : range element
            Element to which the output is written
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
        # TODO: Implement zero padding

        # We pop some kwargs options here so that we always use the ones
        # given during init or implicitly assumed.
        kwargs.pop('axes', None)
        kwargs.pop('halfcomplex', None)
        kwargs.pop('normalise_idft', None)

        # We're always modifying the input, so a copy is unavoidable
        x_cpy = x.copy()

        # Pre-processing before calculating the sums
        dft_preprocess_data(x_cpy, shift=self.shifts, axes=self.axes)

        # The actual call to the FFT library. We store the plan for re-use.
        self._fftw_plan = pyfftw_call(
            x_cpy.asarray(), out.asarray(), direction='forward',
            halfcomplex=self.halfcomplex, axes=self.axes, **kwargs)

        # Post-processing accounting for shift, scaling and interpolation
        dft_postprocess_data(out, self.domain.grid.min_pt, shifts=self.shifts,
                             axes=self.axes, orig_shape=self.domain.shape,
                             orig_stride=self.domain.grid.stride,
                             interp=self.domain.interp)
        return out

    @property
    def axes(self):
        """Axes along the FT is calculated by this operator."""
        return self._axes

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
        if self.domain.exponent == 2.0:
            return self.inverse
        else:
            raise NotImplementedError('adjoint only defined for exponent 2.0, '
                                      'not {}.'.format(self.domain.exponent))

    @property
    def inverse(self):
        """The inverse Fourier transform."""
        # TODO: add appropriate arguments
        raise NotImplementedError


class InverseFourierTransform(Operator):
    pass


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
