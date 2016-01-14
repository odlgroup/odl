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

"""Discrete wavelet transformation on :math:`L^p` spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import next, range, super

# External
from math import pi
import numpy as np
import platform
try:
    import pyfftw
    PYFFTW_AVAILABLE = True
except ImportError:
    pyfftw = None
    PYFFTW_AVAILABLE = False

# Internal
from odl.discr.grid import RegularGrid, sparse_meshgrid
from odl.discr.lp_discr import DiscreteLp, dspace_type
from odl.operator.operator import Operator
from odl.set.sets import RealNumbers, ComplexNumbers
from odl.space.ntuples import Ntuples
from odl.space.fspace import FunctionSpace
from odl.util.utility import is_real_dtype


__all__ = ('DiscreteFourierTransform', 'DiscreteFourierTransformInverse',
           'PYFFTW_AVAILABLE')


_TYPE_MAP_R2C = {np.dtype('float32'): np.dtype('complex64'),
                 np.dtype('float64'): np.dtype('complex128')}

if platform.system() == 'Linux':
    _TYPE_MAP_R2C[np.dtype('float128')] = np.dtype('complex256')


def _fftw_to_local(flag):
    return flag.lstrip('FFTW_').lower()


def _local_to_fftw(flag):
    return 'FFTW_' + flag.upper()


def _fftw_destroys_input(flags, direction, halfcomplex, ndim):
    """Return `True` if FFTW destroys an input array, `False` otherwise."""
    if any(flag in flags or _fftw_to_local(flag) in flags
           for flag in ('FFTW_MEASURE', 'FFTW_PATIENT', 'FFTW_EXHAUSTIVE',
                        'FFTW_DESTROY_INPUT')):
        return True
    elif (direction in ('backward', 'FFTW_BACKWARD') and halfcomplex and
          ndim != 1):
        return True
    else:
        return False


def _shift_list(shift, length):
    """Turn a single boolean or iterable into a list of given length."""
    try:
        shift = iter(shift)
        shift_list = []
        try:
            for i in range(length):
                shift_list.append(next(shift))
        except StopIteration:
            pass
    except TypeError:  # single boolean
        shift_list = [bool(shift)] * length

    shift_list = shift_list[:length]
    if len(shift_list) < length:
        raise ValueError('boolean shift list or iterable gives too few '
                         'entries ({} < {}).'.format(len(shift_list), length))

    return shift_list


def reciprocal(grid, shift=True, halfcomplex=False):
    """Return the reciprocal of the given regular grid.

    This function calculates the reciprocal (Fourier/frequency space)
    grid for a given regular grid defined by the nodes

        :math:`x_k = x_0 + k \odot s, \quad k \\in I_N`

    with an index set :math:`I_N = \{k\\in\mathbb{Z}\ |\ 0\\leq k< N\}`.
    The multiplication ":math:`\odot`" and the comarisons are to be
    understood component-wise.

    This grid's reciprocal is then given by the nodes

        :math:`\\xi_j = \\xi_0 + j \odot \sigma, \quad
        \sigma = 2\pi (s \odot N)^{-1}`.

    The minimum frequency :math:`\\xi_0` can in principle be chosen
    freely, but usually it is chosen in a such a way that the reciprocal
    grid is centered around zero. For this, there are two possibilities:

    1. Make the grid point-symmetric around 0.

    2. Make the grid "almost" point-symmetric around zero by shifting
       it to the left by half a reciprocal stride.

    In the first case, the minimum frequency (per axis) is given as

        :math:`\\xi_{0, \\text{symm}} = -\pi / s + \pi / (sn) =
        -\pi / s + \sigma/2`.

    For the second case, it is:

        :math:`\\xi_{0, \\text{shift}} = -\pi / s.`

    Note that the zero frequency is contained in case 1 for an odd
    number of points, while for an even size, the second option
    guarantees that 0 is contained.

    Parameters
    ----------
    grid : `odl.RegularGrid`
        Original sampling grid
    shift : `bool` or iterable, optional
        If `True`, the grid is shifted by half a stride in the negative
        direction.
        With a boolean array or iterable, this option is applied
        separately on each axis. At least ``grid.ndim`` values must be
        provided.
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
    shift_lst = _shift_list(shift, grid.ndim)

    rmin = np.empty_like(grid.min_pt)
    rmax = np.empty_like(grid.max_pt)
    rsamples = list(grid.shape)

    stride = grid.stride
    shape = np.array(grid.shape)

    # Shifted axes
    shift = np.where(shift_lst)
    rmin[shift] = -pi / stride[shift]
    # Length min->max increases by double the shift, so we
    # have to compensate by a full stride
    rmax[shift] = (-rmin[shift] - 2 * pi / (stride[shift] * shape[shift]))

    # Non-shifted axes
    no_shift = np.where(np.logical_not(shift_lst))
    rmin[no_shift] = (-1.0 + 1.0 / shape[no_shift]) * pi / stride[no_shift]
    rmax[no_shift] = -rmin[no_shift]

    # Change last axis shape and max if halfcomplex
    if halfcomplex:
        rsamples[-1] = shape[-1] // 2 + 1

        # - Odd and not shifted or even and shifted -> 0
        # - Odd and shifted -> - stride / 2
        # - Even and not shifted -> + stride / 2
        odd = shape[-1] % 2 == 1
        shifted = shift_lst[-1]
        half_rstride = pi / (shape[-1] * stride[-1])

        if odd:
            if shifted:
                rmax[-1] = -half_rstride
            else:
                rmax[-1] = 0
        else:
            if shifted:
                rmax[-1] = 0
            else:
                rmax[-1] = half_rstride

    return RegularGrid(rmin, rmax, rsamples, as_midp=False)


def dft_preproc_data(dfunc, shift=True):
    """Pre-process the real-space data before DFT.

    This function multiplies the given data with the separable
    function

        :math:`p(x) = e^{-i(x-x_0)^{\mathrm{T}}\\xi_0},`

    where :math:`x_0` :math:`\\xi_0` are the minimum coodinates of
    the real space and reciprocal grids, respectively. In discretized
    form, this function becomes for each axis separately an array

        :math:`p_k = e^{-i k (s \\xi_0)}.`

    If the reciprocal grid is symmetric, it is
    :math:`\\xi_0 =  \pi/s (-1 + 1/N)`, hence

        :math:`p_{k, \\text{symm}} = e^{i \pi k (1-1/N)}.`

    For a shifted grid, we have :math:`\\xi_0 =  -\pi/s`, thus the array
    is given by

        :math:`p_{k, \\text{shift}} = e^{i \pi k} = (-1)^k.`

    Parameters
    ----------
    dfunc : `DiscreteLpVector`
        Discrete function to be pre-processed. Changes are made
        in place. For real input data, this is only possible if
        ``shift=True`` since the factors :math:`p_k` are real only
        in this case.
    shift : `bool` or iterable, optional
        If `True`, the reciprocal grid is shifted by half a stride in
        the negative direction.
        With a boolean array or iterable, this option is applied
        separately on each axis. At least ``dfunc.space.grid.ndim``
        values must be provided.

    Returns
    -------
    `None`
    """
    if dfunc.space.field == RealNumbers() and not shift:
        raise ValueError('cannot pre-process in-place without shift.')

    nsamples = dfunc.space.grid.shape
    shift_lst = _shift_list(shift, dfunc.ndim)

    def _onedim_arr(length, shift):
        if shift:
            # (-1)^indices
            indices = np.arange(length, dtype='int8')
            arr = -2 * np.mod(indices, 2) + 1
        else:
            indices = np.arange(length, dtype=float)
            arr = np.exp(1j * pi * indices * (1 - 1.0 / length))
        return arr

    onedim_arrs = [_onedim_arr(nsamp, shft)
                   for nsamp, shft in zip(nsamples, shift_lst)]
    meshgrid = sparse_meshgrid(*onedim_arrs, order=dfunc.space.order)

    # Multiply with broadcasting
    for vec in meshgrid:
        np.multiply(dfunc, vec, out=dfunc.asarray())


def dft_postproc_data(dfunc, x0):
    """Post-process the Fourier-space data after DFT.

    This function multiplies the given data with the separable
    function

        :math:`q(\\xi) = e^{-i x_0^{\mathrm{T}}\\xi},`

    where :math:`x_0` :math:`\\xi_0` are the minimum coodinates of
    the real space and reciprocal grids, respectively. In discretized
    form, this function becomes for each axis separately an array

        :math:`q_k = e^{-i x_0
        \\big(\\xi_0 + 2\pi k / (s N)\\big)}.`

    Parameters
    ----------
    dfunc : `DiscreteLpVector`
        Discrete function to be post-processed. Its grid is assumed
        to be the reciprocal grid. Changes are made in place.
    x0 : array-like
        Minimal grid point of the spatial grid before transform

    Returns
    -------
    `None`
    """
    rgrid = dfunc.space.grid

    onedim_arrs = [np.exp(-1j * x * xi)
                   for x, xi in zip(x0, rgrid.coord_vectors)]
    meshgrid = sparse_meshgrid(*onedim_arrs, order=dfunc.space.order)

    # Multiply with broadcasting
    for vec in meshgrid:
        np.multiply(dfunc, vec, out=dfunc.asarray())


def _check_in_out(arr_in, arr_out, axes, halfcomplex, direction):
    """Raise an error if anything is not ok."""
    if direction == 'forward':
        out_shape = arr_in[axes].shape[:-1]
        if halfcomplex:
            out_shape += (arr_in[axes].shape[-1] // 2 + 1,)
        else:
            out_shape += (arr_in[axes].shape[-1],)

        if arr_out[axes].shape != out_shape:
            raise ValueError('Expected output shape {}, got {}.'
                             ''.format(out_shape, arr_out[axes].shape))

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
        in_shape = arr_out[axes].shape[:-1]
        if halfcomplex:
            in_shape += (arr_out[axes].shape[-1] // 2 + 1,)
        else:
            in_shape += (arr_out[axes].shape[-1],)

        if arr_in[axes].shape != in_shape:
            raise ValueError('Expected input shape {}, got {}.'
                             ''.format(in_shape, arr_in[axes].shape))

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
    """Calculate the DFT.

    The discrete Fourier transform calcuates the sum

        :math:`\widehat{f}_k =
        \sum_{j \\in I_N} f_j\ e^{-i 2\pi j\odot k / N}`

    for indices :math:`k \\in I_N` or, in the half-complex case,
    :math:`0 \\leq k_d \\leq \\lfloor N_d / 2 \\rfloor + 1` for the
    last component.

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
    planning_timelimit : `float`, optional
        Limit planning time to roughly this amount of seconds.
        Default: `None` (no limit)
    threads : `int`, optional
        Number of threads to use. Default: 1
    normalise_idft : `bool`, optional
        If `True`, the backward transform is normalized by
        ``1 / N``, where ``N`` is the total number of points in
        ``array_in[axes]``. This ensures that the IDFT is the true
        inverse of the forward DFT.
        Default: `False`
    import_wisdom : `str`, optional
        File name to load FFTW wisdom from. If the file does not exist,
        it is ignored.
    export_wisdom : `str`, optional
        File name to append the accumulated FFTW wisdom to

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
      ``array_out[axes].shape[-1] == array_in[axes].shape[-1] // 2 + 1``,
      and vice versa for backward transforms.
    * All planning schemes except ``'estimate'`` require an internal copy
      of the input array and may therefore be substantially slower in the
      first run. Use these flags only if you calculate multiple transforms
      of the same size.
    * The input can be destroyed if a planner different from
      ``'estimate'`` is used or a complex-to-real backward transform is
      computed.
    """
    import pickle

    assert array_in.flags.aligned

    # We can use _fftw_to_local here since it strigifies and converts to
    # lowercase
    if axes is None:
        axes = np.arange(array_in.ndim)
    else:
        axes = list(axes)
        if axes != list(set(axes)):
            raise ValueError('Duplicate indices not allowed in axes.')

    direction = _fftw_to_local(direction)
    fftw_plan = kwargs.pop('fftw_plan', None)
    planning_effort = _fftw_to_local(kwargs.pop('planning', 'estimate'))
    planning_timelimit = kwargs.pop('planning_timelimit', None)
    threads = kwargs.pop('threads', 1)
    normalise_idft = kwargs.pop('normalise_idft', False)
    wimport = kwargs.pop('import_wisdom', '')
    wexport = kwargs.pop('export_wisdom', '')

    _check_in_out(array_in, array_out, axes, halfcomplex, direction)

    # Cast input to complex if necessary and check for reasonalbe
    # combination of halfcomplex and data type
    array_in_copied = False
    if is_real_dtype(array_in.dtype) and not halfcomplex:
        # Need to cast array_in to complex dtype
        array_in = array_in.astype(_TYPE_MAP_R2C[array_in.dtype])
        array_in_copied = True

    # Import wisdom if possible
    if wimport:
        try:
            with open(wimport, 'r') as wfile:
                wisdom = pickle.load(wfile)
            if wisdom:
                pyfftw.import_wisdom(wisdom)
        except IOError:
            pass

    # Copy input array if it hasn't been done yet and the planner is likely
    # to destroy it. If we already have a plan, we don't have to worry.
    planner_destroys = _fftw_destroys_input(
        [planning_effort], direction, halfcomplex, array_in.ndim)
    must_copy_array_in = fftw_plan is None and planner_destroys

    if must_copy_array_in and not array_in_copied:
        plan_arr_in = np.empty_like(array_in)
        flags = [_local_to_fftw(planning_effort), 'FFTW_DESTROY_INPUT']
    else:
        plan_arr_in = array_in
        flags = [_local_to_fftw(planning_effort)]

    if fftw_plan is None:
        fft_plan = pyfftw.FFTW(
            plan_arr_in, array_out, direction=_local_to_fftw(direction),
            flags=flags, planning_timelimit=planning_timelimit,
            threads=threads, axes=axes)

    fft_plan(array_in, array_out, normalise_idft=normalise_idft)

    if wexport:
        with open(wexport, 'a') as wfile:
            pickle.dump(pyfftw.export_wisdom(), wfile)

    return fftw_plan


class PyfftwTransform(Operator):

    """Plain forward or backward DFT as implemented in ``pyfftw``.

    This operator calculates the DFT without any shifting
    or scaling compensation. See the `pyfftw API documentation`_
    and `What FFTW really computes`_ for further information.

    References
    ----------
    .. _pyfftw API documentation:
       http://hgomersall.github.io/pyFFTW/pyfftw/pyfftw.html
    .. _What FFTW really computes:
       http://www.fftw.org/fftw3_doc/What-FFTW-Really-Computes.html
    """
    def __init__(self, dom, ran=None):
        """Initialize a new instance.

        Parameters
        ----------
        dom : `DiscreteLp`
            Domain of the operator. Its `DiscreteLp.exponent` must be
            at least 1.0, and its `DiscreteLp.grid` must be a
            `RegularGrid`.
        ran : `DiscreteLp`, optional
            Range of the operator. By default, the range is inferred
            from the domain and has a grid with minimum point 0 and
            stride ``(1, ..., 1)``.
        halfcomplex : `bool`, optional
            If `True`, calculate only the negative frequency part
            along the last axis for real input. If `False`,
            calculate the full complex FFT.
            Default: `False`
        """
        if not isinstance(dom, DiscreteLp):
            raise TypeError('domain {!r} is not a `DiscreteLp` instance.'
                            ''.format(dom))

        if not isinstance(dom.dspace, Ntuples):
            raise TypeError('Only numpy.ndarray data supported by pyfftw. '
                            'Got data space {!r}.'.format(dom.dspace))
        if dom.exponent < 1:
            raise ValueError('domain exponent {} < 1 not allowed.'
                             ''.format(dom.exponent))
        if not isinstance(dom.grid, RegularGrid):
            raise TypeError('irregular grids not supported.')

        if ran is None:
            # Calculate range - a complex DiscreteLp with conjugate exponent
            if dom.exponent == 1.0:
                conj_exp = float('inf')
            elif dom.exponent == float('inf'):
                conj_exp = 1.0
            else:
                conj_exp = dom.exponent / (dom.exponent - 1.0)

            # Standard grid with stride 1 and minimum (0, ..., 0)
            # TODO: check how order is handled
            ran_grid = RegularGrid([0] * dom.ndim, np.array(dom.shape) - 1,
                                   dom.shape, as_midp=False)

            ran_fspace = FunctionSpace(ran_grid.convex_hull(),
                                       ComplexNumbers())

            if is_real_dtype(dom.dtype):
                ran_dtype = _TYPE_MAP_R2C[dom.dtype]
            else:
                ran_dtype = dom.dtype

            ran_dspace_type = dspace_type(ran_fspace, impl='numpy',
                                          dtype=ran_dtype)
            ran_dspace = ran_dspace_type(ran_grid.size, dtype=ran_dtype,
                                         exponent=conj_exp)
            ran = DiscreteLp(ran_fspace, ran_grid, ran_dspace,
                             exponent=conj_exp)

        super().__init__(dom, ran, linear=True)

    def _call(self, x, out=None, **kwargs):
        """Implementation of ``self(x[, out])``.

        Parameters
        ----------
        x : `DiscreteLpVector`
            Input vector to be transformed
        out : `DiscreteLpVector`, optional
            Output vector storing the result
        axes : sequence of `int`, optional
            Dimensions in which a transform is to be calculated.
            Default: ``(-1,)``.
        direction : {'FFTW_FORWARD', 'FFTW_BACKWARD'}, optional
            Direction of the transform
        normalise_idft : `bool`, optional
            If `True`, the IDFT (``'FFTW_BACKWARD'``) is normalized by
            ``1 / N``, where ``N`` is the total number of points in
            ``x[axes]``. This ensures that the IDFT is the true inverse
            of the forward DFT.
            Default: `True`
        flags : sequence of `str`, optional
            Flags for the transform. See the `pyfftw API documentation`_
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
        """
        # TODO: Implement and update doc
        # TODO: Handle returned plan, maybe an init flag?


class DiscreteFourierTransform(Operator):

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

    def __init__(self, dom, **kwargs):
        """
        Parameters
        ----------
        dom : `DiscreteLp`
            Domain of the wavelet transform. Its
            :attr:`odl.DiscreteLp.exponent` must be at least 1.0;
            if it is equal to 2.0, this operator has an adjoint which
            is equal to the inverse.
        halfcomplex : `bool`, optional
            If `True`, calculate only the negative frequency part
            along the last axis for real input. If `False`,
            calculate the full complex FFT.

            TODO: doc combination with shift

            This option only applies to 'uni-to-uni' transforms.
            For complex ``dom``, it has no effect.

            Default: `True`

        shift : `bool` or iterable, optional
            If `True`, the reciprocal grid is shifted by half a stride in
            the negative direction.
            With a boolean array or iterable, this option is applied
            separately on each axis. At least ``dom.grid.ndim``
            values must be provided.

            This option only applies to 'uni-to-uni' transforms.

            Default: `True`

        Notes
        -----
        The `Operator.range` of this operator always has the
        `ComplexNumbers` as its `LinearSpace.field`, i.e. if the
        field of ``dom`` is the `RealNumbers`, this operator has no
        `Operator.adjoint`.
        """
        if not isinstance(dom, DiscreteLp):
            raise TypeError('domain {!r} is not a `DiscreteLp` instance.'
                            ''.format(dom))
        if not isinstance(dom.dspace, Ntuples):
            raise NotImplementedError(
                'Only Numpy-based data spaces are supported, got {}.'
                ''.format(dom.dspace))

        # Check exponents
        if dom.exponent < 1:
            raise ValueError('domain exponent {} < 1 not allowed.'
                             ''.format(dom.exponent))
        if dom.exponent == 1.0:
            conj_exp = float('inf')
        elif dom.exponent == float('inf'):
            conj_exp = 1.0  # This is not strictly correct in math, but anyway
        else:
            conj_exp = dom.exponent / (dom.exponent - 1.0)

        if isinstance(dom.grid, RegularGrid):
            if dom.field == ComplexNumbers():
                self._halfcomplex = False
            else:
                self._halfcomplex = bool(kwargs.pop('halfcomplex', True))

            self._shift = bool(kwargs.pop('shift', True))
        else:
            raise NotImplementedError('irregular grids not yet supported.')

        # Calculate range
        recip_grid = reciprocal(dom.grid, shift=self._shift,
                                halfcomplex=self._halfcomplex)

        # Always complex space
        ran_fspace = FunctionSpace(recip_grid.convex_hull(), ComplexNumbers())

        if is_real_dtype(dom.dtype):
            ran_dtype = _TYPE_MAP_R2C[dom.dtype]
        else:
            ran_dtype = dom.dtype
        # TODO: handle impl
        ran_dspace_type = dspace_type(ran_fspace, impl='numpy',
                                      dtype=ran_dtype)
        ran_dspace = ran_dspace_type(recip_grid.size, dtype=ran_dtype,
                                     exponent=conj_exp)
        # TODO: check how order is handled
        ran = DiscreteLp(ran_fspace, recip_grid, ran_dspace, exponent=conj_exp)

        super().__init__(dom, ran, linear=True)

    def _call(self, x, out, **kwargs):
        """Raw out-of-place evaluation method.

        TODO: write doc
        """
        # TODO: custom axes
        # TODO: handle return value of pyfftw_call, maybe an init kwarg?
        # TODO: Implement version using Numpy FFT
        x_cpy = x.copy()
        dft_preproc_data(x_cpy, shift=self._shift)
        pyfftw_call(x_cpy.asarray(), out.asarray(), self._halfcomplex,
                    **kwargs)
        dft_postproc_data(out, self.domain.grid.min_pt)
        return out

    @property
    def adjoint(self):
        """The adjoint wavelet transform."""
        if self.domain.exponent == 2.0:
            return self.inverse
        else:
            raise NotImplementedError('adjoint only defined for exponent 2.0, '
                                      'not {}.'.format(self.domain.exponent))

    @property
    def inverse(self):
        """The inverse wavelet transform."""
        # TODO: add appropriate arguments
        raise NotImplementedError


class DiscreteFourierTransformInverse(Operator):
    pass


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
