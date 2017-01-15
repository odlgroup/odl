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

"""Operators based on (integral) transformations."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np
import scipy.signal as signal

from odl.discr import (
    DiscreteLp, uniform_discr, uniform_discr_fromdiscr)
from odl.operator import Operator, OpDomainError, OpRangeError
from odl.set import ComplexNumbers
from odl.trafos import FourierTransform, DiscreteFourierTransform
from odl.util.normalize import normalized_scalar_param_list, safe_int_conv


__all__ = ('FourierSpaceConvolution', 'RealSpaceConvolution')


_REAL_CONV_SUPPORTED_IMPL = ('scipy_convolve',)
# TODO: get from fourier.py
_FOURIER_CONV_SUPPORTED_IMPL = ('numpy', 'pyfftw')


class ConvolutionBase(Operator):

    """Discretization of the convolution integral as an operator.

    This class provides common attributes and methods for subclasses
    implementing different variants.
    """

    def kernel(self):
        """Return the kernel of this convolution operator."""
        raise NotImplementedError('abstract method')


def _scaled_kernel_ft(kernel, ft_op, ker_mode, ker_kwargs, ker_is_scaled):
    """Return the scaled kernel FT for the given FT operator.

    This helper function calculates the FT of the kernel, discretized in
    an appropriate space.
    """
    # FIXME: the result is wrong if the kernel is given in Fourier space.
    # That case needs to be treated separately.
    ndim = ft_op.domain.ndim
    axes = getattr(ft_op, 'axes', None)
    conv_ndim = ndim if axes is None else len(axes)

    # TODO: adapt for broadcasting. This code enforces that the kernel
    # has the same dimension as the input of the convolution
    if isinstance(kernel,
                  (ft_op.domain.element_type, ft_op.range.element_type)):
        if kernel.ndim != ndim:
            raise ValueError('`kernel` has dimension {}, expected {}'
                             ''.format(kernel.ndim, ndim))
        kernel_space = kernel.space

    else:
        # This encompasses callable and array-like kernels. Those two cases
        # need to be split when broadcasting.

        # Zero-centered version of ft_op.domain
        kernel_space = zero_centered_discr_fromdiscr(ft_op.domain)

    # Initialize the kernel FT operator and compute the kernel scaling
    if isinstance(ft_op, DiscreteFourierTransform):
        ft_op_type = DiscreteFourierTransform
        scaling_factor = 1
    else:
        # Use FourierTransform as default also for other Operator types
        ft_op_type = FourierTransform
        scaling_factor = (2 * np.pi) ** (conv_ndim / 2)

    # Attributes can be undefined for custom FT operators
    ft_kwargs = {}
    impl = getattr(ft_op, 'impl', 'numpy')
    halfcomplex = getattr(ft_op, 'halfcomplex', True)
    shift = getattr(ft_op, 'shift', None)
    if shift is not None:
        ft_kwargs['shift'] = shift

    ker_ft_op = ft_op_type(
        kernel_space, impl=impl, axes=axes, halfcomplex=halfcomplex,
        **ft_kwargs)

    if ker_mode == 'real':
        scaled_kernel = ker_ft_op(kernel)
        scaled_kernel *= scaling_factor

    elif ker_mode == 'fourier':
        scaled_kernel = ft_op.range.element(kernel)
        if not ker_is_scaled:
            scaled_kernel *= scaling_factor

    else:
        raise RuntimeError('bad `ker_mode`')

    return scaled_kernel, scaling_factor, ker_ft_op


class FourierSpaceConvolution(ConvolutionBase):

    """Convolution implemented in Fourier space.

    This operator implements a discrete approximation to the continuous
    convolution with a fixed kernel. It is based on the Fourier
    transform and usually performs better than `RealSpaceConvolution`,
    except for small kernels.
    """

    def __init__(self, domain, kernel, kernel_mode='real', range=None,
                 ft_impl='numpy', integral=True, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscreteLp`
            Uniformly discretized space of functions on which the
            operator can act.
        kernel : `DiscreteLpElement`, callable or `array-like`
            Fixed kernel of the convolution operator. It can be
            given in real or Fourier space (see ``kernel_mode``), and
            specified in several ways, resulting in varying
            `Operator.range`. In all cases, the kernel must have
            the same number of dimensions as ``domain``, but the shapes
            can be different.

            `DiscreteLpElement` : If ``kernel_mode == 'real'``, the range
            is shifted by the midpoint of the kernel domain. The extents
            of ``domain`` and the kernel domain need to match.
            If ``kernel_mode == 'fourier'``, the cell sizes in Fourier
            space need to match, but the shapes can be different.

            callable or array-like : If ``kernel_mode == 'real'``, the
            kernel is interpreted as a continuous/discretized function
            with support centered around zero, which leads to the range
            of this operator being equal to its domain.
            If ``kernel_mode == 'fourier'``, ``kernel`` must be understood
            by the ``element`` method of the range of the Fourier
            transform used in this convolution. For an array-like object,
            this means that the shapes must match.

            See ``Notes`` for further explanation.

        kernel_mode : {'real', 'fourier'}, optional
            Specifies the way the kernel is to be interpreted:

            'real' : real-space kernel (default)

            'fourier' : Fourier-space kernel

        range : `DiscreteLp`, optional
            Space to which the convolution maps. By default, it is inferred
            from ``domain`` and ``kernel``. See ``Notes``.

        ft_impl : {'numpy', 'pyfftw'}, optional
            Implementation of the Fourier transform. Has no effect for
            custom ``ft_op``.

            'numpy' : Default FFT implementation based on Numpy.

            'pyfftw' : Fourier transform using pyFFTW (faster than
            the default FT).

        integral : bool, optional
            If ``True``, approximate the continuous convolution integral
            by using `FourierTransform` internally. Otherwise, the
            unscaled and unshifted `DiscreteFourierTransform` is used.
            Has no effect for a custom ``ft_op``.
            Default: ``True``

        Other parameters
        ----------------
        axes : sequence of ints, optional
            Dimensions in which to convolve. Default: all axes

        kernel_kwargs : dict, optional
            Keyword arguments passed to the call of the kernel function
            if given as a callable.

        kernel_is_scaled : bool, optional
            If ``True``, the kernel is interpreted as already scaled
            as described under the ``scale`` argument.
            Default: False

        ft_kwargs : dict, optional
            Keyword arguments passed to the initializer of the internally
            used Fourier transform. This will have no effect if a
            ``ft_op`` parameter is supplied.

        ft_op : `Operator`, optional
            Use this operator to compute Fourier transforms instead of
            the default `FourierTransform` or `DiscreteFourierTransform`.

        ft_range_tmp : ``ft_op.range`` element, optional
            Temporary for the range of ``ft_op``. It is used to store
            the FT of an input function.

        See also
        --------
        FourierTransform
        RealSpaceConvolution

        Notes
        -----
        - The continuous convolution of two functions
          :math:`f` and :math:`g` defined on :math:`\mathbb{R^d}` is

              :math:`[f \\ast g](x) =
              \int_{\mathbb{R^d}} f(x - y) g(y) \mathrm{d}y`.

          With the help of the Fourier transform, this operation can be
          expressed as a multiplication,

              :math:`\\big[\mathcal{F}(f \\ast g)\\big](\\xi)
              = (2\pi)^{\\frac{d}{2}}\,
              \\big[\mathcal{F}(f)\\big](\\xi) \cdot
              \\big[\mathcal{F}(g)\\big](\\xi)`.

          This implementation covers the case of a fixed convolution
          kernel :math:`k`, i.e. the linear operator

              :math:`\\big[\mathcal{C}_k(f)\\big](x) = [k \\ast f](x)`.

        - If the convolution kernel :math:`k` is defined in real space on
          a rectangular domain :math:`\Omega_0` with midpoint :math:`m_0`,
          then the convolution :math:`k \\ast f` of a function
          :math:`f:\Omega \\to \mathbb{R}` with :math:`k` has a support
          that is a superset of :math:`\Omega + m_0`. Therefore, we
          choose to keep the size of the domain and take
          :math:`\Omega + m_0` as the domain of definition of functions
          in the range of the convolution operator.

          In fact, the support of the convolution is contained in the
          `Minkowski sum
          <https://en.wikipedia.org/wiki/Minkowski_addition>`_
          :math:`\Omega + \Omega_0`.

          For example, if :math:`\Omega = [0, 5]` and
          :math:`\Omega_0 = [1, 3]`, i.e. :math:`m_0 = 2`, then
          :math:`\Omega + \Omega_0 = [1, 8]`. However, we choose the
          shifted interval :math:`\Omega + m_0 = [2, 7]` for the range
          of the convolution since it contains the "main mass" of the
          result.
        """
        # Basic checks
        if not isinstance(domain, DiscreteLp):
            raise TypeError('`domain` {!r} is not a `DiscreteLp` instance.'
                            ''.format(domain))

        kernel_mode, kernel_mode_in = str(kernel_mode).lower(), kernel_mode
        if kernel_mode not in ('real', 'fourier'):
            raise ValueError("`kernel_mode` '{}' not understood."
                             ''.format(kernel_mode_in))

        # Shift range by midpoint if the kernel is a real space element
        if (isinstance(kernel, domain.element_type) and
                kernel_mode == 'real'):
            shift = kernel.space.mid_pt
        else:
            shift = 0

        calc_range = uniform_discr_fromdiscr(domain,
                                             min_pt=domain.min_pt + shift)

        if range is None:
            range = calc_range
        else:
            if range != calc_range:
                raise ValueError('`range` {} inconsistent with range {} '
                                 'calculated from `kernel` and `domain`'
                                 ''.format(range, calc_range))

        super().__init__(domain, range, linear=True)

        # Initialize Fourier transform if not given as argument
        self.__integral = bool(integral)
        ft_op = kwargs.pop('ft_op', None)
        axes = kwargs.pop('axes', None)
        self.__ft_kwargs = kwargs.pop('ft_kwargs', {})
        if axes is not None and 'axes' in self.__ft_kwargs:
            raise ValueError('`axes` cannot be given both directly and in '
                             '`ft_kwargs`')

        if ft_op is None:
            if self.integral:
                ft_op_type = FourierTransform
            else:
                ft_op_type = DiscreteFourierTransform

            self.__ft_op = ft_op_type(
                self.domain, impl=ft_impl, axes=axes, **self.__ft_kwargs)

        else:
            if not isinstance(ft_op, Operator):
                raise TypeError('`ft_op` must be an `Operator` instance, got '
                                '{!r}'.format(ft_op))
            if ft_op.domain != domain:
                raise OpDomainError(
                    '`ft_op` must have `domain` {} as its domain, got {}'
                    ''.format(domain, ft_op.domain))
            self.__ft_op = ft_op

        # Calculate scaled kernel FT
        kernel_kwargs = kwargs.pop('kernel_kwargs', {})
        kernel_is_scaled = kwargs.pop('kernel_is_scaled', False)
        scaled_ker_ft, scaling, ker_ft_op = _scaled_kernel_ft(
            kernel, self.ft_op, kernel_mode, kernel_kwargs, kernel_is_scaled)

        self.__scaled_kernel_ft = scaled_ker_ft
        self.__kernel_scaling = scaling
        self.__kernel_ft_op = ker_ft_op

        # FT range temporary
        ft_range_tmp = kwargs.pop('ft_range_tmp', None)
        if ft_range_tmp is not None and ft_range_tmp not in self.ft_op.range:
            raise OpRangeError('`ft_range_tmp` not in `ft_op.range`')
        self.__ft_range_tmp = ft_range_tmp

        # Cache adjoint operator
        self.__adjoint = FourierSpaceConvolution(
            domain=self.range, range=self.domain,
            kernel=self.__scaled_kernel_ft.conj(), kernel_mode='fourier',
            kernel_is_scaled=True, axes=axes, ft_impl=self.ft_impl,
            integral=self.integral, ft_kwargs=self.__ft_kwargs,
            ft_range_tmp=self.__ft_range_tmp)

    @property
    def integral(self):
        """``True`` if the conv. integral is approximated, else ``False``"""
        return self.__integral

    @property
    def ft_op(self):
        """Fourier transform operator used in this convolution."""
        return self.__ft_op

    @property
    def ft_impl(self):
        """Implementation of the Fourier transform in this convolution."""
        return self.ft_op.impl

    @property
    def axes(self):
        """Axes along which the convolution is taken."""
        return self.ft_op.axes

    @property
    def kernel_ft_op(self):
        """The Fourier transform operator for the kernel.

        Its domain may differ from that of `ft_op`, but the ranges of both
        transforms are the same.
        """
        # TODO: Update docstring when broadcasting convolution is there
        return self.__kernel_ft_op

    def kernel_ft(self, out=None):
        """Return a the FT of this convolution operator's kernel.

        Parameters
        ----------
        out : element of ``kernel_ft_op.range``, optional
            Write the output to this data storage.

        Returns
        -------
        kernel_ft : element of ``kernel_ft_op.range``
            FT of the convolution kernel. If ``out`` was given, this
            object is a reference to it.
        """
        if out is None:
            out = self.kernel_ft_op.range.element()

        out.lincomb(1 / self.__kernel_scaling, self.__scaled_kernel_ft)
        return out

    def kernel(self, out=None):
        """Return the real-space kernel of this convolution operator.

        Parameters
        ----------
        out : element of ``kernel_ft_op.domain``, optional
            Write the output to this data storage.

        Returns
        -------
        kernel : element of ``kernel_ft_op.domain``
            The Convolution kernel. If ``out`` was given, this
            object is a reference to it.
        """
        if out is None:
            out = self.kernel_ft_op.domain.element()

        self.kernel_ft_op.inverse(self.__scaled_kernel_ft, out=out)
        out /= self.__kernel_scaling
        return out

    def _call(self, x, out, **kwargs):
        """Implement ``self(x, out[, **kwargs])``.

        Keyword arguments are passed on to the transform.
        """
        x_trafo = self.ft_op(x, out=self.__ft_range_tmp, **kwargs)
        # TODO: use broadcasting
        x_trafo *= self.__scaled_kernel_ft
        self.ft_op.inverse(x_trafo, out=out, **kwargs)

    @property
    def adjoint(self):
        """Adjoint operator given by the reversed kernel.

        This can be expressed as multiplication with the complex
        conjugate kernel in Fourier space, which is used in this
        implementation.
        """
        return self.__adjoint

    @property
    def inverse(self):
        """Poor man's deconvolution using the reciprocal kernel FT."""
        denom_eps = 1e-5
        inv_ker_ft = self.kernel_ft()
        inv_ker_ft[np.abs(inv_ker_ft) < denom_eps] = denom_eps
        return FourierSpaceConvolution(
            domain=self.range, range=self.domain,
            kernel=inv_ker_ft, kernel_mode='fourier', kernel_is_scaled=False,
            axes=self.axes, ft_impl=self.ft_impl, integral=self.integral,
            ft_kwargs=self.__ft_kwargs, ft_range_tmp=self.__ft_range_tmp)


class RealSpaceConvolution(ConvolutionBase):

    """Convolution implemented in real space.

    This variant of the convolution is based on `scipy.signal.convolve`
    and is usually fast for small kernels. For a kernel of similar size
    as the input, `FourierSpaceConvolution` is probably much faster.
    """

    def __init__(self, domain, kernel, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscreteLp`
            Uniformly discretized space of functions on which the
            operator can act.
        kernel : `DiscreteLpElement`, callable or `array-like`
            Fixed kernel of the convolution operator. It can be
            specified in several ways, resulting in varying
            `Operator.range`. In all cases, the kernel must have
            the same number of dimensions as ``domain``, but the shapes
            can be different.

            `DiscreteLpElement` : The range is shifted by the midpoint
            of the kernel domain.

            callable or array-like : The kernel is interpreted as a
            continuous/discretized function with support centered around
            zero, which leads to the range of this operator being equal
            to its domain.

            See ``Notes`` for further explanation.

        scale : bool, optional
            If ``True``, scale the discrete convolution with
            `DiscreteLp.cell_volume`, such that it approximates a
            continuous convolution.
            Default: ``True``

        Other parameters
        ----------------
        kernel_scaled : bool, optional
            If True, the kernel is interpreted as already scaled
            as described under the ``scale`` argument.
            Default: False
        kernel_kwargs : dict, optional
            Keyword arguments passed to the call of the kernel function
            if given as a callable.

        See also
        --------
        scipy.signal.convolve : real-space convolution

        Examples
        --------
        >>> import odl
        >>> space = odl.uniform_discr(-5, 5, 5)
        >>> print(space.cell_sides)
        [ 2.]

        By default, the discretized convolution is scaled such that
        it approximates the convolution integral. This behavior can be
        switched off with ``scale=False``:

        >>> kernel = [-1, 1]
        >>> conv = odl.trafos.RealSpaceConvolution(space, kernel)
        >>> conv([0, 1, 2, 0, 0])
        uniform_discr(-5.0, 5.0, 5).element([0.0, -2.0, -2.0, 4.0, 0.0])
        >>> conv_noscale = odl.trafos.RealSpaceConvolution(space, kernel,
        ...                                                scale=False)
        >>> conv_noscale([0, 1, 2, 0, 0])
        uniform_discr(-5.0, 5.0, 5).element([0.0, -1.0, -1.0, 2.0, 0.0])

        If the kernel is given as an element of a uniformly discretized
        function space, the convolution operator range is shifted by
        the midpoint of the kernel domain:

        >>> kernel_space = odl.uniform_discr(0, 4, 2)  # midpoint 2.0
        >>> kernel = kernel_space.element([-1, 1])
        >>> conv_shift = odl.trafos.RealSpaceConvolution(space, kernel)
        >>> conv_shift.range  # Shifted by 2.0 in positive direction
        uniform_discr(-3.0, 7.0, 5)
        >>> conv_shift([0, 1, 2, 0, 0])
        uniform_discr(-3.0, 7.0, 5).element([0.0, -2.0, -2.0, 4.0, 0.0])

        Notes
        -----
        - The continuous convolution of two functions
          :math:`f` and :math:`g` defined on :math:`\mathbb{R^d}` is

              :math:`[f \\ast g](x) =
              \int_{\mathbb{R^d}} f(x - y) g(y) \mathrm{d}y`.

          This implementation covers the case of a fixed convolution
          kernel :math:`k`, i.e. the linear operator :math:`\mathcal{C}_k`
          given by

              :math:`\mathcal{C}_k(f): x \mapsto [k \\ast f](x)`.

        - If the convolution kernel :math:`k` is defined on a rectangular
          domain :math:`\Omega_0` with midpoint :math:`m_0`, then the
          convolution :math:`k \\ast f` of a function
          :math:`f:\Omega \\to \mathbb{R}` with :math:`k` has a support
          that is a superset of :math:`\Omega + m_0`. Therefore, we
          choose to keep the size of the domain and take
          :math:`\Omega + m_0` as the domain of definition of functions
          in the range of the convolution operator.

          In fact, the support of the convolution is contained in the
          `Minkowski sum
          <https://en.wikipedia.org/wiki/Minkowski_addition>`_
          :math:`\Omega + \Omega_0`.

          For example, if :math:`\Omega = [0, 5]` and
          :math:`\Omega_0 = [1, 3]`, i.e. :math:`m_0 = 2`, then
          :math:`\Omega + \Omega_0 = [1, 8]`. However, we choose the
          shifted interval :math:`\Omega + m_0 = [2, 7]` for the range
          of the convolution since it contains the "main mass" of the
          result.
        """
        # Basic checks
        if not isinstance(domain, DiscreteLp):
            raise TypeError('domain {!r} is not a DiscreteLp instance'
                            ''.format(domain))
        if not domain.is_uniform:
            raise ValueError('irregular sampling not supported')

        kernel_kwargs = kwargs.pop('kernel_kwargs', {})
        self.__kernel, range = self._compute_kernel_and_range(
            kernel, domain, kernel_kwargs)

        super().__init__(domain, range, linear=True)

        # Hard-coding impl for now until we add more
        self.__impl = 'scipy_convolve'

        # Scale the kernel if desired
        kernel_scaled = kwargs.pop('kernel_scaled', False)
        self.__kernel_scaling = np.prod(self.__kernel.space.cell_sides)
        self.__scale = bool(kwargs.pop('scale', True))
        if self.__scale and not kernel_scaled:
            # Don't modify the input
            self.__kernel = self.__kernel * self.__kernel_scaling

    @staticmethod
    def _compute_kernel_and_range(kernel, domain, kernel_kwargs):
        """Return the kernel and the operator range based on the domain.

        This helper function computes an adequate zero-centered domain
        for the kernel if necessary (i.e. if the kernel is not a
        `DiscreteLpElement`) and calculates the operator range based
        on the operator domain and the kernel space.

        Parameters
        ----------
        kernel : `DiscreteLpElement` or array-like
            Kernel of the convolution.
        domain : `DiscreteLp`
            Domain of the convolution operator.
        kernel_kwargs : dict
            Used as ``space.element(kernel, **kernel_kwargs)`` if
            ``kernel`` is callable.

        Returns
        -------
        kernel : `DiscreteLpElement`
            Element in an adequate space.
        range : `DiscreteLp`
            Range of the convolution operator.
        """
        if isinstance(kernel, domain.element_type):
            if kernel.ndim != domain.ndim:
                raise ValueError('`kernel` has dimension {}, expected {}'
                                 ''.format(kernel.ndim, domain.ndim))
            # Set the range equal to the domain shifted by the midpoint of
            # the kernel space.
            kernel_shift = kernel.space.partition.mid_pt
            range = uniform_discr_fromdiscr(
                domain, min_pt=domain.min_pt + kernel_shift)

        elif callable(kernel):
            # Use "natural" kernel space, which is the zero-centered
            # version of the domain.
            extent = domain.partition.extent()
            std_kernel_space = uniform_discr_fromdiscr(
                domain, min_pt=-extent / 2)
            kernel = std_kernel_space.element(kernel, **kernel_kwargs)
            range = domain

        else:
            # Make a zero-centered space with the same cell sides as
            # domain, but shape according to the kernel.
            kernel = np.asarray(kernel, dtype=domain.dtype)
            shape = kernel.shape
            extent = domain.cell_sides * shape
            kernel_space = uniform_discr_fromdiscr(
                domain, min_pt=-extent / 2, shape=shape)
            kernel = kernel_space.element(kernel)
            range = domain

        return kernel, range

    @property
    def impl(self):
        """Implementation of this operator."""
        return self.__impl

    def kernel(self):
        """Return the real-space kernel of this convolution operator.

        Note that internally, the scaled kernel is stored if
        ``scale=True`` (default) was chosen during initialization, i.e.
        calculating the original kernel requires to reverse the scaling.
        Otherwise, the original unscaled kernel is returned.
        """
        if self._scale:
            return self.__kernel / self.__kernel_scaling
        else:
            return self.__kernel

    def _call(self, x):
        """Return ``self(x)``."""
        # No resampling needed
        return signal.convolve(x, self.__kernel, mode='same')

    @property
    def adjoint(self):
        """Adjoint operator defined by the reversed kernel."""
        # Make space for the adjoint kernel, -S if S is the kernel space
        adj_min_pt = -self._kernel.space.max_pt
        adj_max_pt = -self._kernel.space.min_pt
        adj_ker_space = uniform_discr_fromdiscr(
            self._kernel.space,
            min_pt=adj_min_pt, max_pt=adj_max_pt)

        # Adjoint kernel is k_adj(x) = k(-x), i.e. the kernel array is
        # reversed in each axis.
        reverse_slice = (slice(None, None, -1),) * self.domain.ndim
        adj_kernel = adj_ker_space.element(
            self._kernel.asarray()[reverse_slice])

        return RealSpaceConvolution(self.range, adj_kernel, impl=self.impl,
                                    scale=True, kernel_scaled=True)


def zero_centered_discr_fromdiscr(discr, shape=None, shift=None):
    """Return a space centered around zero using ``discr`` as template.

    The domain of ``discr`` is translated such that it is symmetric
    around 0, and optionally resized to ``shape`` if given.
    Additionally, a ``shift`` can be applied afterwards if given.

    The cell sizes and further properties like ``dtype`` are preserved.

    Parameters
    ----------
    discr : `DiscreteLp`
        Uniformly discretized space to be used as template.
    shape : int or sequence of int, optional
        If specified, resize the space to this shape.
    shift : `array-like`, optional
        Center the space around this point instead of the origin.

    Returns
    -------
    newspace : `DiscreteLp`
        Uniformly discretized space whose domain has midpoint
        ``(0, ..., 0)`` and the same cell size as ``discr``.
    """
    if not isinstance(discr, DiscreteLp):
        raise TypeError('`discr` {!r} is not a DiscreteLp instance'
                        ''.format(discr))
    if not discr.is_uniform:
        raise ValueError('irregular sampling not supported')

    if shape is not None:
        shape = normalized_scalar_param_list(shape, discr.ndim,
                                             param_conv=safe_int_conv)
    else:
        shape = discr.shape

    if shift is not None:
        shift = np.asarray(shift, dtype=float)
    else:
        shift = np.zeros(discr.ndim)

    new_min_pt = -(discr.cell_sides * shape) / 2 + shift
    return uniform_discr_fromdiscr(discr, min_pt=new_min_pt, shape=shape)


def conv_resampling_spaces(space, kernel_space, up_or_down):
    """Return spaces for resampling of input and convolution kernel.

    This function returns spaces usable for resampling of input functions
    of a convolution (by composition with `odl.ResamplingOperator`) and
    the convolution kernel based on the decision to sample up or down.

    Parameters
    ----------
    space, kernel_space : `DiscreteLp`
        Uniformly discretized spaces in which input functions to a
        convolution and the convolution kernel lie, respectively.
    up_or_down : string or sequence of strings
        If ``space`` and ``kernel_space`` have different cell sizes,
        this parameter defines which one is used as common cell size
        via resampling (per axis in case of a sequence):

        'down' : Use the larger one of the two cell sizes.
        This can lead to loss of resolution.

        'up' : Use the smaller one of the two cell sizes.
        This can be computationally expensive.

    Returns
    -------
    space_resamp : `DiscreteLp`
        Resampling space for input functions to the convolution.
    kernel_space_resamp : `DiscreteLp`
        Resampling space for the convolution kernel.

    Examples
    --------
    Compute spaces for upscaling in 2D:

    >>> import odl
    >>> space = odl.uniform_discr([0, 0], [1, 1], (5, 10))
    >>> space.cell_sides
    array([ 0.2,  0.1])
    >>> kernel_space = odl.uniform_discr([0, 0], [1, 1], (10, 5))
    >>> kernel_space.cell_sides
    array([ 0.1,  0.2])
    >>> space_r, kernel_space_r = conv_resampling_spaces(
    ...     space, kernel_space, up_or_down='up')
    >>> space_r.cell_sides
    array([ 0.1,  0.1])
    >>> kernel_space_r.cell_sides
    array([ 0.1,  0.1])

    This can be done per axis:

    >>> space_r, kernel_space_r = conv_resampling_spaces(
    ...     space, kernel_space, up_or_down=['up', 'down'])
    >>> space_r.cell_sides
    array([ 0.1,  0.2])
    >>> kernel_space_r.cell_sides
    array([ 0.1,  0.2])
    """

    up_or_down, up_or_down_in = normalized_scalar_param_list(
        up_or_down, length=space.ndim,
        param_conv=lambda s: str(s).lower()), up_or_down
    for i, (ud, ud_in) in enumerate(zip(up_or_down, up_or_down_in)):
        if ud not in ('up', 'down'):
            raise ValueError("in axis {}: `up_or_down` '{}' not understood"
                             ''.format(i, ud_in))

    # Compute cell sides according to the decisions on up- or downsampling
    new_space_csides, new_kernel_csides = _get_resamp_csides(
        space.cell_sides, kernel_space.cell_sides, up_or_down)

    space_resamp = uniform_discr_fromdiscr(
        space, cell_sides=new_space_csides)

    kernel_space_resamp = uniform_discr_fromdiscr(
        kernel_space, cell_sides=new_kernel_csides)

    return space_resamp, kernel_space_resamp


def _get_resamp_csides(dom_csides, ker_csides, up_or_down):
    """Return cell sides for domain and kernel resampling."""
    new_dom_csides = []
    new_ker_csides = []
    for resamp, ks, ds in zip(up_or_down, dom_csides, ker_csides):

        if np.isclose(ds, ks):
            # Keep old if both csides are the same
            new_dom_csides.append(ds)
            new_ker_csides.append(ks)
        else:
            # Pick the coarser one if 'down' or the finer one if 'up'
            if ((ds > ks and resamp == 'up') or
                    (ds < ks and resamp == 'down')):
                new_dom_csides.append(ks)
                new_ker_csides.append(ks)
            else:
                new_dom_csides.append(ds)
                new_ker_csides.append(ds)

    return new_dom_csides, new_ker_csides


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
