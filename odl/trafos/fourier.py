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
from builtins import super

import numpy as np

from odl.discr import RegularGrid, DiscreteLp, discr_sequence_space
from odl.operator import Operator
from odl.set import RealNumbers, ComplexNumbers
from odl.trafos.backends.pyfftw_bindings import (
    pyfftw_call, PYFFTW_AVAILABLE, _pyfftw_to_local)
from odl.trafos.util import (
    reciprocal_grid, reciprocal_space,
    dft_preprocess_data, dft_postprocess_data)
from odl.util import (is_real_dtype, is_complex_floating_dtype,
                      dtype_repr, conj_exponent, complex_dtype,
                      normalized_scalar_param_list, normalized_axes_tuple)


__all__ = ('DiscreteFourierTransform', 'DiscreteFourierTransformInverse',
           'FourierTransform', 'FourierTransformInverse')


_SUPPORTED_FOURIER_IMPLS = ('numpy',)
if PYFFTW_AVAILABLE:
    _SUPPORTED_FOURIER_IMPLS += ('pyfftw',)


class DiscreteFourierTransformBase(Operator):

    """Base class for discrete fourier transform classes."""

    def __init__(self, inverse, domain, range=None, axes=None, sign='-',
                 halfcomplex=False, impl='numpy'):
        """Initialize a new instance.

        All parameters are given according to the specifics of the forward
        transform. The ``inverse`` parameter is used to control conversions
        for the inverse transform.

        Parameters
        ----------
        inverse : bool
            If ``True``, the inverse transform is created, otherwise the
            forward transform.
        domain : `DiscreteLp`
            Domain of the Fourier transform. If its
            `DiscreteLp.exponent` is equal to 2.0, this operator has
            an adjoint which is equal to the inverse.
        range : `DiscreteLp`, optional
            Range of the Fourier transform. If not given, the range
            is determined from ``domain`` and the other parameters as
            a `discr_sequence_space` with exponent ``p / (p - 1)``
            (read as 'inf' for p=1 and 1 for p='inf').
        axes : int or sequence of ints, optional
            Dimensions in which a transform is to be calculated. ``None``
            means all axes.
        sign : {'-', '+'}, optional
            Sign of the complex exponent.
        halfcomplex : bool, optional
            If ``True``, calculate only the negative frequency part
            along the last axis in ``axes`` for real input. This
            reduces the size of the range to ``floor(N[i]/2) + 1`` in
            this axis ``i``, where ``N`` is the shape of the input
            arrays.
            Otherwise, calculate the full complex FFT. If ``dom_dtype``
            is a complex type, this option has no effect.
        impl : {'numpy', 'pyfftw'}
            Backend for the FFT implementation. The 'pyfftw' backend
            is faster but requires the ``pyfftw`` package.
        """
        if not isinstance(domain, DiscreteLp):
            raise TypeError('`domain` {!r} is not a `DiscreteLp` instance'
                            ''.format(domain))
        if range is not None and not isinstance(range, DiscreteLp):
            raise TypeError('`range` {!r} is not a `DiscreteLp` instance'
                            ''.format(range))

        # Implementation
        impl, impl_in = str(impl).lower(), impl
        if impl not in _SUPPORTED_FOURIER_IMPLS:
            raise ValueError("`impl` '{}' not supported".format(impl_in))
        self.__impl = impl

        # Axes
        if axes is None:
            axes = tuple(np.arange(domain.ndim))
        self.__axes = normalized_axes_tuple(axes, domain.ndim)

        # Half-complex
        if domain.field == ComplexNumbers():
            self.__halfcomplex = False
        else:
            self.__halfcomplex = bool(halfcomplex)

        ran_dtype = complex_dtype(domain.dtype)

        # Sign of the transform
        if sign not in ('+', '-'):
            raise ValueError("`sign` '{}' not understood".format(sign))
        fwd_sign = ('-' if sign == '+' else '+') if inverse else sign
        if fwd_sign == '+' and self.halfcomplex:
            raise ValueError("cannot combine sign '+' with a half-complex "
                             "transform")
        self.__sign = sign

        # Calculate the range
        ran_shape = reciprocal_grid(
            domain.grid, shift=False, halfcomplex=halfcomplex, axes=axes).shape

        if range is None:
            impl = domain.dspace.impl

            range = discr_sequence_space(
                ran_shape, conj_exponent(domain.exponent), impl=impl,
                dtype=ran_dtype, order=domain.order)
        else:
            if range.shape != ran_shape:
                raise ValueError('expected range shape {}, got {}.'
                                 ''.format(ran_shape, range.shape))
            if range.dtype != ran_dtype:
                raise ValueError('expected range data type {}, got {}.'
                                 ''.format(dtype_repr(ran_dtype),
                                           dtype_repr(range.dtype)))

        if inverse:
            super().__init__(range, domain, linear=True)
        else:
            super().__init__(domain, range, linear=True)
        self._fftw_plan = None

    def _call(self, x, out, **kwargs):
        """Implement ``self(x, out[, **kwargs])``.

        Parameters
        ----------
        x : `domain` element
            Discretized function to be transformed
        out : `range` element
            Element to which the output is written

        Notes
        -----
        See the `pyfftw_call` function for ``**kwargs`` options.
        The parameters ``axes`` and ``halfcomplex`` cannot be
        overridden.

        See Also
        --------
        pyfftw_call : Call pyfftw backend directly
        """
        # TODO: Implement zero padding
        if self.impl == 'numpy':
            out[:] = self._call_numpy(x.asarray())
        else:
            out[:] = self._call_pyfftw(x.asarray(), out.asarray(), **kwargs)

    @property
    def impl(self):
        """Backend for the FFT implementation."""
        return self.__impl

    @property
    def axes(self):
        """Axes along the FT is calculated by this operator."""
        return self.__axes

    @property
    def sign(self):
        """Sign of the complex exponent in the transform."""
        return self.__sign

    @property
    def halfcomplex(self):
        """Return ``True`` if the last transform axis is halved."""
        return self.__halfcomplex

    @property
    def adjoint(self):
        """Adjoint transform, equal to the inverse.

        See Also
        --------
        inverse
        """
        if self.domain.exponent == 2.0 and self.range.exponent == 2.0:
            return self.inverse
        else:
            raise NotImplementedError(
                'no adjoint defined for exponents ({}, {}) != (2, 2)'
                ''.format(self.domain.exponent, self.range.exponent))

    @property
    def inverse(self):
        """Inverse Fourier transform.

        Abstract method.
        """
        raise NotImplementedError('abstract method')

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
        raise NotImplementedError('abstract method')

    def _call_pyfftw(self, x, out, **kwargs):
        """Implement ``self(x[, out, **kwargs])`` using pyfftw.

        Parameters
        ----------
        x : `numpy.ndarray`
            Input array to be transformed
        out : `numpy.ndarray`
            Output array storing the result
        flags : sequence of strings, optional
            Flags for the transform. ``'FFTW_UNALIGNED'`` is not
            supported, and ``'FFTW_DESTROY_INPUT'`` is enabled by
            default. See the `pyfftw API documentation`_
            for futher information.
            Default: ``('FFTW_MEASURE',)``
        threads : positive int, optional
            Number of threads to use. Default: 1
        planning_timelimit : float or ``None``, optional
            Rough upper limit in seconds for the planning step of the
            transform. ``None`` means no limit. See the
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

        kwargs.pop('normalise_idft', None)  # Using `False` here
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

    def init_fftw_plan(self, planning_effort='measure', **kwargs):
        """Initialize the FFTW plan for this transform for later use.

        If the implementation of this operator is not ``'pyfftw'``, this
        method should not be called.

        Parameters
        ----------
        planning_effort : {'estimate', 'measure', 'patient', 'exhaustive'}
            Flag for the amount of effort put into finding an optimal
            FFTW plan. See the `FFTW doc on planner flags
            <http://www.fftw.org/fftw3_doc/Planner-Flags.html>`_.
        planning_timelimit : float, optional
            Limit planning time to roughly this amount of seconds.
            Default: None (no limit)
        threads : int, optional
            Number of threads to use. Default: 1

        Raises
        ------
        ValueError
            If `impl` is not ``'pyfftw'``

        Notes
        -----
        To save memory, clear the plan when the transform is no longer
        used (the plan stores 2 arrays).

        See Also
        --------
        clear_fftw_plan
        """
        if self.impl != 'pyfftw':
            raise ValueError('cannot create fftw plan without fftw backend')

        x = self.domain.element()
        y = self.range.element()
        kwargs.pop('planning_timelimit', None)

        direction = 'forward' if self.sign == '-' else 'backward'
        self._fftw_plan = pyfftw_call(
            x.asarray(), y.asarray(), direction=direction,
            halfcomplex=self.halfcomplex, axes=self.axes,
            planning_effort=planning_effort, **kwargs)

    def clear_fftw_plan(self):
        """Delete the FFTW plan of this transform.

        Raises
        ------
        ValueError
            If `impl` is not ``'pyfftw'``

        Notes
        -----
        If no plan exists, this is a no-op.
        """
        if self.impl != 'pyfftw':
            raise ValueError('cannot create fftw plan without fftw backend')

        self._fftw_plan = None


class DiscreteFourierTransform(DiscreteFourierTransformBase):

    """Plain forward DFT, only evaluating the trigonometric sum.

    This operator calculates the forward DFT::

        f_hat[k] = sum_j( f[j] * exp(-+ 1j*2*pi * j*k/N) )

    without any further shifting or scaling compensation. See the
    `Numpy FFT documentation`_, the `pyfftw API documentation`_ or
    `What FFTW really computes`_ for further information.

    See Also
    --------
    numpy.fft.fftn : n-dimensional FFT routine
    numpy.fft.rfftn : n-dimensional half-complex FFT
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

    def __init__(self, domain, range=None, axes=None, sign='-',
                 halfcomplex=False, impl='numpy'):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscreteLp`
            Domain of the Fourier transform. If its
            `DiscreteLp.exponent` is equal to 2.0, this operator has
            an adjoint which is equal to the inverse.
        range : `DiscreteLp`, optional
            Range of the Fourier transform. If not given, the range
            is determined from ``domain`` and the other parameters as
            a `discr_sequence_space` with exponent ``p / (p - 1)``
            (read as 'inf' for p=1 and 1 for p='inf').
        axes : int or sequence of ints, optional
            Dimensions in which a transform is to be calculated. ``None``
            means all axes.
        sign : {'-', '+'}, optional
            Sign of the complex exponent.
        halfcomplex : bool, optional
            If ``True``, calculate only the negative frequency part
            along the last axis in ``axes`` for real input. This
            reduces the size of the range to ``floor(N[i]/2) + 1`` in
            this axis ``i``, where ``N`` is the shape of the input
            arrays.
            Otherwise, calculate the full complex FFT. If ``dom_dtype``
            is a complex type, this option has no effect.
        impl : {'numpy', 'pyfftw'}
            Backend for the FFT implementation. The ``'pyfftw'`` backend
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
        super().__init__(inverse=False, domain=domain, range=range, axes=axes,
                         sign=sign, halfcomplex=halfcomplex, impl=impl)

    def _call_numpy(self, x):
        """Return ``self(x)`` using numpy.

        See Also
        --------
        DiscreteFourierTransformBase._call_numpy
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

        See Also
        --------
        DiscreteFourierTransformBase._call_numpy
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(out, np.ndarray)

        kwargs.pop('normalise_idft', None)  # Using `False` here
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
    def inverse(self):
        """Inverse Fourier transform."""
        sign = '+' if self.sign == '-' else '-'
        return DiscreteFourierTransformInverse(
            domain=self.range, range=self.domain, axes=self.axes,
            halfcomplex=self.halfcomplex, sign=sign)


class DiscreteFourierTransformInverse(DiscreteFourierTransformBase):

    """Plain backward DFT, only evaluating the trigonometric sum.

    This operator calculates the inverse DFT::

        f[k] = 1/prod(N) * sum_j( f_hat[j] * exp(+- 1j*2*pi * j*k/N) )

    without any further shifting or scaling compensation. See the
    `Numpy FFT documentation`_, the `pyfftw API documentation`_ or
    `What FFTW really computes`_ for further information.

    See Also
    --------
    DiscreteFourierTransform
    FourierTransformInverse
    numpy.fft.ifftn : n-dimensional inverse FFT routine
    numpy.fft.irfftn : n-dimensional half-complex inverse FFT
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
    def __init__(self, range, domain=None, axes=None, sign='+',
                 halfcomplex=False, impl='numpy'):
        """Initialize a new instance.

        Parameters
        ----------
        range : `DiscreteLp`
            Range of the inverse Fourier transform. If its
            `DiscreteLp.exponent` is equal to 2.0, this operator has
            an adjoint which is equal to the inverse.
        domain : `DiscreteLp`, optional
            Domain of the inverse Fourier transform. If not given, the
            domain is determined from ``range`` and the other parameters
            as a `discr_sequence_space` with exponent ``p / (p - 1)``
            (read as 'inf' for p=1 and 1 for p='inf').
        axes : sequence of ints, optional
            Dimensions in which a transform is to be calculated. `None`
            means all axes.
        sign : {'-', '+'}, optional
            Sign of the complex exponent.
        halfcomplex : bool, optional
            If ``True``, interpret the last axis in ``axes`` as the
            negative frequency part of the transform of a real signal
            and calculate a "half-complex-to-real" inverse FFT. In this
            case, the domain has by default the shape
            ``floor(N[i]/2) + 1`` in this axis ``i``.
            Otherwise, domain and range have the same shape. If
            ``range`` is a complex space, this option has no effect.
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
        super().__init__(inverse=True, domain=range, range=domain, axes=axes,
                         sign=sign, halfcomplex=halfcomplex, impl=impl)

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
        """Implement ``self(x[, out, **kwargs])`` using pyfftw.

        Parameters
        ----------
        x : `domain` element
            Input element to be transformed.
        out : `range` element
            Output element storing the result.
        flags : sequence of strings, optional
            Flags for the transform. ``'FFTW_UNALIGNED'`` is not
            supported, and ``'FFTW_DESTROY_INPUT'`` is enabled by
            default. See the `pyfftw API documentation`_
            for futher information.
            Default: ``('FFTW_MEASURE',)``
        threads : positive int, optional
            Number of threads to use. Default: 1
        planning_timelimit : float, optional
            Rough upper limit in seconds for the planning step of the
            transform. The default is no limit. See the
            `pyfftw API documentation`_ for futher information.

        Returns
        -------
        out : `range` element
            Result of the transform. If ``out`` was given, the returned
            object is a reference to it.

        References
        ----------
        .. _pyfftw API documentation:
           http://hgomersall.github.io/pyFFTW/pyfftw/pyfftw.html
        """
        kwargs.pop('normalise_idft', None)  # Using `True` here
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
            domain=self.range, range=self.domain, axes=self.axes,
            halfcomplex=self.halfcomplex, sign=sign)


class FourierTransformBase(Operator):

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

    See Also
    --------
    DiscreteFourierTransform
    FourierTransformInverse
    dft_preprocess_data
    pyfftw_call
    dft_postprocess_data
    """

    def __init__(self, inverse, domain, range=None, impl='numpy', **kwargs):
        """Initialize a new instance.

        All parameters are given according to the specifics of the forward
        transform. The ``inverse`` parameter is used to control conversions
        for the inverse transform.

        Parameters
        ----------
        inverse : bool
            If ``True``, create the inverse transform, otherwise the forward
            transform.
        domain : `DiscreteLp`
            Domain of the Fourier transform. If the
            `DiscreteLp.exponent` of ``domain`` and ``range`` are equal
            to 2.0, this operator has an adjoint which is equal to its
            inverse.
        range : `DiscreteLp`, optional
            Range of the Fourier transform. If not given, the range
            is determined from ``domain`` and the other parameters. The
            exponent is chosen to be the conjugate ``p / (p - 1)``,
            which reads as 'inf' for p=1 and 1 for p='inf'.
        impl : {'numpy', 'pyfftw'}
            Backend for the FFT implementation. The 'pyfftw' backend
            is faster but requires the ``pyfftw`` package.
        axes : int or sequence of ints, optional
            Dimensions along which to take the transform.
            Default: all axes
        sign : {'-', '+'}, optional
            Sign of the complex exponent. Default: '-'
        halfcomplex : bool, optional
            If ``True``, calculate only the negative frequency part
            along the last axis for real input. If ``False``,
            calculate the full complex FFT.
            For complex ``domain``, it has no effect.
            Default: ``True``
        shift : bool or sequence of bools, optional
            If ``True``, the reciprocal grid is shifted by half a stride in
            the negative direction. With a boolean sequence, this option
            is applied separately to each axis.
            If a sequence is provided, it must have the same length as
            ``axes`` if supplied. Note that this must be set to ``True``
            in the halved axis in half-complex transforms.
            Default: ``True``

        Other Parameters
        ----------------
        tmp_r : `DiscreteLpElement` or `numpy.ndarray`
            Temporary for calculations in the real space (domain of
            this transform). It is shared with the inverse.

            Variants using this: R2C, R2HC, C2R (inverse)

        tmp_f : `DiscreteLpElement` or `numpy.ndarray`
            Temporary for calculations in the frequency (reciprocal)
            space. It is shared with the inverse.

            Variants using this: R2C, C2R (inverse), HC2R (inverse)

        Notes
        -----
        * The transform variants are:

          - **C2C**: complex-to-complex.
            The default variant, one-to-one and unitary.

          - **R2C**: real-to-complex.
            This variants adjoint and inverse may suffer
            from information loss since the result is cast to real.

          - **R2HC**: real-to-halfcomplex.
            This variant stores only a half-space of frequencies and
            is guaranteed to be one-to-one (invertible).

        * The `Operator.range` of this operator always has the
          `ComplexNumbers` as `LinearSpace.field`, i.e. if the
          field of ``domain`` is the `RealNumbers`, this operator's adjoint
          is defined by identifying real and imaginary parts with
          the components of a real product space element.
          See the `mathematical background documentation
          <odlgroup.github.io/odl/math/trafos/fourier_transform.html#adjoint>`_
          for details.
        """
        if not isinstance(domain, DiscreteLp):
            raise TypeError('domain {!r} is not a `DiscreteLp` instance'
                            ''.format(domain))
        if domain.impl != 'numpy':
            raise NotImplementedError(
                'Only Numpy-based data spaces are supported, got {}'
                ''.format(domain.dspace))

        # axes
        axes = kwargs.pop('axes', np.arange(domain.ndim))
        self.__axes = normalized_axes_tuple(axes, domain.ndim)

        # Implementation
        impl, impl_in = str(impl).lower(), impl
        if impl not in _SUPPORTED_FOURIER_IMPLS:
            raise ValueError("`impl` '{}' not supported".format(impl_in))
        self.__impl = impl

        # Handle half-complex yes/no and shifts
        if isinstance(domain.grid, RegularGrid):
            if domain.field == ComplexNumbers():
                self.__halfcomplex = False
            else:
                self.__halfcomplex = bool(kwargs.pop('halfcomplex', True))

            self.__shifts = normalized_scalar_param_list(
                kwargs.pop('shift', True), length=len(self.axes),
                param_conv=bool)
        else:
            raise NotImplementedError('irregular grids not yet supported')

        sign = kwargs.pop('sign', '-')
        if sign not in ('+', '-'):
            raise ValueError("`sign` '{}' not understood".format(sign))
        fwd_sign = ('-' if sign == '+' else '+') if inverse else sign
        if fwd_sign == '+' and self.halfcomplex:
            raise ValueError("cannot combine sign '+' with a half-complex "
                             "transform")
        self.__sign = sign

        # Need to filter out this situation since the pre-processing step
        # casts to complex otherwise, and then no half-complex transform
        # is possible.
        if self.halfcomplex and not self.shifts[-1]:
            raise ValueError('`shift` must be `True` in the halved (last) '
                             'axis in half-complex transforms')

        if range is None:
            # self._halfcomplex and self._axes need to be set for this
            range = reciprocal_space(domain, axes=self.axes,
                                     halfcomplex=self.halfcomplex,
                                     shift=self.shifts)

        if inverse:
            super().__init__(range, domain, linear=True)
        else:
            super().__init__(domain, range, linear=True)
        self._fftw_plan = None

        # Storing temporaries directly as arrays
        tmp_r = kwargs.pop('tmp_r', None)
        tmp_f = kwargs.pop('tmp_f', None)

        if tmp_r is not None:
            tmp_r = domain.element(tmp_r).asarray()
        if tmp_f is not None:
            tmp_f = range.element(tmp_f).asarray()

        self._tmp_r = tmp_r
        self._tmp_f = tmp_f

    def _call(self, x, out, **kwargs):
        """Implement ``self(x, out[, **kwargs])``.

        Parameters
        ----------
        x : `domain` element
            Discretized function to be transformed
        out : `range` element
            Element to which the output is written

        Notes
        -----
        See the `pyfftw_call` function for ``**kwargs`` options.
        The parameters ``axes`` and ``halfcomplex`` cannot be
        overridden.

        See Also
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
            Result of the transform
        """
        raise NotImplementedError('abstract method')

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
        planning_timelimit : float or ``None``, optional
            Limit planning time to roughly this many seconds.
            Default: ``None`` (no limit)
        threads : int, optional
            Number of threads to use. Default: 1

        Returns
        -------
        out : `numpy.ndarray`
            Result of the transform. The returned object is a reference
            to the input parameter ``out``.
        """
        raise NotImplementedError('abstract method')

    @property
    def impl(self):
        """Backend for the FFT implementation."""
        return self.__impl

    @property
    def axes(self):
        """Axes along the FT is calculated by this operator."""
        return self.__axes

    @property
    def sign(self):
        """Sign of the complex exponent in the transform."""
        return self.__sign

    @property
    def halfcomplex(self):
        """Return ``True`` if the last transform axis is halved."""
        return self.__halfcomplex

    @property
    def shifts(self):
        """Return the boolean list indicating shifting per axis."""
        return self.__shifts

    @property
    def adjoint(self):
        """Adjoint transform, equal to the inverse.

        See Also
        --------
        inverse
        """
        if self.domain.exponent == 2.0 and self.range.exponent == 2.0:
            return self.inverse
        else:
            raise NotImplementedError(
                'no adjoint defined for exponents ({}, {}) != (2, 2)'
                ''.format(self.domain.exponent, self.range.exponent))

    @property
    def inverse(self):
        """Inverse Fourier transform.

        See Also
        --------
        adjoint : Equivalent since the fourier transform is unitary.
        """
        sign = '+' if self.sign == '-' else '-'
        return FourierTransformInverse(
            domain=self.range, range=self.domain, impl=self.impl,
            axes=self.axes, halfcomplex=self.halfcomplex, shift=self.shifts,
            sign=sign, tmp_r=self._tmp_r, tmp_f=self._tmp_f)

    def create_temporaries(self, r=True, f=True):
        """Allocate and store reusable temporaries.

        Existing temporaries are overridden.

        Parameters
        ----------
        r : bool, optional
            Create temporary for the real space
        f : bool, optional
            Create temporary for the frequency space

        Notes
        -----
        To save memory, clear the temporaries when the transform is
        no longer used.

        See Also
        --------
        clear_temporaries
        clear_fftw_plan : can also hold references to the temporaries
        """
        inverse = isinstance(self, FourierTransformInverse)

        if inverse:
            rspace = self.range
            fspace = self.domain
        else:
            rspace = self.domain
            fspace = self.range

        if r:
            self._tmp_r = rspace.element().asarray()
        if f:
            self._tmp_f = fspace.element().asarray()

    def clear_temporaries(self):
        """Set the temporaries to ``None``."""
        self._tmp_r = None
        self._tmp_f = None

    def init_fftw_plan(self, planning_effort='measure', **kwargs):
        """Initialize the FFTW plan for this transform for later use.

        If the implementation of this operator is not 'pyfftw', this
        method should not be called.

        Parameters
        ----------
        planning_effort : {'estimate', 'measure', 'patient', 'exhaustive'}
            Flag for the amount of effort put into finding an optimal
            FFTW plan. See the `FFTW doc on planner flags
            <http://www.fftw.org/fftw3_doc/Planner-Flags.html>`_.
        planning_timelimit : float or ``None``, optional
            Limit planning time to roughly this many seconds.
            Default: ``None`` (no limit)
        threads : int, optional
            Number of threads to use. Default: 1

        Raises
        ------
        ValueError
            If `impl` is not 'pyfftw'

        Notes
        -----
        To save memory, clear the plan when the transform is no longer
        used (the plan stores 2 arrays).

        See Also
        --------
        clear_fftw_plan
        """
        if self.impl != 'pyfftw':
            raise ValueError('cannot create fftw plan without fftw backend')

        # Using available temporaries if possible
        inverse = isinstance(self, FourierTransformInverse)

        if inverse:
            rspace = self.range
            fspace = self.domain
        else:
            rspace = self.domain
            fspace = self.range

        if rspace.field == ComplexNumbers():
            # C2C: Use either one of 'r' or 'f' temporary if initialized
            if self._tmp_r is not None:
                arr_in = arr_out = self._tmp_r
            elif self._tmp_f is not None:
                arr_in = arr_out = self._tmp_f
            else:
                arr_in = arr_out = rspace.element().asarray()

        elif self.halfcomplex:
            # R2HC / HC2R: Use 'r' and 'f' temporary distinctly if initialized
            if self._tmp_r is not None:
                arr_r = self._tmp_r
            else:
                arr_r = rspace.element().asarray()
            if self._tmp_f is not None:
                arr_f = self._tmp_f
            else:
                arr_f = fspace.element().asarray()

            if inverse:
                arr_in, arr_out = arr_f, arr_r
            else:
                arr_in, arr_out = arr_r, arr_f

        else:
            # R2C / C2R: Use 'f' temporary for both sides if initialized
            if self._tmp_f is not None:
                arr_in = arr_out = self._tmp_f
            else:
                arr_in = arr_out = fspace.element().asarray()

        kwargs.pop('planning_timelimit', None)

        direction = 'forward' if self.sign == '-' else 'backward'
        self._fftw_plan = pyfftw_call(
            arr_in, arr_out, direction=direction,
            halfcomplex=self.halfcomplex, axes=self.axes,
            planning_effort=planning_effort, **kwargs)

    def clear_fftw_plan(self):
        """Delete the FFTW plan of this transform.

        Raises
        ------
        ValueError
            If `impl` is not 'pyfftw'

        Notes
        -----
        If no plan exists, this is a no-op.
        """

        if self.impl != 'pyfftw':
            raise ValueError('cannot create fftw plan without fftw backend')

        self._fftw_plan = None


class FourierTransform(FourierTransformBase):

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

    See Also
    --------
    DiscreteFourierTransform
    FourierTransformInverse
    dft_preprocess_data
    pyfftw_call
    dft_postprocess_data
    """

    def __init__(self, domain, range=None, impl='numpy', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscreteLp`
            Domain of the Fourier transform. If the
            `DiscreteLp.exponent` of ``domain`` and ``range`` are equal
            to 2.0, this operator has an adjoint which is equal to its
            inverse.
        range : `DiscreteLp`, optional
            Range of the Fourier transform. If not given, the range
            is determined from ``domain`` and the other parameters. The
            exponent is chosen to be the conjugate ``p / (p - 1)``,
            which reads as 'inf' for p=1 and 1 for p='inf'.
        impl : {'numpy', 'pyfftw'}
            Backend for the FFT implementation. The 'pyfftw' backend
            is faster but requires the ``pyfftw`` package.
        axes : int or sequence of ints, optional
            Dimensions along which to take the transform.
            Default: all axes
        sign : {'-', '+'}, optional
            Sign of the complex exponent. Default: '-'
        halfcomplex : bool, optional
            If ``True``, calculate only the negative frequency part
            along the last axis for real input. If ``False``,
            calculate the full complex FFT.
            For complex ``domain``, it has no effect.
            Default: ``True``
        shift : bool or sequence of bools, optional
            If ``True``, the reciprocal grid is shifted by half a stride in
            the negative direction. With a boolean sequence, this option
            is applied separately to each axis.
            If a sequence is provided, it must have the same length as
            ``axes`` if supplied. Note that this must be set to ``True``
            in the halved axis in half-complex transforms.
            Default: ``True``

        Other Parameters
        ----------------
        tmp_r : `DiscreteLpElement` or `numpy.ndarray`
            Temporary for calculations in the real space (domain of
            this transform). It is shared with the inverse.

            Variants using this: R2C, R2HC, C2R (inverse)

        tmp_f : `DiscreteLpElement` or `numpy.ndarray`
            Temporary for calculations in the frequency (reciprocal)
            space. It is shared with the inverse.

            Variants using this: R2C, C2R (inverse), HC2R (inverse)

        Notes
        -----
        * The transform variants are:

          - **C2C**: complex-to-complex.
            The default variant, one-to-one and unitary.

          - **R2C**: real-to-complex.
            This variants adjoint and inverse may suffer
            from information loss since the result is cast to real.

          - **R2HC**: real-to-halfcomplex.
            This variant stores only a half-space of frequencies and
            is guaranteed to be one-to-one (invertible).

        * The `Operator.range` of this operator always has the
          `ComplexNumbers` as `LinearSpace.field`, i.e. if the
          field of ``domain`` is the `RealNumbers`, this operator's adjoint
          is defined by identifying real and imaginary parts with
          the components of a real product space element.
          See the `mathematical background documentation
          <odlgroup.github.io/odl/math/trafos/fourier_transform.html#adjoint>`_
          for details.
        """
        super().__init__(inverse=False, domain=domain, range=range,
                         impl=impl, **kwargs)

    def _preprocess(self, x, out=None):
        """Return the pre-processed version of ``x``.

        C2C: use ``tmp_r`` or ``tmp_f`` (C2C operation)
        R2C: use ``tmp_f`` (R2C operation)
        HALFC: use ``tmp_r`` (R2R operation)

        The result is stored in ``out`` if given, otherwise in
        a temporary or a new array.
        """
        if out is None:
            if self.domain.field == ComplexNumbers():
                out = self._tmp_r if self._tmp_r is not None else self._tmp_f
            elif self.domain.field == RealNumbers() and not self.halfcomplex:
                out = self._tmp_f
            else:
                out = self._tmp_r
        return dft_preprocess_data(
            x, shift=self.shifts, axes=self.axes, sign=self.sign,
            out=out)

    def _postprocess(self, x, out=None):
        """Return the post-processed version of ``x``.

        C2C: use ``tmp_f`` (C2C operation)
        R2C: use ``tmp_f`` (C2C operation)
        HALFC: use ``tmp_f`` (C2C operation)

        The result is stored in ``out`` if given, otherwise in
        a temporary or a new array.
        """
        if out is None:
            if self.domain.field == ComplexNumbers():
                out = self._tmp_r if self._tmp_r is not None else self._tmp_f
            else:
                out = self._tmp_f
        return dft_postprocess_data(
            out, real_grid=self.domain.grid, recip_grid=self.range.grid,
            shift=self.shifts, axes=self.axes, sign=self.sign,
            interp=self.domain.interp, op='multiply', out=out)

    def _call_numpy(self, x):
        """Return ``self(x)`` for numpy back-end.

        Parameters
        ----------
        x : `numpy.ndarray`
            Array representing the function to be transformed

        Returns
        -------
        out : `numpy.ndarray`
            Result of the transform
        """
        # Pre-processing before calculating the DFT
        # Note: since the FFT call is out-of-place, it does not matter if
        # preprocess produces real or complex output in the R2C variant.
        # There is no significant time difference between (full) R2C and
        # C2C DFT in Numpy.
        preproc = self._preprocess(x)

        # The actual call to the FFT library, out-of-place unfortunately
        if self.halfcomplex:
            out = np.fft.rfftn(preproc, axes=self.axes)
        else:
            if self.sign == '-':
                out = np.fft.fftn(preproc, axes=self.axes)
            else:
                out = np.fft.ifftn(preproc, axes=self.axes)
                # Numpy's FFT normalizes by 1 / prod(shape[axes]), we
                # need to undo that
                out *= np.prod(np.take(self.domain.shape, self.axes))

        # Post-processing accounting for shift, scaling and interpolation
        self._postprocess(out, out=out)
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
        planning_timelimit : float or ``None``, optional
            Limit planning time to roughly this many seconds.
            Default: ``None`` (no limit)
        threads : int, optional
            Number of threads to use. Default: 1

        Returns
        -------
        out : `numpy.ndarray`
            Result of the transform. The returned object is a reference
            to the input parameter ``out``.
        """
        # We pop some kwargs options here so that we always use the ones
        # given during init or implicitly assumed.
        kwargs.pop('axes', None)
        kwargs.pop('halfcomplex', None)
        kwargs.pop('normalise_idft', None)  # We use `False`

        # Pre-processing before calculating the sums, in-place for C2C and R2C
        if self.halfcomplex:
            preproc = self._preprocess(x)
            assert is_real_dtype(preproc.dtype)
        else:
            # out is preproc in this case
            preproc = self._preprocess(x, out=out)
            assert is_complex_floating_dtype(preproc.dtype)

        # The actual call to the FFT library. We store the plan for re-use.
        # The FFT is calculated in-place, except if the range is real and
        # we don't use halfcomplex.
        direction = 'forward' if self.sign == '-' else 'backward'
        self._fftw_plan = pyfftw_call(
            preproc, out, direction=direction, halfcomplex=self.halfcomplex,
            axes=self.axes, normalise_idft=False, **kwargs)

        assert is_complex_floating_dtype(out.dtype)

        # Post-processing accounting for shift, scaling and interpolation
        out = self._postprocess(out, out=out)
        assert is_complex_floating_dtype(out.dtype)
        return out

    @property
    def inverse(self):
        """The inverse Fourier transform."""
        sign = '+' if self.sign == '-' else '-'
        return FourierTransformInverse(
            domain=self.range, range=self.domain, impl=self.impl,
            axes=self.axes, halfcomplex=self.halfcomplex, shift=self.shifts,
            sign=sign, tmp_r=self._tmp_r, tmp_f=self._tmp_f)


class FourierTransformInverse(FourierTransformBase):

    """Inverse of the discretized Fourier transform between L^p spaces.

    This operator is the exact inverse of the `FourierTransform`, and
    **not** a discretization of the Fourier integral with "+" sign in
    the complex exponent. For the latter, use the ``sign`` parameter
    of the forward transform.

    See Also
    --------
    FourierTransform
    DiscreteFourierTransformInverse
    """

    def __init__(self, range, domain=None, impl='numpy', **kwargs):
        """
        Parameters
        ----------
        range : `DiscreteLp`
            Range of the inverse Fourier transform. If the
            `DiscreteLp.exponent` of ``domain`` and ``range`` are equal
            to 2.0, this operator has an adjoint which is equal to its
            inverse.
        domain : `DiscreteLp`, optional
            Domain of the inverse Fourier transform. If not given, the
            domain is determined from ``range`` and the other parameters.
            The exponent is chosen to be the conjugate ``p / (p - 1)``,
            which reads as 'inf' for p=1 and 1 for p='inf'.
        impl : {'numpy', 'pyfftw'}
            Backend for the FFT implementation. The 'pyfftw' backend
            is faster but requires the ``pyfftw`` package.
        axes : int or sequence of ints, optional
            Dimensions along which to take the transform.
            Default: all axes
        sign : {'-', '+'}, optional
            Sign of the complex exponent. Default: ``'+'``
        halfcomplex : bool, optional
            If ``True``, calculate only the negative frequency part
            along the last axis for real input. If ``False``,
            calculate the full complex FFT.
            For complex ``domain``, it has no effect.
            Default: ``True``
        shift : bool or sequence of bools, optional
            If ``True``, the reciprocal grid is shifted by half a stride in
            the negative direction. With a boolean sequence, this option
            is applied separately to each axis.
            If a sequence is provided, it must have the same length as
            ``axes`` if supplied. Note that this must be set to ``True``
            in the halved axis in half-complex transforms.
            Default: ``True``

        Other Parameters
        ----------------
        tmp_r : `DiscreteLpElement` or `numpy.ndarray`
            Temporary for calculations in the real space (range of
            this transform). It is shared with the inverse.

            Variants using this: C2R, R2C (forward), R2HC (forward)

        tmp_f : `DiscreteLpElement` or `numpy.ndarray`
            Temporary for calculations in the frequency (reciprocal)
            space. It is shared with the inverse.

            Variants using this: C2R, HC2R, R2C (forward)

        Notes
        -----
        * The transform variants are:

          - **C2C**: complex-to-complex.
            The default variant, one-to-one and unitary.

          - **C2R**: complex-to-real.
            This variants adjoint and inverse may suffer
            from information loss since the result is cast to real.

          - **HC2R**: halfcomplex-to-real.
            This variant interprets input as a signal on a half-space
            of frequencies. It is guaranteed to be one-to-one
            (invertible).

        * The `Operator.range` of this operator always has the
          `ComplexNumbers` as `LinearSpace.field`, i.e. if the
          field of ``domain`` is the `RealNumbers`, this operator's adjoint
          is defined by identifying real and imaginary parts with
          the components of a real product space element.
          See the `mathematical background documentation
          <odlgroup.github.io/odl/math/trafos/fourier_transform.html#adjoint>`_
          for details.
        """
        # TODO: variants wrt placement of 2*pi
        super().__init__(inverse=True, domain=range, range=domain,
                         impl=impl, **kwargs)

    def _preprocess(self, x, out=None):
        """Return the pre-processed version of ``x``.

        Note that pre-processing in IFT is the same as post-processing
        in FT with ``op='divide'``.

        C2C: use ``tmp_r`` or``tmp_f`` (C2C operation)
        R2C: use ``tmp_f`` (C2C operation)
        HALFC: use ``tmp_f`` (C2C operation)

        The result is stored in ``out`` if given, otherwise in
        a temporary or a new array.
        """
        if out is None:
            if self.range.field == ComplexNumbers():
                out = self._tmp_r if self._tmp_r is not None else self._tmp_f
            else:
                out = self._tmp_f
        return dft_postprocess_data(
            x, real_grid=self.range.grid, recip_grid=self.domain.grid,
            shift=self.shifts, axes=self.axes, sign=self.sign,
            interp=self.domain.interp, op='divide', out=out)

    def _postprocess(self, x, out=None):
        """Return the post-processed version of ``x``.

        C2C: use ``tmp_r`` or ``tmp_f`` (C2C operation)
        R2C: use ``tmp_f`` (C2C operation)
        HALFC: use ``tmp_r`` (R2R operation)

        The result is stored in ``out`` if given, otherwise in
        a temporary or a new array.
        """
        if out is None:
            if self.range.field == ComplexNumbers():
                out = self._tmp_r if self._tmp_r is not None else self._tmp_f
            elif self.range.field == RealNumbers() and not self.halfcomplex:
                out = self._tmp_f
            else:  # halfcomplex
                out = self._tmp_r
        return dft_preprocess_data(
            x, shift=self.shifts, axes=self.axes, sign=self.sign, out=out)

    def _call_numpy(self, x):
        """Return ``self(x)`` for numpy back-end.

        Parameters
        ----------
        x : `numpy.ndarray`
            Array representing the function to be transformed

        Returns
        -------
        out : `numpy.ndarray`
            Result of the transform
        """
        # Pre-processing before calculating the DFT
        preproc = self._preprocess(x)

        # The actual call to the FFT library
        # Normalization by 1 / prod(shape[axes]) is done by Numpy's FFT if
        # one of the "i" functions is used. For sign='-' we need to do it
        # ourselves.
        if self.halfcomplex:
            s = np.asarray(self.range.shape)[list(self.axes)]
            out = np.fft.irfftn(preproc, axes=self.axes, s=s)
        else:
            if self.sign == '-':
                out = np.fft.fftn(preproc, axes=self.axes)
                out /= np.prod(np.take(self.domain.shape, self.axes))
            else:
                out = np.fft.ifftn(preproc, axes=self.axes)

        # Post-processing in IFT = pre-processing in FT (in-place)
        self._postprocess(out, out=out)
        if self.halfcomplex:
            assert is_real_dtype(out.dtype)

        if self.range.field == RealNumbers():
            return out.real
        else:
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
        planning_timelimit : float or ``None``, optional
            Limit planning time to roughly this many seconds.
            Default: ``None`` (no limit)
        threads : int, optional
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
        kwargs.pop('normalise_idft', None)  # We use `True`

        # Pre-processing in IFT = post-processing in FT, but with division
        # instead of multiplication and switched grids. In-place for C2C only.
        if self.range.field == ComplexNumbers():
            # preproc is out in this case
            preproc = self._preprocess(x, out=out)
        else:
            preproc = self._preprocess(x)

        # The actual call to the FFT library. We store the plan for re-use.
        direction = 'forward' if self.sign == '-' else 'backward'
        if self.range.field == RealNumbers() and not self.halfcomplex:
            # Need to use a complex array as out if we do C2R since the
            # FFT has to be C2C
            self._fftw_plan = pyfftw_call(
                preproc, preproc, direction=direction,
                halfcomplex=self.halfcomplex, axes=self.axes,
                normalise_idft=True, **kwargs)
            fft_arr = preproc
        else:
            # Only here we can use out directly
            self._fftw_plan = pyfftw_call(
                preproc, out, direction=direction,
                halfcomplex=self.halfcomplex, axes=self.axes,
                normalise_idft=True, **kwargs)
            fft_arr = out

        # Normalization is only done for 'backward', we need it for 'forward',
        # too.
        if self.sign == '-':
            fft_arr /= np.prod(np.take(self.domain.shape, self.axes))

        # Post-processing in IFT = pre-processing in FT. In-place for
        # C2C and HC2R. For C2R, this is out-of-place and discards the
        # imaginary part.
        self._postprocess(fft_arr, out=out)
        return out

    @property
    def inverse(self):
        """Inverse of the inverse, the forward FT."""
        sign = '+' if self.sign == '-' else '-'
        return FourierTransform(
            domain=self.range, range=self.domain, impl=self.impl,
            axes=self.axes, halfcomplex=self.halfcomplex, shift=self.shifts,
            sign=sign, tmp_r=self._tmp_r, tmp_f=self._tmp_f)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
