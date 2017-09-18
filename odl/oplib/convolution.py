# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Discretized continuous convolution and fully discrete convolution."""

from __future__ import division
import numpy as np

from odl.operator import Operator
from odl.space import tensor_space
from odl.space.base_tensors import TensorSpace, Tensor
from odl.trafos.backends import PYFFTW_AVAILABLE
from odl.util import is_real_dtype, is_floating_dtype, dtype_str

__all__ = ('DiscreteConvolution',)


class DiscreteConvolution(Operator):

    """Fully discrete convolution with a given kernel."""

    def __init__(self, domain, kernel, range=None, axis=None, impl='fft',
                 **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `TensorSpace`
            Space on which the convolution is defined. If ``domain`` is
            a `DiscreteLp`, it must be uniformly discretized.
        kernel : array-like
            The kernel with which input elements are convolved. It must
            have the same number of dimensions as ``domain``, and its
            shape can be at most equal to ``domain.shape``. In axes
            with size 1, broadcasting is applied.

            **Important:**

            - The kernel must **always** have the same number of dimensions
              as ``domain``, even for convolutions along axes.
            - In axes where no convolution is performed, the shape of the
              kernel must either be 1 (broadcasting along these axes), or
              equal to the domain shape (stacked kernel, only supported
              for `impl='fft'`).
            - The ``'fft'`` implementation needs the kernel to have
              floating-point ``dtype``, hence the smallest possible
              float data type is used in that case to store the kernel.
            - If the convolution kernel is complex, the ``domain`` **must**
              be a complex space. Real-to-complex convolutions are not
              allowed and need to defined by composition with
              `ComplexEmbedding`.

            See Examples for further clarification.

        range : `TensorSpace`, optional
            Space of output elements of the convolution. Must be of the
            same shape as ``domain``. If not given, the range is equal to
            ``domain.astype(result_dtype)``, where ``result_dtype`` is
            the data type of the convolution result. If ``impl='real'``,
            integer dtypes are preserved, while for ``impl='fft'``,
            the smallest possible floating-point type is chosen.
        axis : int or sequence of ints, optional
            Coordinate axis or axes in which to take the convolution.
            ``None`` means all input axes.
        impl : {'fft', 'real'}
            Implementation of the convolution as FFT-based or using
            direct summation. The fastest available FFT backend is
            chosen automatically. Real space convolution is based on
            `scipy.signal.convolve`.
            See Notes for further information on the backends.

        Other Parameters
        ----------------
        padding : int or sequence of ints, optional
            Zero-padding used before Fourier transform in the FFT backend.
            Does not apply for ``impl='real'``. A sequence is applied per
            axis, with padding values corresponding to ``axis`` entries
            as provided.
            Default: ``min(kernel.shape - 1, 64)``
        padded_shape : sequence of ints, optional
            Apply zero-padding with this target shape. Cannot be used
            together with ``padding``.
        cache_kernel_ft : bool, optional
            If ``True``, store the Fourier transform of the kernel for
            later reuse.
            Default: ``False``

        Notes
        -----
        - In general, the out-of-place call (no ``out`` parameter) is
          expected to be faster for this operator for a variety of reasons:

          * The ``impl='real'`` backend does not support an `out` argument
            and will thus create a new array anyway. Providing `out` will
            require an additional copy from the new array to `out`.
          * For ``impl='fft'``, the intermediate padding and unpadding steps
            make it impossible to use an `out` object efficiently since
            size and/or dtype do not match the requirements.
            In addition, the Numpy FFT backend does not support an
            `out` parameter.

        - ``scipy.convolve`` does not support an ``axis`` parameter.
          However, a convolution along axes with a lower-dimensional
          kernel can be achieved by adding empty dimensions. For example,
          to convolve along axis 0 we can do the following ::

              ker_1d = np.array([-1.0, 1.0])
              ker_axis0 = ker_1d[:, None]
              conv = DiscreteConvolution(space_2d, ker_axis0, impl='real')

          Not possible with this approach is a convolution with a
          *different* kernel in each column ("stacked" kernel).

        - The NumPy FFT backend always uses ``'float64'/'complex128'``
          internally, so different data types will simply result in
          additional casting, not speedup or different precision.

        Examples
        --------
        Convolve in all axes:

        >>> space = odl.rn((3, 3))
        >>> kernel = [[0, 0, 0],  # A discrete Dirac delta
        ...           [0, 1, 0],
        ...           [0, 0, 0]]
        >>> conv = DiscreteConvolution(space, kernel)
        >>> x = space.element([[1, 2, 3],
        ...                    [2, 4, 6],
        ...                    [-3, -6, -9]])
        >>> conv(x)
        rn((3, 3)).element(
            [[ 1.,  2.,  3.],
             [ 2.,  4.,  6.],
             [-3., -6., -9.]]
        )

        For even-sized kernels, the convolution is performed in a
        "backwards" manner, i.e., the lower indices are affected by
        implicit zero-padding:

        >>> kernel = [[1, 1],  # 2x2 blurring kernel
        ...           [1, 1]]
        >>> conv = DiscreteConvolution(space, kernel)
        >>> x = space.element([[1, 2, 3],
        ...                    [2, 4, 6],
        ...                    [-3, -6, -9]])
        >>> conv(x)
        rn((3, 3)).element(
            [[  1.,   3.,   5.],
             [  3.,   9.,  15.],
             [ -1.,  -3.,  -5.]]
        )

        Convolution in selected axes can be done either with broadcasting
        or with "stacked kernels":

        >>> kernel_1d = [1, -1]  # backward difference kernel
        >>> kernel = np.array(kernel_1d)[None, :]  # broadcasting in axis 0
        >>> conv = DiscreteConvolution(space, kernel, axis=1)
        >>> x = space.element([[1, 2, 3],
        ...                    [2, 4, 6],
        ...                    [-3, -6, -9]])
        >>> conv(x)
        rn((3, 3)).element(
            [[ 1.,  1.,  1.],
             [ 2.,  2.,  2.],
             [-3., -3., -3.]]
        )
        >>> kernel_stack = [[1, -1],  # separate kernel per row
        ...                 [2, -2],
        ...                 [3, -3]]
        >>> conv = DiscreteConvolution(space, kernel_stack, axis=1)
        >>> conv(x)
        rn((3, 3)).element(
            [[ 1.,  1.,  1.],
             [ 4.,  4.,  4.],
             [-9., -9., -9.]]
        )
        """
        # Avoid name clash with range iterator
        import builtins
        range, ran = builtins.range, range

        if not isinstance(domain, TensorSpace):
            raise TypeError('`domain` must be a TensorSpace, got {}'
                            ''.format(type(domain)))

        if not isinstance(kernel, Tensor):
            with np.warnings.catch_warnings():
                np.warnings.filterwarnings('error', category=np.ComplexWarning)
                kernel = np.asarray(kernel, dtype=domain.dtype)

            ker_space = tensor_space(kernel.shape, kernel.dtype)
            kernel = ker_space.element(kernel)

        if ran is None:
            if str(impl).lower() == 'fft':
                # Need a floating point dtype for FFT
                result_dtype = np.result_type(domain.dtype, np.float16)
            else:
                result_dtype = domain.dtype
            ran = domain.astype(result_dtype)

        # Disallow real-to-complex convolutions
        if domain.is_real and ran.is_complex:
            raise ValueError('cannot combine `domain` with real dtype {} '
                             'and `range` with complex dtype {}'
                             ''.format(dtype_str(domain.dtype),
                                       dtype_str(ran.dtype)))

        super(DiscreteConvolution, self).__init__(domain, ran, linear=True)
        self.__kernel = kernel

        ndim = self.domain.ndim
        if kernel.ndim != ndim:
            raise ValueError('`kernel` must have {} (=ndim) dimensions, but '
                             'got a {}-dimensional kernel'
                             ''.format(ndim, kernel.ndim))

        if axis is None:
            self.__axes = tuple(range(ndim))
        else:
            try:
                iter(axis)
            except TypeError:
                self.__axes = (int(axis),)
            else:
                self.__axes = tuple(int(ax) for ax in axis)

        if not all(-ndim <= ax < ndim for ax in self.axes):
            raise ValueError('`axis` must (all) satisfy -{n} <= axis < {n}, '
                             'got {}'.format(axis, n=ndim))

        for i in range(ndim):
            if i in self.axes and kernel.shape[i] > self.domain.shape[i]:
                raise ValueError(
                    'kernel size in convolution axis {} can at most be equal '
                    'to domain size {}, but got size {}'
                    ''.format(i, self.domain.shape[i], kernel.shape[i]))
            elif (i not in self.axes and
                  kernel.shape[i] not in (1, self.domain.shape[i])):
                raise ValueError(
                    'kernel size in non-convolution axis {} must be either 1 '
                    '(broadcasting) or equal to domain size {}, but got size '
                    '{}'.format(i, self.domain.shape[i], kernel.shape[i]))

        self.__impl = str(impl).lower()
        if self.impl == 'real':
            self.__real_impl = 'scipy'
            self.__fft_impl = None
        elif self.impl == 'fft':
            self.__real_impl = None
            self.__fft_impl = 'pyfftw' if PYFFTW_AVAILABLE else 'numpy'
        else:
            raise ValueError('unknown `impl` {!r}'.format(impl))

        if self.impl == 'real':
            for i in range(ndim):
                if i not in self.axes and kernel.shape[i] != 1:
                    raise ValueError(
                        "for `impl='real', all non-convolution axes must "
                        'have size 1, but got size {} in axis {}'
                        ''.format(kernel.shape[i], i))

        # Handle padding and padded_shape
        padding = kwargs.pop('padding', None)
        padded_shape = kwargs.pop('padded_shape', None)
        if padding is not None and padded_shape is not None:
            raise TypeError('cannot give both `padding` and `padded_shape`')

        if padded_shape is not None and len(padded_shape) != ndim:
            raise ValueError('`padded_shape` contains an invalid number of '
                             'entries: need {} (=ndim), got {}'
                             ''.format(ndim, len(padded_shape)))

        if padding is None:
            full_padding = np.minimum(np.array(kernel.shape) - 1, 64)
            padding = [full_padding[i] if i in self.axes else 0
                       for i in range(ndim)]
        else:
            try:
                iter(padding)
            except TypeError:
                padding = [int(padding) if i in self.axes else 0
                           for i in range(ndim)]
            else:
                padding = [int(p) for p in padding]
                if len(padding) == len(self.axes):
                    padding_lst = [0] * self.domain.ndim
                    for ax, pad in zip(self.axes, padding):
                        padding_lst[ax] = pad
                    padding = padding_lst

        if len(padding) != ndim:
            raise ValueError('`padding` contains an invalid number of '
                             'entries: need {} (=ndim), got {}'
                             ''.format(ndim, len(padding)))

        if padded_shape is None:
            padded_shape = tuple(np.array(self.domain.shape) + padding)

        self.__padded_shape = tuple(padded_shape)

        for i in range(ndim):
            if self.padded_shape[i] < self.domain.shape[i]:
                raise ValueError(
                    '`padded_shape` in axis {} must be larger than or equal '
                    'to domain size {}, but got {}'
                    ''.format(i, self.domain.shape[i], self.padded_shape[i]))

        self.__cache_kernel_ft = bool(kwargs.pop('cache_kernel_ft', False))
        self._kernel_ft = None

        if kwargs:
            raise TypeError('got unexpected kwargs {}'.format(kwargs))

    @property
    def kernel(self):
        """The `Tensor` used as kernel in the convolution."""
        return self.__kernel

    @property
    def axes(self):
        """The dimensions along which the convolution is taken."""
        return self.__axes

    @property
    def impl(self):
        """Implementation variant, ``'fft' or 'real'``."""
        return self.__impl

    @property
    def real_impl(self):
        """Backend for real-space conv., or ``None`` if not applicable."""
        return self.__real_impl

    @property
    def fft_impl(self):
        """Backend used for FFTs, or ``None`` if not applicable."""
        return self.__fft_impl

    @property
    def padded_shape(self):
        """Domain shape after padding for FFT-based convolution."""
        return self.__padded_shape

    @property
    def cache_kernel_ft(self):
        """If ``True``, the kernel FT is cached for later reuse."""
        return self.__cache_kernel_ft

    def _call(self, x, out=None):
        """Perform convolution of ``f`` with `kernel`."""
        if self.impl == 'real' and self.real_impl == 'scipy':
            return self._call_scipy_convolve(x, out)
        elif self.impl == 'fft' and self.fft_impl == 'numpy':
            return self._call_numpy_fft(x, out)
        elif self.impl == 'fft' and self.fft_impl == 'pyfftw':
            return self._call_pyfftw(x, out)
        else:
            raise RuntimeError('bad `impl` {!r} or `fft_impl` {!r}'
                               ''.format(self.impl, self.fft_impl))

    def _call_scipy_convolve(self, x, out=None):
        """Perform real-space convolution using ``scipy.signal.convolve``."""
        import scipy.signal

        conv = scipy.signal.convolve(x, self.kernel, mode='same',
                                     method='direct')
        if out is None:
            out = conv
        else:
            out[:] = conv
        return out

    def _call_numpy_fft(self, x, out=None):
        """Perform FFT-based convolution using NumPy's backend."""
        # Use real-to-complex FFT if possible, it's faster
        if (is_real_dtype(self.kernel.dtype) and
                is_real_dtype(self.domain.dtype)):
            fft = np.fft.rfftn
            ifft = np.fft.irfftn
        else:
            fft = np.fft.fftn
            ifft = np.fft.ifftn

        # Prepare kernel, preserving length-1 axes for broadcasting
        ker_padded_shp = [1 if self.kernel.shape[i] == 1
                          else self.padded_shape[i]
                          for i in range(self.domain.ndim)]
        kernel_prep = prepare_for_fft(self.kernel, ker_padded_shp, self.axes)

        # Pad the input with zeros
        paddings = []
        for i in range(self.domain.ndim):
            diff = self.padded_shape[i] - x.shape[i]
            left = diff // 2
            right = diff - left
            paddings.append((left, right))

        x_prep = np.pad(x, paddings, 'constant')

        # Perform FFTs of x and kernel (or retrieve from cache)
        x_ft = fft(x_prep, axes=self.axes)

        if self._kernel_ft is not None:
            kernel_ft = self._kernel_ft
        else:
            kernel_ft = fft(kernel_prep, axes=self.axes)
            if self.cache_kernel_ft:
                self._kernel_ft = kernel_ft

        # Multiply `x_ft` with `kernel_ft` and transform back. Note that
        # both have dtype 'float64' since that's what `numpy.fft` always uses.
        x_ft *= kernel_ft
        # `irfft` needs an explicit shape, otherwise the result shape may not
        # be the same as the original one
        s = [x_prep.shape[i]
             for i in range(self.domain.ndim) if i in self.axes]
        ifft_x = ifft(x_ft, axes=self.axes, s=s)

        # Unpad to get the "relevant" part
        slc = [slice(l, n - r) for (l, r), n in zip(paddings, x_prep.shape)]
        if out is None:
            out = ifft_x[slc]
        else:
            out[:] = ifft_x[slc]

        return out

    def _call_pyfftw(self, x, out=None):
        """Perform FFT-based convolution using the pyfftw backend."""
        import multiprocessing
        import pyfftw

        # Pad the input with zeros
        paddings = []
        for i in range(self.domain.ndim):
            diff = self.padded_shape[i] - x.shape[i]
            left = diff // 2
            right = diff - left
            paddings.append((left, right))

        x_prep = np.pad(x, paddings, 'constant')
        x_prep_shape = x_prep.shape

        # Real-to-halfcomplex only if both domain and kernel are eligible
        use_halfcx = (is_real_dtype(self.domain.dtype) and
                      is_real_dtype(self.kernel.dtype))

        def fft_out_array(arr, use_halfcx):
            """Make an output array for FFTW with suitable dtype and shape."""
            ft_dtype = np.result_type(arr.dtype, 1j)
            ft_shape = list(arr.shape)
            if use_halfcx:
                ft_shape[self.axes[-1]] = ft_shape[self.axes[-1]] // 2 + 1
            return np.empty(ft_shape, ft_dtype)

        # Perform FFT of `x`. Use 'FFTW_ESTIMATE', since other options destroy
        # the input and would require a copy.
        x_ft = fft_out_array(x_prep, use_halfcx)
        if not use_halfcx and x_ft.dtype != x_prep.dtype:
            # Need to perform C2C transform, hence a cast
            x_prep = x_prep.astype(x_ft.dtype)
        elif x_prep.dtype == 'float16':
            # No native support for half floats
            x_prep = x_prep.astype('float32')

        plan_x = pyfftw.FFTW(x_prep, x_ft, axes=self.axes,
                             direction='FFTW_FORWARD',
                             flags=['FFTW_ESTIMATE'],
                             threads=multiprocessing.cpu_count())
        plan_x(x_prep, x_ft)
        plan_x = None  # can be gc'ed
        x_prep = None

        # Perform FFT of kernel if necessary
        if self._kernel_ft is not None:
            kernel_ft = self._kernel_ft
        else:
            # Prepare kernel, preserving length-1 axes for broadcasting
            if is_floating_dtype(self.kernel.dtype):
                kernel = self.kernel
            else:
                flt_dtype = np.result_type(self.kernel.dtype, np.float16)
                kernel = np.asarray(self.kernel, dtype=flt_dtype)

            ker_padded_shp = [1 if self.kernel.shape[i] == 1
                              else self.padded_shape[i]
                              for i in range(self.domain.ndim)]
            kernel_prep = prepare_for_fft(kernel, ker_padded_shp, self.axes)
            kernel = None  # can be gc'ed

            kernel_ft = fft_out_array(kernel_prep, use_halfcx)
            if not use_halfcx and kernel_ft.dtype != kernel_prep.dtype:
                # Need to perform C2C transform, hence a cast
                kernel_prep = kernel_prep.astype(kernel_ft.dtype)
            elif kernel_prep.dtype == 'float16':
                # No native support
                kernel_prep = kernel_prep.astype('float32')

            plan_kernel = pyfftw.FFTW(kernel_prep, kernel_ft, axes=self.axes,
                                      direction='FFTW_FORWARD',
                                      flags=['FFTW_ESTIMATE'],
                                      threads=multiprocessing.cpu_count())
            plan_kernel(kernel_prep, kernel_ft)
            plan_kernel = None  # can be gc'ed
            kernel_prep = None

            if self.cache_kernel_ft:
                self._kernel_ft = kernel_ft

        # Multiply x_ft with kernel_ft and transform back. Some care
        # is required with respect to dtypes, in particular when
        # x_ft.dtype < kernel_ft.dtype.
        if x_ft.dtype < kernel_ft.dtype:
            x_ft = x_ft * kernel_ft
        else:
            x_ft *= kernel_ft

        # Perform inverse FFT
        if use_halfcx:
            x_ift_dtype = np.empty(0, dtype=x_ft.dtype).real.dtype
        else:
            x_ift_dtype = x_ft.dtype
        x_ift = np.empty(x_prep_shape, x_ift_dtype)
        plan_ift = pyfftw.FFTW(x_ft, x_ift, axes=self.axes,
                               direction='FFTW_BACKWARD',
                               flags=['FFTW_ESTIMATE'],
                               threads=multiprocessing.cpu_count())

        plan_ift(x_ft, x_ift)
        x_ft = None  # can be gc'ed

        # Unpad to get the "relevant" part
        slc = [slice(l, n - r) for (l, r), n in zip(paddings, x_prep_shape)]
        if out is None:
            out = x_ift[slc]
        else:
            out[:] = x_ift[slc]

        return out


def prepare_for_fft(kernel, padded_shape, axes=None):
    """Return a kernel with desired shape with middle entry at index 0.

    This function applies the appropriate steps to prepare a kernel for
    FFT-based convolution. It first pads the kernel with zeros *to the
    right* up to ``padded_shape``, and then rolls the entries such that
    the old middle element, i.e., the one at ``(kernel.shape - 1) // 2``,
    lies at index 0.

    Parameters
    ----------
    kernel : array-like
        The kernel to be prepared for FFT convolution.
    padded_shape : sequence of ints
        The target shape to be reached by zero-padding.
    axes : sequence of ints, optional
        Dimensions in which to perform shifting. ``None`` means all axes.

    Returns
    -------
    prepared : `numpy.ndarray`
        The zero-padded and rolled kernel ready for FFT.

    Examples
    --------
    >>> kernel = np.array([[1, 2, 3],
    ...                    [4, 5, 6]])  # middle element is 2
    >>> prepare_for_fft(kernel, padded_shape=(4, 4))
    array([[2, 3, 0, 1],
           [5, 6, 0, 4],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    >>> prepare_for_fft(kernel, padded_shape=(5, 5))
    array([[2, 3, 0, 0, 1],
           [5, 6, 0, 0, 4],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    """
    kernel = np.asarray(kernel)
    if kernel.flags.f_contiguous and not kernel.flags.c_contiguous:
        order = 'F'
    else:
        order = 'C'

    padded = np.zeros(padded_shape, kernel.dtype, order)

    if axes is None:
        axes = list(range(kernel.ndim))

    if any(padded_shape[i] != kernel.shape[i] for i in range(kernel.ndim)
           if i not in axes):
        raise ValueError(
            '`padded_shape` can only differ from `kernel.shape` in `axes`; '
            'got `padded_shape={}`, `kernel.shape={}`, `axes={}`'
            ''.format(padded_shape, kernel.shape, axes))

    orig_slc = [slice(n) for n in kernel.shape]
    padded[orig_slc] = kernel
    # This shift makes sure that the middle element is shifted to index 0
    shift = [-((kernel.shape[i] - 1) // 2) if i in axes else 0
             for i in range(kernel.ndim)]
    return np.roll(padded, shift, axis=axes)


if __name__ == '__main__':
    from odl.util import run_doctests
    run_doctests()
