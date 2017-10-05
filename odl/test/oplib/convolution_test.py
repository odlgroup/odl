# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division
import numpy as np
import pytest
import scipy.signal

import odl
from odl.oplib import DiscreteConvolution
from odl.util.testutils import simple_fixture, noise_elements, all_almost_equal


# --- pytest fixtures --- #

size_1d = simple_fixture('size', [3, 16])
shape_2d = simple_fixture('shape', [(3, 4), (3, 8), (16, 4)])
ker_kind = simple_fixture('kind', ['smooth', 'diff1', 'diff2'])
conv_type = simple_fixture('type', ['full',
                                    'bcast0', 'bcast1',
                                    'stack0', 'stack1'])
conv_impl = simple_fixture('impl', ['fft', 'real'])


@pytest.fixture(scope='module')
def kernel_1d(ker_kind):
    if ker_kind == 'smooth':
        kernel = [1, 1]
    elif ker_kind == 'diff1':
        kernel = [1, -1]
    elif ker_kind == 'diff2':
        kernel = [1, -2, 1]
    else:
        assert False

    return ker_kind, np.array(kernel)


# --- DiscreteConvolution --- #


def test_dconv_init_and_properties():
    """Check init, error handling and props of ``DiscreteConvolution``."""
    # 1D case
    r3 = odl.rn(3, dtype='float32')
    conv = DiscreteConvolution(r3, [1, 1])
    assert conv.domain == conv.range == r3
    assert conv.is_linear
    assert conv.padded_shape == (4,)  # adds kernel_size - 1 by default
    assert conv.kernel.space == odl.rn(2, dtype='float32')
    assert conv.axes == (0,)
    assert conv.impl == 'fft'
    assert conv.real_impl is None

    ran = odl.rn(3, dtype=float)
    conv = DiscreteConvolution(r3, [1, 1], range=ran)
    assert conv.range == ran

    conv = DiscreteConvolution(r3, [1, 1], axis=0)
    assert conv.axes == (0,)

    conv = DiscreteConvolution(r3, [1, 1], impl='real')
    assert conv.impl == 'real'
    assert conv.real_impl == 'scipy'

    conv = DiscreteConvolution(r3, [1, 1], impl='fft')
    assert conv.impl == 'fft'

    conv = DiscreteConvolution(r3, [1, 1], padding=4)
    assert conv.padded_shape == (7,)

    conv = DiscreteConvolution(r3, [1, 1], padding=(4,))
    assert conv.padded_shape == (7,)

    conv = DiscreteConvolution(r3, [1, 1], padded_shape=(8,))
    assert conv.padded_shape == (8,)

    with pytest.raises(ValueError):
        DiscreteConvolution(r3, [[1, 1],
                                 [1, 1]])  # kernel has too many dims
    with pytest.raises(ValueError):
        DiscreteConvolution(r3, [1, 1, 1, 1])  # kernel too large
    with pytest.raises(ValueError):
        DiscreteConvolution(r3, [1j, 1j])  # complex kernel with real domain
    with pytest.raises(ValueError):
        # Real to complex not supported
        DiscreteConvolution(r3, [1, 1], range=odl.cn(3, dtype=complex))
    with pytest.raises(ValueError):
        DiscreteConvolution(r3, [1, 1], axis=1)  # axis out of bounds
    with pytest.raises(ValueError):
        DiscreteConvolution(r3, [1, 1], impl='pyfftw')  # bad impl
    with pytest.raises(ValueError):
        DiscreteConvolution(r3, [1, 1], padding=(2, 2))  # padding too long
    with pytest.raises(ValueError):
        # padded_shape too long
        DiscreteConvolution(r3, [1, 1], padded_shape=(8, 8))
    with pytest.raises(ValueError):
        # padded_shape cannot be smaller than original shape
        DiscreteConvolution(r3, [1, 1], padded_shape=(2,))
    with pytest.raises(TypeError):
        # cannot give both padding and padded_shape
        DiscreteConvolution(r3, [1, 1], padding=3, padded_shape=(9,))
    with pytest.raises(TypeError):
        DiscreteConvolution(r3, [1, 1], arg=0)  # bad kwarg

    # 2D case, full kernel
    rn = odl.rn((3, 4))
    kernel_full = [[1, 1],
                   [-1, -1]]

    conv = DiscreteConvolution(rn, kernel_full)
    assert conv.domain == conv.range == rn
    assert conv.padded_shape == (4, 5)  # + kernel.shape - 1
    assert conv.kernel.space == odl.rn((2, 2))
    assert conv.axes == (0, 1)

    ran = odl.rn((3, 4), dtype=float)
    conv = DiscreteConvolution(rn, kernel_full, range=ran)
    assert conv.range == ran

    conv = DiscreteConvolution(rn, kernel_full, axis=(1, 0))
    assert conv.axes == (1, 0)

    conv = DiscreteConvolution(rn, kernel_full, padding=4)
    assert conv.padded_shape == (7, 8)

    conv = DiscreteConvolution(rn, kernel_full, padding=(4, 8))
    assert conv.padded_shape == (7, 12)

    conv = DiscreteConvolution(rn, kernel_full, padded_shape=(8, 8))
    assert conv.padded_shape == (8, 8)

    with pytest.raises(ValueError):
        DiscreteConvolution(rn, [1, 1])  # kernel has too few dims
    with pytest.raises(ValueError):
        kernel_too_large_0 = [[1, 1]] * 5
        DiscreteConvolution(rn, kernel_too_large_0)
    with pytest.raises(ValueError):
        kernel_too_large_1 = [[1, 1, 1, 1, 1]] * 2
        DiscreteConvolution(rn, kernel_too_large_1)
    with pytest.raises(ValueError):
        DiscreteConvolution(rn, kernel_full, axis=0)  # too short in axis 1
    with pytest.raises(ValueError):
        DiscreteConvolution(rn, kernel_full, axis=1)  # too short in axis 0
    with pytest.raises(ValueError):
        DiscreteConvolution(rn, kernel_full, axis=(0, 1, 2))  # axis oob
    with pytest.raises(ValueError):
        # padding too long
        DiscreteConvolution(rn, kernel_full, padding=(2, 2, 2))
    with pytest.raises(ValueError):
        # padded_shape too long
        DiscreteConvolution(rn, kernel_full, padded_shape=(8, 8, 8))

    # 2D case, conv along axis 0, broadcast along axis 1
    rn = odl.rn((3, 4))
    kernel_bcast = [[1],
                    [1]]

    conv = DiscreteConvolution(rn, kernel_bcast)
    assert conv.kernel.space == odl.rn((2, 1))
    assert conv.padded_shape == (4, 4)  # + kernel_shape - 1
    assert conv.axes == (0, 1)

    conv = DiscreteConvolution(rn, kernel_bcast, axis=0)
    assert conv.padded_shape == (4, 4)  # + kernel_shape - 1 in conv axis
    assert conv.axes == (0,)

    conv = DiscreteConvolution(rn, kernel_bcast, axis=0, impl='real')
    assert conv.impl == 'real'

    conv = DiscreteConvolution(rn, kernel_bcast, axis=0, padding=4)
    assert conv.padded_shape == (7, 4)  # pad only in conv axis

    conv = DiscreteConvolution(rn, kernel_bcast, axis=0, padded_shape=(8, 4))
    assert conv.padded_shape == (8, 4)

    # 2D case, conv along axis 0, stack along axis 1
    rn = odl.rn((3, 4))
    kernel_stack = [[1, 1, 1, 1],
                    [1, 1, 1, 1]]
    conv = DiscreteConvolution(rn, kernel_stack, axis=0)
    assert conv.kernel.space == odl.rn((2, 4))
    assert conv.padded_shape == (4, 4)  # pad only in conv axis

    conv = DiscreteConvolution(rn, kernel_stack, axis=0, padding=4)
    assert conv.padded_shape == (7, 4)  # pad only in conv axis

    with pytest.raises(ValueError):
        # stacked kernels not supported by 'real' impl
        DiscreteConvolution(rn, kernel_stack, axis=0, impl='real')

    # 2D case, conv along axis 1, broadcast along axis 0
    rn = odl.rn((3, 4))
    kernel_bcast = [[1, 1]]

    conv = DiscreteConvolution(rn, kernel_bcast)
    assert conv.kernel.space == odl.rn((1, 2))
    assert conv.padded_shape == (3, 5)  # + kernel_shape - 1
    assert conv.axes == (0, 1)

    conv = DiscreteConvolution(rn, kernel_bcast, axis=1)
    assert conv.padded_shape == (3, 5)  # + kernel_shape - 1 in conv axis
    assert conv.axes == (1,)

    conv = DiscreteConvolution(rn, kernel_bcast, axis=1, impl='real')
    assert conv.impl == 'real'

    conv = DiscreteConvolution(rn, kernel_bcast, axis=1, padding=4)
    assert conv.padded_shape == (3, 8)  # pad only in conv axis

    conv = DiscreteConvolution(rn, kernel_bcast, axis=1, padded_shape=(3, 8))
    assert conv.padded_shape == (3, 8)

    # 2D case, conv along axis 1, stack along axis 0
    rn = odl.rn((3, 4))
    kernel_stack = [[1, 1],
                    [1, 1],
                    [1, 1]]
    conv = DiscreteConvolution(rn, kernel_stack, axis=1)
    assert conv.kernel.space == odl.rn((3, 2))
    assert conv.padded_shape == (3, 5)  # pad only in conv axis

    conv = DiscreteConvolution(rn, kernel_stack, axis=1, padding=4)
    assert conv.padded_shape == (3, 8)  # pad only in conv axis

    with pytest.raises(ValueError):
        # stacked kernels not supported by 'real' impl
        DiscreteConvolution(rn, kernel_stack, axis=1, impl='real')


def test_dconv_1d(size_1d, kernel_1d, floating_dtype, conv_impl):
    """Check discrete convolution in 1d."""
    ker_kind, kernel = kernel_1d
    rn = odl.tensor_space(size_1d, dtype=floating_dtype)
    conv = DiscreteConvolution(rn, kernel, impl=conv_impl)

    inp_arr, inp = noise_elements(rn)

    if ker_kind == 'smooth':
        expected = scipy.signal.convolve(inp_arr, [1, 1], mode='same')
    elif ker_kind == 'diff1':
        # Backward diff, need to add first element in the beginning
        expected = np.concatenate([[inp_arr[0]], np.diff(inp_arr, n=1)])
    elif ker_kind == 'diff2':
        # Second diff, need to add the truncated convolution at the beginning
        # and the end
        expected = np.concatenate([[-2 * inp_arr[0] + inp_arr[1]],
                                   np.diff(inp_arr, n=2),
                                   [inp_arr[-2] - 2 * inp_arr[-1]]])
    else:
        assert False

    # Make sure we don't compare with too high precision
    expected = expected.astype(conv.range.dtype, copy=False)

    conv_res = conv(inp)
    assert all_almost_equal(conv_res, expected.astype(conv_res.dtype))

    out = conv.range.element()
    conv(inp, out=out)
    assert all_almost_equal(out, expected.astype(conv_res.dtype))


def test_dconv_2d(shape_2d, kernel_1d, conv_type, conv_impl, floating_dtype):
    """Check discrete convolution in 2d."""
    ker_kind, ker_1d = kernel_1d

    if conv_type.startswith('stack') and conv_impl == 'real':
        pytest.skip('stacked kernels not supported in real-space convolution')

    if floating_dtype == 'float16' and conv_impl == 'real':
        pytest.xfail('bug in scipy.signal.convolve for half float')

    if conv_type == 'full':
        axis = None
    else:
        axis = int(conv_type[-1])

    rn = odl.tensor_space(shape_2d, dtype=floating_dtype)
    if conv_type == 'full':
        kernel = np.outer(ker_1d, ker_1d)
    elif conv_type == 'bcast0':
        kernel = ker_1d[:, None]
    elif conv_type == 'bcast1':
        kernel = ker_1d[None, :]
    elif conv_type == 'stack0':
        kernel = ker_1d[:, None] * np.arange(shape_2d[1])[None, :]
    elif conv_type == 'stack1':
        kernel = ker_1d[None, :] * np.arange(shape_2d[0])[:, None]
    else:
        assert False

    conv = DiscreteConvolution(rn, kernel, impl=conv_impl, axis=axis)

    inp_arr, inp = noise_elements(rn)

    # Reference impl of stacked convolutions
    def conv_stack_0(arr_2d, ker_1d):
        cols = [
            i * scipy.signal.convolve(arr_2d[:, i], ker_1d, mode='same')
            for i in range(arr_2d.shape[1])]
        return np.hstack([col[:, None] for col in cols])

    def conv_stack_1(arr_2d, ker_1d):
        rows = [
            i * scipy.signal.convolve(arr_2d[i], ker_1d, mode='same')
            for i in range(arr_2d.shape[0])]
        return np.vstack([row[None, :] for row in rows])

    if ker_kind == 'smooth':
        if conv_type == 'full':
            expected = scipy.signal.convolve(inp_arr, [[1, 1],
                                                       [1, 1]], mode='same')
        elif conv_type == 'bcast0':
            expected = scipy.signal.convolve(inp_arr, [[1],
                                                       [1]], mode='same')
        elif conv_type == 'bcast1':
            expected = scipy.signal.convolve(inp_arr, [[1, 1]], mode='same')
        elif conv_type == 'stack0':
            expected = conv_stack_0(inp_arr, [1, 1])
        elif conv_type == 'stack1':
            expected = conv_stack_1(inp_arr, [1, 1])

    elif ker_kind == 'diff1':
        if conv_type == 'full':
            padded = np.pad(inp_arr, (1, 0), mode='constant')
            expected = np.diff(np.diff(padded, n=1, axis=0), n=1, axis=1)
        elif conv_type == 'bcast0':
            padded = np.pad(inp_arr, [(1, 0), (0, 0)], mode='constant')
            expected = np.diff(padded, n=1, axis=0)
        elif conv_type == 'bcast1':
            padded = np.pad(inp_arr, [(0, 0), (1, 0)], mode='constant')
            expected = np.diff(padded, n=1, axis=1)
        elif conv_type == 'stack0':
            padded = np.pad(inp_arr, [(1, 0), (0, 0)], mode='constant')
            expected = (np.diff(padded, n=1, axis=0) *
                        np.arange(inp_arr.shape[1])[None, :])
        elif conv_type == 'stack1':
            padded = np.pad(inp_arr, [(0, 0), (1, 0)], mode='constant')
            expected = (np.diff(padded, n=1, axis=1) *
                        np.arange(inp_arr.shape[0])[:, None])

    elif ker_kind == 'diff2':
        if conv_type == 'full':
            padded = np.pad(inp_arr, (1, 1), mode='constant')
            expected = np.diff(np.diff(padded, n=2, axis=0), n=2, axis=1)
        elif conv_type == 'bcast0':
            padded = np.pad(inp_arr, [(1, 1), (0, 0)], mode='constant')
            expected = np.diff(padded, n=2, axis=0)
        elif conv_type == 'bcast1':
            padded = np.pad(inp_arr, [(0, 0), (1, 1)], mode='constant')
            expected = np.diff(padded, n=2, axis=1)
        elif conv_type == 'stack0':
            padded = np.pad(inp_arr, [(1, 1), (0, 0)], mode='constant')
            expected = (np.diff(padded, n=2, axis=0) *
                        np.arange(inp_arr.shape[1])[None, :])
        elif conv_type == 'stack1':
            padded = np.pad(inp_arr, [(0, 0), (1, 1)], mode='constant')
            expected = (np.diff(padded, n=2, axis=1) *
                        np.arange(inp_arr.shape[0])[:, None])

    else:
        assert False

    # Make sure we don't compare with too high precision
    expected = expected.astype(conv.range.dtype, copy=False)

    conv_res = conv(inp)
    assert all_almost_equal(conv_res, expected.astype(conv_res.dtype))

    out = conv.range.element()
    conv(inp, out=out)
    assert all_almost_equal(out, expected.astype(conv_res.dtype))


def test_dconv_kernel_caching():
    """Check if the kernel caching in DiscreteConvolution works."""
    rn = odl.rn((3, 4))
    conv = DiscreteConvolution(rn, [[1, 1],
                                    [1, 1]], cache_kernel_ft=True, impl='fft')

    conv(rn.one())
    assert conv._kernel_ft is not None


if __name__ == '__main__':
    odl.util.test_file(__file__)
