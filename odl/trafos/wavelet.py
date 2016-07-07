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

"""Discrete wavelet transformation on L2 spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range, str, super

import numpy as np
from itertools import product
try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False

from odl.discr import DiscreteLp
from odl.operator import Operator


__all__ = ('WaveletTransform', 'WaveletTransformInverse',
           'PYWAVELETS_AVAILABLE')


_SUPPORTED_IMPL = ('pywt',)


def matrix_mul_axis(matrix, array, axis):
    """Return the matrix-vector product along an axis in ``array``.

    This function computes the usual matrix-vector product of ``matrix``
    with ``array``, where the summation is performed along the
    given ``axis``.

    Parameters
    ----------
    matrix : array-like
        Array with shape ``(m, n)``, acting as the matrix in the
        product.
    array : array-like
        Array with ``array.ndim >= 1``, acting as right operand in
        the product. It must fulfill ``array.shape[axis] == n``.
    axis : int
        Axis along which the summation in the matrix-vector product is
        performed. It must fulfill ``-array.ndim - 1 <= axis < array.ndim``,
        where negative values are interpreted in the usual "backwards
        indexing" sense.

    Returns
    -------
    prod : `numpy.ndarray`
        Matrix-vector product along the given axis. Its shape is
        ``(..., m, ...)``, where the value ``m`` appears at the index
        ``axis`` and the other values are the same as in ``array.shape``.
    """
    matrix = np.asarray(matrix)
    array = np.asarray(array)
    axis, axis_in = int(axis), axis

    if array.ndim == 0:
        raise ValueError('`array` must have at least 1 dimension')

    if matrix.ndim != 2:
        raise ValueError('`matrix` must have 2 dimensions, got ndim == {}'
                         ''.format(matrix.ndim))

    if axis != axis_in:
        raise ValueError('`axis` {} is not integer'.format(axis_in))

    if not -array.ndim <= axis < array.ndim:
        raise ValueError('`axis` {} not in the valid range {} -> {}'
                         ''.format(axis, -array.ndim, array.ndim - 1))

    if array.shape[axis] != matrix.shape[1]:
        raise ValueError('`array.shape[axis]` == {} and `matrix.shape[1] '
                         '== {} do not match'.format(array.shape[axis],
                                                     matrix.shape[1]))

    out_arr = np.tensordot(matrix, array, axes=[[1], [axis]])
    try:
        # np.moveaxis was added in Numpy 1.11
        return np.moveaxis(out_arr, 0, axis)
    except AttributeError:
        # Fallback for older Numpy versions
        newshape = list(range(1, array.ndim))
        newshape.insert(axis, 0)
        return np.transpose(out_arr, newshape)


def corr_one_level(rec, coeffs, top_matrices, bottom_matrices, axis):
    for coeff, tmat, bmat in zip(
            coeffs, top_matrices, bottom_matrices):
        top_slc = [slice(None)] * rec.ndim
        top_slc[axis] = slice(None, tmat.T.shape[1])
        top_slc = tuple(top_slc)
        bot_slc = [slice(None)] * rec.ndim
        bot_slc[axis] = slice(-bmat.T.shape[1], None)
        bot_slc = tuple(bot_slc)

        top_corr = matrix_mul_axis(tmat.T, coeff[top_slc], axis)
        bot_corr = matrix_mul_axis(bmat.T, coeff[bot_slc], axis)

        top_slc_rec = list(top_slc)
        top_slc_rec[axis] = slice(None, tmat.T.shape[0])
        bot_slc_rec = list(top_slc)
        bot_slc_rec[axis] = slice(None, bmat.T.shape[0])
        rec[top_slc_rec] += top_corr
        rec[bot_slc_rec] += bot_corr


def adjoint_correction_matrices(shape, wbasis, mode):
    """Construct adjoint correction matrices.

    Parameters
    ----------
    shape : `tuple`
        The shape of the original image
    wbasis : ``pywt.Wavelet``
        Describes properties of a selected wavelet basis.
        See PyWavelet `documentation
        <http://www.pybytes.com/pywavelets/ref/wavelets.html>`_
    mode : `str`
        Signal extention modes as defined by ``pywt.MODES.modes``
        http://www.pybytes.com/pywavelets/ref/signal-extension-modes.html

        The extension modes corrected for are:
        'constant': constant padding -- border values are replicated

        'symmetric': symmetric padding -- signal extension by
        mirroring samples

        'periodic': periodic padding -- signal is trated as a periodic one
    """
    num = shape[0]
    len_filter = wbasis.dec_len
    dec_lo = wbasis.dec_lo
    dec_hi = wbasis.dec_hi

    # TOP
    if mode == 'constant':
        T_lowpass = []
        T_highpass = []
        start = 2
        # PyWavelet filters all are even lengths
        niter = int(len_filter / 2 - 1)
        for ii in range(niter):
            T_lowpass.append(sum(dec_lo[start::]))
            T_highpass.append(sum(dec_hi[start::]))
            start += 2
        T_lowpass = np.asarray(T_lowpass)[:, None]
        T_highpass = np.asarray(T_highpass)[:, None]

        if (num % 2 == 0):
            B_lowpass = []
            B_highpass = []
            stop = len_filter - 2
            for _ in range(niter):
                B_lowpass.append(sum(dec_lo[0:stop]))
                B_highpass.append(sum(dec_hi[0:stop]))
                stop -= 2
            B_lowpass.reverse()
            B_highpass.reverse()

        else:
            # Bottom for odd data
            B_lowpass = []
            B_highpass = []
            stop = len_filter - 2
            for _ in range(niter):
                B_lowpass.append(sum(dec_lo[0:stop]))
                B_highpass.append(sum(dec_hi[0:stop]))
                stop -= 2
            B_lowpass.append(dec_lo[0])
            B_highpass.append(dec_hi[0])
            B_lowpass.reverse()
            B_highpass.reverse()

        B_lowpass = np.asarray(B_lowpass)[:, None]
        B_highpass = np.asarray(B_highpass)[:, None]

    if mode == 'periodic':
        rows = int(len_filter / 2 - 1)
        cols = len_filter - 2
        T_lowpass = np.zeros((rows, cols))
        T_highpass = np.zeros((rows, cols))
        flip_filter_low = np.flipud(dec_lo)
        flip_filter_high = np.flipud(dec_hi)
        start_T = 0  # cols - (len_filter - 2)
        stop_filter = len_filter - 2
        for ii in range(rows):
            T_lowpass[ii, start_T::] = flip_filter_low[0:stop_filter]
            T_highpass[ii, start_T::] = flip_filter_high[0:stop_filter]
            start_T += 2
            stop_filter -= 2

        if (num % 2 == 0):
            B_lowpass = np.zeros((rows, cols))
            B_highpass = np.zeros((rows, cols))
            stop_B = 2
            start_f = len_filter - 2
            for ii in range(rows):
                B_lowpass[ii, 0:stop_B] = flip_filter_low[start_f::]
                B_highpass[ii, 0:stop_B] = flip_filter_high[start_f::]
                stop_B += 2
                start_f -= 2
        else:
            rows = int(len_filter / 2)
            cols = len_filter - 1
            B_lowpass = np.zeros((rows, cols))
            B_highpass = np.zeros((rows, cols))
            stop_B = 1
            start_f = len_filter - 1
            for ii in range(rows):
                B_lowpass[ii, 0:stop_B] = flip_filter_low[start_f::]
                B_highpass[ii, 0:stop_B] = flip_filter_high[start_f::]
                stop_B += 2
                start_f -= 2

    if mode == 'symmetric':
        rows = int(len_filter / 2 - 1)
        cols = len_filter - 2
        T_lowpass = np.zeros((rows, cols))
        T_highpass = np.zeros((rows, cols))
        stop_T = len_filter - 2
        start_filter = 2
        for ii in range(rows):
            T_lowpass[ii, 0:stop_T] = dec_lo[start_filter::]
            T_highpass[ii, 0:stop_T] = dec_hi[start_filter::]
            stop_T -= 2
            start_filter += 2

        if (num % 2 == 0):
            B_lowpass = np.zeros((rows, cols))
            B_highpass = np.zeros((rows, cols))
            start_B = cols - 2
            stop_filter = 2
            for ii in range(rows):
                B_lowpass[ii, start_B::] = dec_lo[0:stop_filter]
                B_highpass[ii, start_B::] = dec_hi[0:stop_filter]
                start_B -= 2
                stop_filter += 2
        else:
            rows = int(len_filter / 2)
            cols = len_filter - 1
            B_lowpass = np.zeros((rows, cols))
            B_highpass = np.zeros((rows, cols))
            start_B = cols - 1
            stop_filter = 1
            for ii in range(rows):
                B_lowpass[ii, start_B::] = dec_lo[0:stop_filter]
                B_highpass[ii, start_B::] = dec_hi[0:stop_filter]
                start_B -= 2
                stop_filter += 2

    return T_lowpass, T_highpass, B_lowpass, B_highpass


def correct_adjoint_of_wavelet_transform(shape, wbasis, mode, wave_coeffs,
                                         nscales):
    """
    Depending on the boundary condition, the `waverec` and `wavedec`
    of PyWavelets are not exact adjoints of one another.
    This fuction corrects the output of `waverec` such that it will
    fullfill the adjoint test

    Parameters
    ----------
    shape : `tuple`
        The shape of the original image
    wbasis : ``pywt.Wavelet``
        Describes properties of a selected wavelet basis.
        See PyWavelet `documentation
        <http://www.pybytes.com/pywavelets/ref/wavelets.html>`_
    mode : `str`
        Signal extention modes as defined by ``pywt.MODES.modes``
        http://www.pybytes.com/pywavelets/ref/signal-extension-modes.html

        The extension modes corrected for are:
        'constant': constant padding -- border values are replicated

        'symmetric': symmetric padding -- signal extension by
        mirroring samples

        'periodic': periodic padding -- signal is trated as a periodic one
    wave_coeffs : `array-like`
        wavelet coefficient vector computed from the original image

    nscales : `int`
        Number of scales in the coefficient list.
        The maximum number of usable scales can be determined
        by ``pywt.dwt_max_level``. For more information see
        the corresponding `documentation of PyWavelets
        <http://www.pybytes.com/pywavelets/ref/\
dwt-discrete-wavelet-transform.html#maximum-decomposition-level\
-dwt-max-level>`_ .

    Returns
    -------
    corrected_adjoint_coeffs : `array-like`
        the corrected adjoint
    """
    size_list = coeff_size_list(shape, wbasis, mode, nscales=nscales)
    ndim = len(shape)

    if ndim == 1:
        if mode == 'constant':
            if wbasis.name.startswith('bior'):
                adjoint_name = wbasis.name.replace('bior', 'rbio')
                wbasis_adjoint = pywt.Wavelet(adjoint_name)
                T_low, T_high, B_low, B_high = adjoint_correction_matrices(
                    shape, wbasis_adjoint, mode)
                wbasis = wbasis_adjoint
            else:
                T_low, T_high, B_low, B_high = adjoint_correction_matrices(
                    shape, wbasis, mode)

            wcoef_list = array_to_pywt_list(wave_coeffs, size_list)

            approx = wcoef_list[0]
            for ii in range(1, nscales + 1):
                detail = wcoef_list[ii]
                tmp_list = []
                tmp_list.append(approx)
                tmp_list.append(detail)
                shape = size_list[ii + 1]
                size_list_tmp = coeff_size_list(shape, wbasis, mode, nscales=1)
                wcoeffs_tmp = pywt_list_to_array(tmp_list, size_list_tmp)
                approx = pywt.waverec(tmp_list, wbasis, mode)
                if np.shape(approx) != shape:
                    approx = approx[0:-1]

                corr = np.zeros_like(approx)
                top_low_high = np.zeros_like(wcoeffs_tmp)
                top_low_high[0:len(T_low)] = T_low[:, 0]
                start_d = size_list[ii][0]
                top_low_high[start_d:start_d + len(T_high)] = T_high[:, 0]
                bot_low_high = np.zeros_like(wcoeffs_tmp)
                stop = size_list[ii][0]
                bot_low_high[stop - len(B_low):stop] = B_low[:, 0]
                bot_low_high[-len(B_high)::] = B_high[:, 0]
                corr[0] = np.dot(top_low_high, wcoeffs_tmp)
                corr[-1] = np.dot(bot_low_high, wcoeffs_tmp)

                approx += corr

            corrected_adjoint_coeffs = corr + approx
            return corrected_adjoint_coeffs

        if mode == 'periodic':
            if wbasis.name.startswith('bior'):
                adjoint_name = wbasis.name.replace('bior', 'rbio')
                wbasis_adjoint = pywt.Wavelet(adjoint_name)
                T_low, T_high, B_low, B_high = adjoint_correction_matrices(
                    shape, wbasis_adjoint, mode)
                wbasis = wbasis_adjoint
            else:
                T_low, T_high, B_low, B_high = adjoint_correction_matrices(
                    shape, wbasis, mode)

            rows_Ttrans, cols_Ttrans = np.shape(np.transpose(T_low))
            rows_Btrans, cols_Btrans = np.shape(np.transpose(B_low))

            wcoef_list = array_to_pywt_list(wave_coeffs, size_list)
            approx = wcoef_list[0]
            for ii in range(1, nscales + 1):
                detail = wcoef_list[ii]
                tmp_list = []
                tmp_list.append(approx)
                tmp_list.append(detail)
                shape = size_list[ii + 1]
                size_list_tmp = coeff_size_list(shape, wbasis, mode, nscales=1)
                wcoeffs_tmp = pywt_list_to_array(tmp_list, size_list_tmp)
                approx = pywt.waverec(tmp_list, wbasis, mode)
                if np.shape(approx) != shape:
                    approx = approx[0:-1]

                corr_mtx_part1 = np.zeros((rows_Btrans, np.size(wcoeffs_tmp)))
                start_low = size_list[ii][0] - cols_Btrans
                stop_low = size_list[ii][0]
                corr_mtx_part1[:, start_low:stop_low] = np.transpose(B_low)
                start_high = 2 * size_list[ii][0] - cols_Btrans
                corr_mtx_part1[:, start_high::] = np.transpose(B_high)
                corr_part1 = np.dot(corr_mtx_part1, wcoeffs_tmp)

                corr_mtx_part2 = np.zeros((rows_Ttrans, np.size(wcoeffs_tmp)))
                corr_mtx_part2[:, 0:cols_Ttrans] = np.transpose(T_low)
                start_high = size_list[ii][0]
                stop_high = start_high + cols_Ttrans
                corr_mtx_part2[:, start_high:stop_high] = np.transpose(T_high)
                corr_part2 = np.dot(corr_mtx_part2, wcoeffs_tmp)

                corr_vect = np.zeros_like(approx)
                corr_vect[0:len(corr_part1)] = corr_part1
                corr_vect[-len(corr_part2)::] = corr_part2

                approx += corr_vect
            return approx

        if mode == 'symmetric':
            if wbasis.name.startswith('bior'):
                adjoint_name = wbasis.name.replace('bior', 'rbio')
                wbasis_adjoint = pywt.Wavelet(adjoint_name)
                T_low, T_high, B_low, B_high = adjoint_correction_matrices(
                    shape, wbasis_adjoint, mode)
                wbasis = wbasis_adjoint
            else:
                T_low, T_high, B_low, B_high = adjoint_correction_matrices(
                    shape, wbasis, mode)
            rows_Ttrans, cols_Ttrans = np.shape(np.transpose(T_low))
            rows_Btrans, cols_Btrans = np.shape(np.transpose(B_low))

            wcoef_list = array_to_pywt_list(wave_coeffs, size_list)
            approx = wcoef_list[0]
            for ii in range(1, nscales + 1):
                detail = wcoef_list[ii]
                tmp_list = []
                tmp_list.append(approx)
                tmp_list.append(detail)
                shape = size_list[ii + 1]
                size_list_tmp = coeff_size_list(shape, wbasis, mode, nscales=1)
                wcoeffs_tmp = pywt_list_to_array(tmp_list, size_list_tmp)
                approx = pywt.waverec(tmp_list, wbasis, mode)
                if np.shape(approx) != shape:
                    approx = approx[0:-1]

                corr_mtx_part1 = np.zeros((rows_Ttrans, np.size(wcoeffs_tmp)))
                corr_mtx_part1[:, 0:cols_Ttrans] = np.transpose(T_low)
                start_high = size_list[ii][0]
                stop_high = start_high + cols_Ttrans
                corr_mtx_part1[:, start_high:stop_high] = np.transpose(T_high)
                corr_part1 = np.dot(corr_mtx_part1, wcoeffs_tmp)

                corr_mtx_part2 = np.zeros((rows_Btrans, np.size(wcoeffs_tmp)))
                start_Blow = size_list[ii][0] - cols_Btrans
                stop_Blow = size_list[ii][0]
                corr_mtx_part2[:, start_Blow:stop_Blow] = np.transpose(B_low)
                corr_mtx_part2[:, -cols_Btrans::] = np.transpose(B_high)
                corr_part2 = np.dot(corr_mtx_part2, wcoeffs_tmp)

                corr_vect = np.zeros_like(approx)
                corr_vect[0:len(corr_part1)] = corr_part1
                corr_vect[-len(corr_part2)::] = corr_part2
                approx += corr_vect
            return approx

    if ndim == 2:
        if mode == 'constant' or 'symmetric':
            if wbasis.name.startswith('bior'):
                adjoint_name = wbasis.name.replace('bior', 'rbio')
                wbasis_adjoint = pywt.Wavelet(adjoint_name)
                T_low, T_high, B_low, B_high = adjoint_correction_matrices(
                    shape, wbasis_adjoint, mode)
                wbasis = wbasis_adjoint
            else:
                T_low, T_high, B_low, B_high = adjoint_correction_matrices(
                    shape, wbasis, mode)

            wcoef_list = array_to_pywt_list(wave_coeffs, size_list)
            approx = wcoef_list[0]
            for ii in range(1, nscales + 1):
                (lh, hl, hh) = wcoef_list[ii]
                shape = size_list[ii + 1]
                l = pywt.idwt(approx, lh, wbasis, mode, axis=0)
                h = pywt.idwt(hl, hh, wbasis, mode, axis=0)
                # Correct l and h
                correct_len_0 = shape[0]
                if l.shape[0] != correct_len_0:
                    l = l[:-1, :]
                    h = h[:-1, :]

                coeffs = [(l, (approx, lh)), (h, (hl, hh))]
                for rec, (rec_l, rec_h) in coeffs:
                    corr_one_level(rec, (rec_l, rec_h), (T_low, T_high),
                                   (B_low, B_high), axis=0)

                approx = pywt.idwt(l, h, wbasis, mode, axis=1)

                correct_len_1 = shape[1]
                if approx.shape[1] != correct_len_1:
                    approx = approx[:, :-1]

                corr_one_level(approx, (l, h), (T_low, T_high),
                               (B_low, B_high), axis=1)
            return approx

        if mode == 'periodic':
            if wbasis.name.startswith('bior'):
                adjoint_name = wbasis.name.replace('bior', 'rbio')
                wbasis_adjoint = pywt.Wavelet(adjoint_name)
                T_low, T_high, B_low, B_high = adjoint_correction_matrices(
                    shape, wbasis_adjoint, mode)
                wbasis = wbasis_adjoint
            else:
                T_low, T_high, B_low, B_high = adjoint_correction_matrices(
                    shape, wbasis, mode)

            wcoef_list = array_to_pywt_list(wave_coeffs, size_list)
            approx = wcoef_list[0]
            for ii in range(1, nscales + 1):
                (lh, hl, hh) = wcoef_list[ii]
                shape = size_list[ii + 1]
                l = pywt.idwt(approx, lh, wbasis, mode, axis=0)
                h = pywt.idwt(hl, hh, wbasis, mode, axis=0)
                correct_len_0 = shape[0]
                if l.shape[0] != correct_len_0:
                    l = l[:-1, :]
                    h = h[:-1, :]

                coeffs = [(l, (approx, lh)), (h, (hl, hh))]
                for rec, (rec_l, rec_h) in coeffs:
                    corr_one_level(rec, (rec_l, rec_h), (B_low, B_high),
                                   (T_low, T_high), axis=0)

                approx = pywt.idwt(l, h, wbasis, mode, axis=1)

                correct_len_1 = shape[1]
                if approx.shape[1] != correct_len_1:
                    approx = approx[:, :-1]

                corr_one_level(approx, (l, h), (B_low, B_high),
                               (T_low, T_high), axis=1)
            return approx

    if ndim == 3:
        if mode == 'constant' or 'symmetric':
            if wbasis.name.startswith('bior'):
                adjoint_name = wbasis.name.replace('bior', 'rbio')
                wbasis_adjoint = pywt.Wavelet(adjoint_name)
                T_low, T_high, B_low, B_high = adjoint_correction_matrices(
                    shape, wbasis_adjoint, mode)
                wbasis = wbasis_adjoint
            else:
                T_low, T_high, B_low, B_high = adjoint_correction_matrices(
                    shape, wbasis, mode)

            wcoef_list = array_to_pywt_list(wave_coeffs, size_list)
            approx = wcoef_list[0]
            for ii in range(1, nscales + 1):
                details = wcoef_list[ii]
                llh = details['aad']
                lhl = details['ada']
                lhh = details['add']
                hll = details['daa']
                hlh = details['dad']
                hhl = details['dda']
                hhh = details['ddd']
                shape = size_list[ii + 1]
                ll = pywt.idwt(approx, llh, wbasis, mode, axis=2)
                lh = pywt.idwt(lhl, lhh, wbasis, mode, axis=2)
                hl = pywt.idwt(hll, hlh, wbasis, mode, axis=2)
                hh = pywt.idwt(hhl, hhh, wbasis, mode, axis=2)
                correct_len_0 = shape[2]
                if ll.shape[2] != correct_len_0:
                    ll = ll[:, :, :-1]
                    lh = lh[:, :, :-1]
                    hl = hl[:, :, :-1]
                    hh = hh[:, :, :-1]

                coeffs = [(ll, (approx, llh)), (lh, (lhl, lhh)),
                          (hl, (hll, hlh)), (hh, (hhl, hhh))]
                for rec, (rec_l, rec_h) in coeffs:
                    corr_one_level(rec, (rec_l, rec_h), (T_low, T_high),
                                   (B_low, B_high), axis=2)

                l = pywt.idwt(ll, lh, wbasis, mode, axis=0)
                h = pywt.idwt(hl, hh, wbasis, mode, axis=0)
                correct_len_0 = shape[0]
                if l.shape[0] != correct_len_0:
                    l = l[:-1, :, :]
                    h = h[:-1, :, :]
                coeffs = [(l, (ll, lh)), (h, (hl, hh))]
                for rec, (rec_l, rec_h) in coeffs:
                    corr_one_level(rec, (rec_l, rec_h), (T_low, T_high),
                                   (B_low, B_high), axis=0)

                approx = pywt.idwt(l, h, wbasis, mode, axis=1)
                correct_len_1 = shape[1]
                if approx.shape[1] != correct_len_1:
                    approx = approx[:, :-1, :]

                corr_one_level(approx, (l, h), (T_low, T_high),
                               (B_low, B_high), axis=1)
            return approx

        if mode == 'periodic':
            if wbasis.name.startswith('bior'):
                adjoint_name = wbasis.name.replace('bior', 'rbio')
                wbasis_adjoint = pywt.Wavelet(adjoint_name)
                T_low, T_high, B_low, B_high = adjoint_correction_matrices(
                    shape, wbasis_adjoint, mode)
                wbasis = wbasis_adjoint
            else:
                T_low, T_high, B_low, B_high = adjoint_correction_matrices(
                    shape, wbasis, mode)

            wcoef_list = array_to_pywt_list(wave_coeffs, size_list)
            approx = wcoef_list[0]

            for ii in range(1, nscales + 1):
                details = wcoef_list[ii]
                llh = details['aad']
                lhl = details['ada']
                lhh = details['add']
                hll = details['daa']
                hlh = details['dad']
                hhl = details['dda']
                hhh = details['ddd']
                shape = size_list[ii + 1]
                ll = pywt.idwt(approx, llh, wbasis, mode, axis=2)
                lh = pywt.idwt(lhl, lhh, wbasis, mode, axis=2)
                hl = pywt.idwt(hll, hlh, wbasis, mode, axis=2)
                hh = pywt.idwt(hhl, hhh, wbasis, mode, axis=2)
                correct_len_0 = shape[2]
                if ll.shape[2] != correct_len_0:
                    ll = ll[:, :, :-1]
                    lh = lh[:, :, :-1]
                    hl = hl[:, :, :-1]
                    hh = hh[:, :, :-1]

                coeffs = [(ll, (approx, llh)), (lh, (lhl, lhh)),
                          (hl, (hll, hlh)), (hh, (hhl, hhh))]
                for rec, (rec_l, rec_h) in coeffs:
                    corr_one_level(rec, (rec_l, rec_h), (B_low, B_high),
                                   (T_low, T_high), axis=2)

                l = pywt.idwt(ll, lh, wbasis, mode, axis=0)
                h = pywt.idwt(hl, hh, wbasis, mode, axis=0)
                correct_len_0 = shape[0]
                if l.shape[0] != correct_len_0:
                    l = l[:-1, :, :]
                    h = h[:-1, :, :]
                coeffs = [(l, (ll, lh)), (h, (hl, hh))]
                for rec, (rec_l, rec_h) in coeffs:
                    corr_one_level(rec, (rec_l, rec_h), (B_low, B_high),
                                   (T_low, T_high), axis=0)

                approx = pywt.idwt(l, h, wbasis, mode, axis=1)
                correct_len_1 = shape[1]
                if approx.shape[1] != correct_len_1:
                    approx = approx[:, :-1, :]

                corr_one_level(approx, (l, h), (B_low, B_high),
                               (T_low, T_high), axis=1)
            return approx


def coeff_size_list(shape, wbasis, mode, nscales=None, axes=None):
    """Construct a size list from given wavelet coefficients.

    Related to 1D, 2D and 3D multidimensional wavelet transforms that utilize
    `PyWavelets
    <http://www.pybytes.com/pywavelets/>`_.

    Parameters
    ----------
    shape : `tuple`
        Number of pixels/voxels in the image. Its length must be 1, 2 or 3.

    wbasis : ``pywt.Wavelet``
        Selected wavelet basis. For more information see the
        `PyWavelets documentation on wavelet bases
        <http://www.pybytes.com/pywavelets/ref/wavelets.html>`_.

    mode : `str`
        Signal extention mode. Possible extension modes are

        'zero': zero-padding -- signal is extended by adding zero samples

        'constant': constant padding -- border values are replicated

        'symmetric': symmetric padding -- signal extension by mirroring samples

        'periodic': periodic padding -- signal is trated as a periodic one

        'smooth': smooth padding -- signal is extended according to the
        first derivatives calculated on the edges (straight line)

        'periodization': periodization -- like periodic-padding but gives the
        smallest possible number of decomposition coefficients.

    nscales : `int`, optional
        Number of scales in the multidimensional wavelet
        transform.  This parameter is checked against the maximum number of
        scales returned by ``pywt.dwt_max_level``. For more information
        see the `PyWavelets documentation on the maximum level of scales
        <http://www.pybytes.com/pywavelets/ref/\
dwt-discrete-wavelet-transform.html#maximum-decomposition-level\
-dwt-max-level>`_.
        If `nscales=None` `axes` has to be given

    axes : sequence of `int`, optional
         Dimensions in which to calculate the wavelet transform.
         If `axes=None` `nscales` has to be given

    Returns
    -------
    size_list : list
        A list containing the sizes of the wavelet (approximation
        and detail) coefficients at different scaling levels:

        ``size_list[0]`` = size of approximation coefficients at
        the coarsest level

        ``size_list[1]`` = size of the detail coefficients at the
        coarsest level

        ...

        ``size_list[N]`` = size of the detail coefficients at the
        finest level

        ``size_list[N+1]`` = size of the original image

        ``N`` = number of scaling levels = nscales
    """
    if len(shape) not in (1, 2, 3):
        raise ValueError('Shape must have length 1, 2 or 3, got {}.'
                         ''.format(len(shape)))

    if nscales is None and axes is None:
        raise ValueError('Either nscales or axes has to be defined')

    if axes is None:
        max_level = pywt.dwt_max_level(shape[0], filter_len=wbasis.dec_len)
        if nscales > max_level:
            raise ValueError('Too many scaling levels, got {}, maximum useful'
                             ' level is {}'
                             ''.format(nscales, max_level))

        # dwt_coeff_len calculates the number of coefficients at the next
        # scaling level given the input size, the length of the filter and
        # the applied mode.
        # We use this in the following way (per dimension):
        # - length[0] = original data length
        # - length[n+1] = dwt_coeff_len(length[n], ...)
        # - until n = nscales
        size_list = [shape]
        for scale in range(nscales):
            shp = tuple(pywt.dwt_coeff_len(n, filter_len=wbasis.dec_len,
                                           mode=mode)
                        for n in size_list[scale])
            size_list.append(shp)

        # Add a duplicate of the last entry for the approximation coefficients
        size_list.append(size_list[-1])
        # We created the list in reversed order compared to what pywt expects
        size_list.reverse()

    if nscales is None:
        size_list = [shape]
        ndim = len(shape)
        axes_counts = [axes.count(i) for i in range(ndim)]
        reduced_shape = []
        for ax_len, ax_count in zip(shape, axes_counts):
            n = ax_len
            for _ in range(ax_count):
                n = pywt.dwt_coeff_len(n, filter_len=wbasis.dec_len,
                                       mode=mode)
            reduced_shape.append(n)

        size_list.append(tuple(reduced_shape))
        size_list.append(size_list[-1])
        size_list.reverse()

    return size_list


def pywt_dict_to_array(coeffs, size_list, axes):
    """Convert a PyWavelet coefficient dictionary into a flat array.

    Related to 2D and 3D discrete wavelet transforms with `axes` option.
    Computing 1D wavelet transform multiple times along the axis
    corresponds to computing 1D multilevel wavelet transform.

    Parameters
    ----------
    coeff : ordered `dict`
        Coefficients are organized in the dictionary with the following
        appreviations in the key words:

        ``a`` = approximation,

        ``d`` = detail

    size_list : `list`
        A list containing the sizes of the wavelet (approximation
        and detail) coefficients when `axes` option is used.

    axes :
   Returns
    -------
    arr : `numpy.ndarray`
        Flattened and concatenated coefficient array
        The length of the array depends on the size of input image to
        be transformed, on the chosen wavelet basis, on the used boundary
        condition and on the defined axes.
    """
    keys = list(coeffs.keys())
    keys.sort()
    flat_sizes = [np.prod(shp) for shp in size_list[:-1]]
    num_dcoeffs = len(keys) - 1

    flat_total_size = flat_sizes[0] + num_dcoeffs * sum(flat_sizes[1:])
    flat_coeff = np.empty(flat_total_size)

    start = 0
    stop = flat_sizes[0]

    details = tuple(coeffs[key] for key in keys if 'd' in key)
    coeff_list = []
    coeff_list.append(details)
    coeff_list.append(coeffs[keys[0]])
    coeff_list.reverse()
    flat_coeff[start:stop] = coeffs[keys[0]].ravel()
    for fsize, detail_coeffs in zip(flat_sizes[1:], coeff_list[1:]):
        for dcoeff in detail_coeffs:
            start, stop = stop, stop + fsize
            flat_coeff[start:stop] = dcoeff.ravel()

    return flat_coeff


def pywt_list_to_array(coeff, size_list):
    """Convert a Pywavelets coefficient list into a flat array.

    Related to 1D, 2D and 3D multilevel discrete wavelet transforms.

    Parameters
    ----------
    coeff : ordered list
        Coefficient are organized in the list in the following way:

        In 1D:

        ``[aN, (dN), ..., (d1)]``

        The abbreviations refer to

        ``a`` = approximation,

        ``d`` = detail

        In 2D:

        ``[aaN, (adN, daN, ddN), ..., (ad1, da1, dd1)]``

        The abbreviations refer to

        ``aa`` = approx. on 1st dim, approx. on 2nd dim (approximation),

        ``ad`` = approx. on 1st dim, detail on 2nd dim (horizontal),

        ``da`` = detail on 1st dim, approx. on 2nd dim (vertical),

        ``dd`` = detail on 1st dim, detail on 2nd dim (diagonal),

        In 3D:

        ``[aaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...
        (aad1, ada1, add1, daa1, dad1, dda1, ddd1)]``

        The abbreviations refer to

        ``aaa`` = approx. on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        ``aad`` = approx. on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ``ada`` = approx. on 1st dim, detail on 3nd dim, approx. on 3rd dim,

        ``add`` = approx. on 1st dim, detail on 3nd dim, detail on 3rd dim,

        ``daa`` = detail on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        ``dad`` = detail on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ``dda`` = detail on 1st dim, detail on 2nd dim, approx. on 3rd dim,

        ``ddd`` = detail on 1st dim, detail on 2nd dim, detail on 3rd dim,

        ``N`` refers to the number of scaling levels

    size_list : list
        A list containing the sizes of the wavelet (approximation
        and detail) coefficients at different scaling levels.

        ``size_list[0]`` = size of approximation coefficients at
        the coarsest level,

        ``size_list[1]`` = size of the detailed coefficients at
        the coarsest level,

        ``size_list[N]`` = size of the detailed coefficients at
        the finest level,

        ``size_list[N+1]`` = size of original image,

        ``N`` =  the number of scaling levels

    Returns
    -------
    arr : `numpy.ndarray`
        Flattened and concatenated coefficient array
        The length of the array depends on the size of input image to
        be transformed and on the chosen wavelet basis.
      """
    flat_sizes = [np.prod(shp) for shp in size_list[:-1]]
    ndim = len(size_list[0])
    dcoeffs_per_scale = 2 ** ndim - 1

    flat_total_size = flat_sizes[0] + dcoeffs_per_scale * sum(flat_sizes[1:])
    flat_coeff = np.empty(flat_total_size)

    start = 0
    stop = flat_sizes[0]
    flat_coeff[start:stop] = coeff[0].ravel()

    if dcoeffs_per_scale == 1:
        for fsize, detail_coeffs in zip(flat_sizes[1:], coeff[1:]):
            start, stop = stop, stop + fsize
            flat_coeff[start:stop] = detail_coeffs.ravel()
    elif dcoeffs_per_scale == 3:
        for fsize, detail_coeffs in zip(flat_sizes[1:], coeff[1:]):
            for dcoeff in detail_coeffs:
                start, stop = stop, stop + fsize
                flat_coeff[start:stop] = dcoeff.ravel()
    elif dcoeffs_per_scale == 7:
        for ind in range(1, len(size_list) - 1):
            detail_coeffs_dict = coeff[ind]
            keys = list(detail_coeffs_dict.keys())
            keys.sort()
            details = tuple(detail_coeffs_dict[key] for key in
                            keys if 'd' in key)
            fsize = flat_sizes[ind]
            for dcoeff in details:
                start, stop = stop, stop + fsize
                flat_coeff[start:stop] = dcoeff.ravel()

    return flat_coeff


def array_to_pywt_dict(coeff, size_list, axes):
    """Convert a flat array into a PyWavelet coefficient dictionary.

    For 2D and 3D discrete wavelet transform with `axes` option.
    Computing 1D wavelet transform multiple times along the axis
    correspond to computing 1D multilevel wavelet transform

    Parameters
    ----------
    coeff : `DiscreteLpVector`
        A flat coefficient vector containing the approximation,
        and detail coefficients

    size_list : list
       A list of wavelet coefficient sizes.

    Returns
    -------
    coeff_dict : an ordered `dict` . In the key words following
        appreviations are used

        ``a`` = approximation,

        ``d`` = detail
    """
    rep = len(axes)
    keys = list(''.join(k) for k in product('ad', repeat=rep))
    num_coeffs = len(keys)
    shape = size_list[1]
    values = tuple(c.reshape(shape) for c in
                   np.split(np.asarray(coeff), num_coeffs))
    coeff_dict = {key: value for key, value in zip(keys, values)}

    return coeff_dict


def array_to_pywt_list(coeff, size_list):
    """Convert a flat array into a `pywt
    <http://www.pybytes.com/pywavelets/>`_ coefficient list.

    For multilevel 1D, 2D and 3D discrete wavelet transforms.

    Parameters
    ----------
    coeff : `DiscreteLpElement`
        A flat coefficient vector containing the approximation,
        and detail coefficients in the following order
        [aaaN, aadN, adaN, addN, daaN, dadN, ddaN, dddN, ...
        aad1, ada1, add1, daa1, dad1, dda1, ddd1]

    size_list : list
       A list of coefficient sizes such that,

       ``size_list[0]`` = size of approximation coefficients at the coarsest
                          level,

       ``size_list[1]`` = size of the detailedetails at the coarsest level,

       ``size_list[N]`` = size of the detailed coefficients at the finest
                          level,

       ``size_list[N+1]`` = size of original image,

       ``N`` =  the number of scaling levels

    Returns
    -------
    coeff : ordered list
        Coefficient are organized in the list in the following way:

        In 1D:

        ``[aN, (dN), ... (d1)]``

        The abbreviations refer to

        ``a`` = approximation,

        ``d`` = detail,

        In 2D:

        ``[aaN, (adN, daN, ddN), ... (ad1, da1, dd1)]``

        The abbreviations refer to

        ``aa`` = approx. on 1st dim, approx. on 2nd dim (approximation),

        ``ad`` = approx. on 1st dim, detail on 2nd dim (horizontal),

        ``da`` = detail on 1st dim, approx. on 2nd dim (vertical),

        ``dd`` = detail on 1st dim, detail on 2nd dim (diagonal),

        In 3D:

        ``[aaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...
        (aad1, ada1, add1, daa1, dad1, dda1, ddd1)]``

        The abbreviations refer to

        ``aaa`` = approx. on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        ``aad`` = approx. on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ``ada`` = approx. on 1st dim, detail on 3nd dim, approx. on 3rd dim,

        ``add`` = approx. on 1st dim, detail on 3nd dim, detail on 3rd dim,

        ``daa`` = detail on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        ``dad`` = detail on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ``dda`` = detail on 1st dim, detail on 2nd dim, approx. on 3rd dim,

        ``ddd`` = detail on 1st dim, detail on 2nd dim, detail on 3rd dim,

        ``N`` refers to the number of scaling levels

    """
    flat_sizes = [np.prod(shp) for shp in size_list[:-1]]
    start = 0
    stop = flat_sizes[0]
    coeff_list = [np.asarray(coeff)[start:stop].reshape(size_list[0])]
    ndim = len(size_list[0])
    dcoeffs_per_scale = 2 ** ndim - 1

    if dcoeffs_per_scale == 1:
        for fsize, shape in zip(flat_sizes[1:], size_list[1:]):
            start, stop = stop, stop + dcoeffs_per_scale * fsize
            detail_coeffs = np.asarray(coeff)[start:stop]
            coeff_list.append(detail_coeffs)
    elif ndim == 2:
        for fsize, shape in zip(flat_sizes[1:], size_list[1:]):
            start, stop = stop, stop + dcoeffs_per_scale * fsize
            detail_coeffs = tuple(c.reshape(shape) for c in
                                  np.split(np.asarray(coeff)[start:stop],
                                           dcoeffs_per_scale))
            coeff_list.append(detail_coeffs)
    elif ndim == 3:
        for ind in range(1, len(size_list) - 1):
            fsize = flat_sizes[ind]
            shape = size_list[ind]
            start, stop = stop, stop + dcoeffs_per_scale * fsize
            detail_coeffs = tuple(c.reshape(shape) for c in
                                  np.split(np.asarray(coeff)[start:stop],
                                           dcoeffs_per_scale))
            (aad, ada, add, daa, dad, dda, ddd) = detail_coeffs
            coeff_dict = {'aad': aad, 'ada': ada, 'add': add,
                          'daa': daa, 'dad': dad, 'dda': dda, 'ddd': ddd}
            coeff_list.append(coeff_dict)

    return coeff_list


class WaveletTransform(Operator):

    """Discrete wavelet transform between discrete Lp spaces."""

    def __init__(self, domain, wbasis, pad_mode, nscales=None, axes=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscreteLp`
            Domain of the wavelet transform (the "image domain").
            The exponent :math:`p` of the discrete :math:`L^p`
            space must be equal to 2.0.

        wbasis :  {`str`, ``pywt.Wavelet``}
            If a string is given, converts to a ``pywt.Wavelet``.
            Describes properties of a selected wavelet basis.
            See PyWavelet `documentation
            <http://www.pybytes.com/pywavelets/ref/wavelets.html>`_

            Possible wavelet families are:

            Haar (``haar``)

            Daubechies (``db``)

            Symlets (``sym``)

            Coiflets (``coif``)

            Biorthogonal (``bior``)

            Reverse biorthogonal (``rbio``)

            Discrete FIR approximation of Meyer wavelet (``dmey``)

        pad_mode : string
             Signal extention modes as defined by ``pywt.MODES.modes``
             http://www.pybytes.com/pywavelets/ref/signal-extension-modes.html

             Possible extension modes are:

            'zero': zero-padding -- signal is extended by adding zero samples

            'constant': constant padding -- border values are replicated

            'symmetric': symmetric padding -- signal extension by
            mirroring samples

            'periodic': periodic padding -- signal is trated as a periodic one

            'smooth': smooth padding -- signal is extended according to the
            first derivatives calculated on the edges (straight line)

            'periodization': periodization -- like periodic-padding but gives
            the smallest possible number of decomposition coefficients.

        nscales : `int`, optional
            Number of scales in the coefficient list.
            The maximum number of usable scales can be determined
            by ``pywt.dwt_max_level``. For more information see
            the corresponding `documentation of PyWavelets
            <http://www.pybytes.com/pywavelets/ref/\
dwt-discrete-wavelet-transform.html#maximum-decomposition-level\
-dwt-max-level>`_ .
            `nscales` option cannot be combined with `axes` option:
            if `nscales=None` `axes` has to be given.

        axes : sequence of `int`, optional
            Dimensions in which to calculate the wavelet transform.
            The sequence's length has to be equal to dimension of the ``grid``
            `None` means traditional transform along the axes in ``grid``.
            `axes` option cannot be combined with `nscales` option:
            if `axes=None` `nscales` has to be given.

        Examples
        --------
        >>> import odl, pywt
        >>> wbasis = pywt.Wavelet('db1')
        >>> discr_domain = odl.uniform_discr([0, 0], [1, 1], (16, 16))
        >>> op = WaveletTransform(discr_domain, nscales=1,
        ...                       wbasis=wbasis, pad_mode='per')
        >>> op.is_biorthogonal
        True
        """
        self.pad_mode = str(pad_mode).lower()

        if isinstance(wbasis, pywt.Wavelet):
            self.wbasis = wbasis
        else:
            self.wbasis = pywt.Wavelet(wbasis)

        if not isinstance(domain, DiscreteLp):
            raise TypeError('`domain` {!r} is not a `DiscreteLp` instance.'
                            ''.format(domain))

        if domain.exponent != 2.0:
            raise ValueError('`domain` Lp exponent is {} instead of 2.0.'
                             ''.format(domain.exponent))

        if domain.ndim not in [1, 2, 3]:
            raise NotImplementedError('Dimension of the domain {} not 1, '
                                      '2 or 3'.format(len(domain.ndim)))

        if axes is not None:
            if nscales is not None:
                raise ValueError('Cannot use both nscales and axes options '
                                 'at the same time, set other to None')

            else:
                self.nscales = None
            if domain.ndim == 1:
                raise ValueError('Wavelet transform in 1D multiple times '
                                 'along the axis corresponds to 1D multilevel '
                                 'wavelet transform. Set axes to  None and '
                                 'nscales to {}.'.format(len(axes)))

            self.axes = tuple(int(ax) for ax in axes)
            max_level = pywt.dwt_max_level(domain.shape[0],
                                           filter_len=self.wbasis.dec_len)

            axes_counts = [axes.count(i) for i in range(domain.ndim)]
            for i in range(len(axes_counts)):
                if axes_counts[i] > max_level:
                    raise ValueError('Wavelet transforms per axes cannot be '
                                     'performed more than maximum useful '
                                     'level computed by pywt.dwt_max_level. '
                                     'Max level here is {}.'.format(max_level))

            self.size_list = coeff_size_list(domain.shape, self.wbasis,
                                             self.pad_mode, nscales=None,
                                             axes=self.axescoeffs)

            rep = len(axes)
            keys = list(''.join(k) for k in product('ad', repeat=rep))
            num_coeffs = len(keys)
            ran_size = num_coeffs * np.prod(self.size_list[0])

        elif nscales is not None:
            self.axes = None
            self.nscales = int(nscales)
            max_level = pywt.dwt_max_level(domain.shape[0],
                                           filter_len=self.wbasis.dec_len)
            if self.nscales > max_level:
                raise ValueError('Cannot use more than {} scaling levels, '
                                 'got {}. Maximum useful number of levels '
                                 'can be computed using pywt.dwt_max_level '
                                 ''.format(max_level, self.nscales))
            self.size_list = coeff_size_list(
                domain.shape, self.wbasis, self.pad_mode, self.nscales,
                axes=None)

            multiplicity = {1: 1, 2: 3, 3: 7}
            ran_size = (np.prod(self.size_list[0]) +
                        sum(multiplicity[domain.ndim] * np.prod(shape)
                            for shape in self.size_list[1:-1]))

        else:
            raise ValueError('Either `nscales` or `axes` has to be given')

        # TODO: Maybe allow other ranges like Besov spaces (yet to be created)
        ran = domain.dspace_type(ran_size, dtype=domain.dtype)
        super().__init__(domain, ran, linear=True)

    @property
    def is_orthogonal(self):
        """Whether or not the wavelet basis is orthogonal."""
        return self.wbasis.orthogonal

    @property
    def is_biorthogonal(self):
        """Whether or not the wavelet basis is bi-orthogonal."""
        return self.wbasis.biorthogonal

    def _call(self, x):
        """Compute the discrete wavelet transform.

        Parameters
        ----------
        x : `domain` element

        Returns
        -------
        arr : `numpy.ndarray`
            Flattened and concatenated coefficient array
            The length of the array depends on the size of input image to
            be transformed and on the chosen wavelet basis.
        """
        if self.axes is None:
            if self.domain.ndim == 1:
                coeff_list = pywt.wavedec(x, self.wbasis, self.pad_mode,
                                          self.nscales)
                coeff_arr = pywt_list_to_array(coeff_list, self.size_list)
                return self.range.element(coeff_arr)

            if self.domain.ndim == 2:
                coeff_list = pywt.wavedec2(x, self.wbasis, self.pad_mode,
                                           self.nscales)
                coeff_arr = pywt_list_to_array(coeff_list, self.size_list)
                return self.range.element(coeff_arr)

            if self.domain.ndim == 3:
                coeff_list = pywt.wavedecn(x, self.wbasis, self.pad_mode,
                                           self.nscales)
                coeff_arr = pywt_list_to_array(coeff_list, self.size_list)

                return self.range.element(coeff_arr)
        else:
            coeff_dict = pywt.dwtn(x, self.wbasis, self.pad_mode, self.axes)
            coeff_arr = pywt_dict_to_array(coeff_dict, self.size_list,
                                           self.axes)
            return self.range.element(coeff_arr)

    @property
    def adjoint(self):
        """Adjoint wavelet transform.

        Returns
        -------
        adjoint : `WaveletTransformInverse`
            If the transform is orthogonal, the adjoint is the inverse.

        Raises
        ------
        OpNotImplementedError
            If `is_orthogonal` is not true, the adjoint is not implemented.
        """
        if self.is_orthogonal:
            output = self.inverse
            output /= self.domain.cell_volume
            return output
            # return self.inverse
        elif self.wbasis.name.startswith('bior'):
            adjoint_name = self.wbasis.name.replace('bior', 'rbio')
            wbasis_adjoint = pywt.Wavelet(adjoint_name)
            output = WaveletTransformInverse(
                ran=self.domain, wbasis=wbasis_adjoint, mode=self.pad_mode,
                nscales=self.nscales, axes=self.axes)
            output /= self.domain.cell_volume
            return output
        else:
            return super().adjoint

    @property
    def inverse(self):
        """Inverse wavelet transform.

        Returns
        -------
        inverse : `WaveletTransformInverse`

        See Also
        --------
        adjoint
        """
        return WaveletTransformInverse(
            range=self.domain, wbasis=self.wbasis, pad_mode=self.pad_mode,
            nscales=self.nscales, axes=self.axes)


class WaveletTransformInverse(Operator):

    """Discrete inverse wavelet tranform between discrete Lp spaces."""

    def __init__(self, range, wbasis, mode, nscales=None, axes=None):
        """Initialize a new instance.

         Parameters
        ----------
        range : `DiscreteLp`
            Domain of the wavelet transform (the "image domain").
            The exponent :math:`p` of the discrete :math:`L^p`
            space must be equal to 2.0.

        wbasis :  ``pywt.Wavelet``
            Describes properties of a selected wavelet basis.
            See PyWavelet `documentation
            <http://www.pybytes.com/pywavelets/ref/wavelets.html>`_

            Possible wavelet families are:

            Haar (``haar``)

            Daubechies (``db``)

            Symlets (``sym``)

            Coiflets (``coif``)

            Biorthogonal (``bior``)

            Reverse biorthogonal (``rbio``)

            Discrete FIR approximation of Meyer wavelet (``dmey``)

        pad_mode : string
             Signal extention modes as defined by ``pywt.MODES.modes``
             http://www.pybytes.com/pywavelets/ref/signal-extension-modes.html

             Possible extension modes are:

            'zero': zero-padding -- signal is extended by adding zero samples

            'constant': constant padding -- border values are replicated

            'symmetric': symmetric padding -- signal extension by
            mirroring samples

            'periodic': periodic padding -- signal is trated as a periodic one

            'smooth': smooth padding -- signal is extended according to the
            first derivatives calculated on the edges (straight line)

            'periodization': periodization -- like periodic-padding but gives
            the smallest possible number of decomposition coefficients.

        nscales : `int`, optional
            Number of scales in the coefficient list.
            The maximum number of usable scales can be determined
            by ``pywt.dwt_max_level``. For more information see
            the corresponding `documentation of PyWavelets
            <http://www.pybytes.com/pywavelets/ref/\
dwt-discrete-wavelet-transform.html#maximum-decomposition-level\
-dwt-max-level>`_ .
            `nscales` option cannot be combined with `axes` option:
            if `nscales=None` `axes` has to be given.

        axes : sequence of `int`, optional
            Dimensions in which to calculate the wavelet transform.
            The sequence's length has to be equal to dimension of the ``grid``
            `None` means traditional transform along the axes in ``grid``.
            `axes` option cannot be combined with `nscales` option:
            if `axes=None` `nscales` has to be given.
        """
        from builtins import range as range_seq

        if nscales is None and axes is None:
            raise ValueError('Either nscales or axes has to be defined')
        elif nscales is not None and axes is not None:
            raise ValueError('Cannot use both nscales and axes options '
                             ' at the same time, set other to None')

        self.pad_mode = str(mode).lower()

        if isinstance(wbasis, pywt.Wavelet):
            self.wbasis = wbasis
        else:
            self.wbasis = pywt.Wavelet(wbasis)

        if not isinstance(range, DiscreteLp):
            raise TypeError('domain {!r} is not a `DiscreteLp` instance.'
                            ''.format(range))

        if range.exponent != 2.0:
            raise ValueError('domain Lp exponent is {} instead of 2.0.'
                             ''.format(range.exponent))

        if range.ndim not in [1, 2, 3]:
            raise NotImplementedError('Dimension of the domain {} not 1, '
                                      '2 or 3'.format(len(range.ndim)))

        if axes is not None:
            if nscales is not None:
                raise ValueError('Cannot use both nscales and axes options '
                                 'at the same time, set other to None')
            self.nscales = None
            if range.ndim == 1:
                raise ValueError('Wavelet transform in 1D multiple times '
                                 'along the axis corresponds to 1D multilevel '
                                 'wavelet transform. Set axes to None and '
                                 'nscales to {}.'.format(len(axes)))

            self.axes = tuple(int(ax) for ax in axes)
            max_level = pywt.dwt_max_level(range.shape[0],
                                           filter_len=self.wbasis.dec_len)

            axes_counts = [axes.count(i) for i in range_seq(range.ndim)]
            for i in range_seq(len(axes_counts)):
                if axes_counts[i] > max_level:
                    raise ValueError('Wavelet transforms per axes cannot be '
                                     'performed more than maximum useful '
                                     'level computed by pywt.dwt_max_level. '
                                     'Max level here is {}.'.format(max_level))

            self.size_list = coeff_size_list(range.shape, self.wbasis,
                                             self.pad_mode, nscales=None,
                                             axes=self.axes)

            rep = len(axes)
            keys = list(''.join(k) for k in product('ad', repeat=rep))
            num_coeffs = len(keys)
            dom_size = num_coeffs * np.prod(self.size_list[0])

        elif nscales is not None:
            self.axes = None
            self.nscales = int(nscales)
            max_level = pywt.dwt_max_level(range.shape[0],
                                           filter_len=self.wbasis.dec_len)
            if self.nscales > max_level:
                raise ValueError('Cannot use more than {} scaling levels, '
                                 'got {}. Maximum useful number of levels '
                                 'can be computed using pywt.dwt_max_level '
                                 ''.format(max_level, self.nscales))
            self.size_list = coeff_size_list(
                range.shape, self.wbasis, self.pad_mode, self.nscales,
                axes=None)

            multiplicity = {1: 1, 2: 3, 3: 7}
            dom_size = (np.prod(self.size_list[0]) +
                        sum(multiplicity[range.ndim] * np.prod(shape)
                            for shape in self.size_list[1:-1]))

        else:
            raise ValueError('Either `nscales` or `axes` has to be given')

        # TODO: Maybe allow other ranges like Besov spaces (yet to be created)
        domain = range.dspace_type(dom_size, dtype=range.dtype)
        super().__init__(domain, range, linear=True)

    @property
    def is_orthogonal(self):
        """Whether or not the wavelet basis is orthogonal."""
        return self.wbasis.orthogonal

    @property
    def is_biorthogonal(self):
        """Whether or not the wavelet basis is bi-orthogonal."""
        return self.wbasis.biorthogonal

    def _call(self, coeff):
        """Compute the discrete 1D, 2D or 3D inverse wavelet transform."""
        if self.axes is None:
            if len(self.range.shape) == 1:
                coeff_list = array_to_pywt_list(coeff, self.size_list)
                x = pywt.waverec(coeff_list, self.wbasis, self.pad_mode)
                return self.range.element(x)
            elif len(self.range.shape) == 2:
                coeff_list = array_to_pywt_list(coeff, self.size_list)
                x = pywt.waverec2(coeff_list, self.wbasis, self.pad_mode)
                return self.range.element(x)
            elif len(self.range.shape) == 3:
                coeff_list = array_to_pywt_list(coeff, self.size_list)
                x = pywt.waverecn(coeff_list, self.wbasis, self.pad_mode)
                return x

        else:
            coeff_dict = array_to_pywt_dict(coeff, self.size_list, self.axes)
            x = pywt.idwtn(coeff_dict, self.wbasis, self.pad_mode, self.axes)
            return x

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `WaveletTransform`
            If the transform is orthogonal, the adjoint is the inverse.

        Raises
        ------
        OpNotImplementedError
            If `is_orthogonal` is not true, the adjoint is not implemented.

        See Also
        --------
        inverse
        """
        if self.is_orthogonal:
            return self.inverse
        elif self.wbasis.name.startswith('bior'):
            adjoint_name = self.wbasis.name.replace('bior', 'rbio')
            wbasis_adjoint = pywt.Wavelet(adjoint_name)
            return WaveletTransform(dom=self.range, wbasis=wbasis_adjoint,
                                    mode=self.pad_mode, nscales=self.nscales,
                                    axes=self.axes)
        else:
            return super().adjoint

    @property
    def inverse(self):
        """The inverse wavelet transform."""
        return WaveletTransform(domain=self.range, wbasis=self.wbasis,
                                pad_mode=self.pad_mode, nscales=self.nscales,
                                axes=self.axes)

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
