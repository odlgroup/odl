# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division

import numpy as np
import pytest

import odl
from odl.util.normalize import (
    normalized_axes_tuple, normalized_scalar_param_list)
from odl.util.testutils import simple_fixture

# --- pytest fixtures --- #


length = simple_fixture('length', [1, 2])


single_conv_params = [(-1.0, float),
                      (2, float),
                      ('10', float),
                      (np.array(2.3, dtype=float), float),
                      (np.array(2, dtype=int), float),
                      (-1, int),
                      ('10', int),
                      (np.array(2, dtype=int), int),
                      (1, bool),
                      (False, bool)]

single_conv_ids = [' input = {0[0]}, conv = {0[1]} '.format(p)
                   for p in single_conv_params]


@pytest.fixture(scope="module", ids=single_conv_ids, params=single_conv_params)
def single_conv(request):
    return request.param


lengts = [1, 2, 5]

seq_conv_params = [([-1.0], float),
                   ((2,), float),
                   (['10'], float),
                   ([np.array(2.3, dtype=float)], float),
                   ([np.array(2, dtype=int)], float),
                   ((-1,), int),
                   (['10'], int),
                   ((np.array(2, dtype=int),), int),
                   ([1], bool),
                   ((False,), bool)]


seq_conv_ids = [' input = {0[0]}, conv = {0[1]} '.format(p)
                for p in seq_conv_params]


@pytest.fixture(scope="module", ids=seq_conv_ids, params=seq_conv_params)
def seq_conv(request):
    return request.param


# For ndim = 3
axes_conv_params = [(0, (0,)),
                    (-1, (2,)),
                    ((1,), (1,)),
                    ([-1], (2,)),
                    ((1, 2, 0), (1, 2, 0)),
                    ((2, 1, -3), (2, 1, 0)),
                    ([0, 1], (0, 1)),
                    (np.arange(2), (0, 1))]
axes_conv_ids = [' axes={0[0]}, conv={0[1]} '.format(axis)
                 for axis in axes_conv_params]


@pytest.fixture(scope="module", ids=axes_conv_ids, params=axes_conv_params)
def axes_conv(request):
    return request.param


# --- normalized_scalar_param_list --- #


def test_normalized_scalar_param_list_single_val(length, single_conv):

    value, conversion = single_conv

    expected_noconv = [value] * length
    norm_param_noconv = normalized_scalar_param_list(value, length)
    assert expected_noconv == norm_param_noconv

    expected_conv = [conversion(value)] * length
    norm_param_conv = normalized_scalar_param_list(
        value, length, param_conv=conversion)
    assert expected_conv == norm_param_conv


def test_normalized_scalar_param_list_sequence(length, seq_conv):

    value, conversion = seq_conv
    value = value * length

    expected_noconv = list(value)
    norm_param_noconv = normalized_scalar_param_list(value, length)
    assert expected_noconv == norm_param_noconv

    expected_conv = [conversion(v) for v in value]
    norm_param_conv = normalized_scalar_param_list(
        value, length, param_conv=conversion)
    assert expected_conv == norm_param_conv


def test_normalized_scalar_param_list_with_none():

    param1 = [1, None, 0]

    def conv_int_none(x):
        if x is None:
            return 0
        else:
            return int(x)

    norm_param_noconv = normalized_scalar_param_list(param1, length=3)
    assert norm_param_noconv == param1

    norm_param_conv1 = normalized_scalar_param_list(
        param1, length=3, param_conv=conv_int_none, keep_none=True)
    assert norm_param_conv1 == param1

    norm_param_conv2 = normalized_scalar_param_list(
        param1, length=3, param_conv=conv_int_none, keep_none=False)
    assert norm_param_conv2 == [1, 0, 0]

    norm_param_noconv = normalized_scalar_param_list(None, length=3)
    assert norm_param_noconv == [None] * 3

    norm_param_conv1 = normalized_scalar_param_list(
        None, length=3, param_conv=conv_int_none, keep_none=True)
    assert norm_param_conv1 == [None] * 3

    norm_param_conv2 = normalized_scalar_param_list(
        None, length=3, param_conv=conv_int_none, keep_none=False)
    assert norm_param_conv2 == [0] * 3


def test_normalized_scalar_param_list_error():

    # Wrong length
    with pytest.raises(ValueError):
        normalized_scalar_param_list([1, 2], length=3)


# --- normalized_axes_tuple

def test_normalized_axes_tuple(axes_conv):
    """Test if all valid sequences are converted correctly."""
    axes, conversion = axes_conv
    assert normalized_axes_tuple(axes, ndim=3) == conversion


def test_normalized_axes_tuple_raise():
    """Test if errors are raised for invalid input."""

    with pytest.raises(TypeError):
        normalized_axes_tuple(1.5, ndim=3)  # float

    with pytest.raises(TypeError):
        normalized_axes_tuple(None, ndim=3)  # garbage

    with pytest.raises(ValueError):
        normalized_axes_tuple((0, 1.5), ndim=3)  # sequence containing float

    with pytest.raises(TypeError):
        normalized_axes_tuple((0, None), ndim=3)  # sequence containing garbge

    with pytest.raises(ValueError):
        normalized_axes_tuple((0, 0, 1), ndim=3)  # duplicate

    with pytest.raises(ValueError):
        normalized_axes_tuple((0,), ndim=0)  # nonpositive ndim

    with pytest.raises(ValueError):
        normalized_axes_tuple((0, 1), ndim=1)  # axis out of range

    with pytest.raises(ValueError):
        normalized_axes_tuple(-3, ndim=2)  # axis out of range

    with pytest.raises(ValueError):
        normalized_axes_tuple((0, 2), ndim=2)  # axis out of range


if __name__ == '__main__':
    odl.util.test_file(__file__)
