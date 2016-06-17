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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import pytest

# Internal
from odl.util.normalize import normalized_scalar_param_list


length_params = [1, 2]
length_ids = [' length = {} '.format(p) for p in length_params]


@pytest.fixture(scope="module", ids=length_ids, params=length_params)
def length(request):
    return request.param


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


def test_normalized_scalar_param_list_single_val(length, single_conv):

    value, conversion = single_conv

    expected_noconv = [value] * length
    norm_param_noconv = normalized_scalar_param_list(value, length)
    assert expected_noconv == norm_param_noconv

    expected_conv = [conversion(value)] * length
    norm_param_conv = normalized_scalar_param_list(
        value, length, param_conv=conversion)
    assert expected_conv == norm_param_conv


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


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
