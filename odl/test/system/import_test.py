# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division
import pytest


def test_all_imports():
    import odl
    # Create Cn
    odl.cn(3)
    odl.space.cn(3)
    C3 = odl.space.space_utils.cn(3)

    # Three ways of creating the identity
    odl.IdentityOperator(C3)
    odl.operator.IdentityOperator(C3)
    odl.operator.default_ops.IdentityOperator(C3)

    # Test that utility needs to be explicitly imported
    odl.util.utility.array1d_repr
    with pytest.raises(AttributeError):
        odl.array1d_repr


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
