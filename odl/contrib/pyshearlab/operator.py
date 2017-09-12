# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""ODL integration with pyshearlab."""


import odl
import numpy as np
import pyshearlab
from threading import Lock


__all__ = ('PyShearlabOperator',)


class PyShearlabOperator(odl.Operator):
    def __init__(self, space, scales):
        self.shearletSystem = pyshearlab.SLgetShearletSystem2D(
            0, space.shape[0], space.shape[1], scales)
        range = space ** self.shearletSystem['nShearlets']
        self.mutex = Lock()
        odl.Operator.__init__(self, space, range, True)

    def _call(self, x):
        with self.mutex:
            result = pyshearlab.SLsheardec2D(x, self.shearletSystem)
            return np.moveaxis(result, -1, 0)

    @property
    def adjoint(self):
        return PyShearlabOperatorAdjoint(self)

    @property
    def inverse(self):
        return PyShearlabOperatorInverse(self)


class PyShearlabOperatorAdjoint(odl.Operator):
    def __init__(self, op):
        self.op = op
        odl.Operator.__init__(self, op.range, op.domain, True)

    def _call(self, x):
        with self.op.mutex:
            x = np.moveaxis(x, 0, -1)
            return pyshearlab.SLshearadjoint2D(x, self.op.shearletSystem)

    @property
    def adjoint(self):
        return self.op

    @property
    def inverse(self):
        return PyShearlabOperatorAdjointInverse(self.op)


class PyShearlabOperatorInverse(odl.Operator):
    def __init__(self, op):
        self.op = op
        odl.Operator.__init__(self, op.range, op.domain, True)

    def _call(self, x):
        with self.op.mutex:
            x = np.moveaxis(x, 0, -1)
            return pyshearlab.SLshearrec2D(x, self.op.shearletSystem)

    @property
    def adjoint(self):
        return PyShearlabOperatorAdjointInverse(self.op)

    @property
    def inverse(self):
        return self.op


class PyShearlabOperatorAdjointInverse(odl.Operator):
    def __init__(self, op):
        self.op = op
        odl.Operator.__init__(self, op.domain, op.range, True)

    def _call(self, x):
        with self.op.mutex:
            result = pyshearlab.SLshearrecadjoint2D(x, self.op.shearletSystem)
            return np.moveaxis(result, -1, 0)

    @property
    def adjoint(self):
        return self.op.inverse

    @property
    def inverse(self):
        return self.op.adjoint
