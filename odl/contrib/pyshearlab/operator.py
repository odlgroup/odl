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


__all__ = ('PyShearlabOperator',)


class PyShearlabOperator(odl.Operator):
    def __init__(self, space, scales):
        self.shearletSystem = pyshearlab.SLgetShearletSystem2D(0, space.shape[0], space.shape[1], scales)
        range = space ** self.shearletSystem['nShearlets']
        odl.Operator.__init__(self, space, range, True)
        
    def _call(self, x):
        result = pyshearlab.SLsheardec2D(x, self.shearletSystem)
        return np.moveaxis(result, -1, 0)
    
    @property
    def adjoint(self):
        op = self
        
        class PyShearlabOperatorAdjoint(odl.Operator):
            def _call(self, x):
                x = np.moveaxis(x, 0, -1)
                return pyshearlab.SLshearadjoint2D(x, op.shearletSystem)
            
            @property
            def adjoint(self):
                return op
            
        return PyShearlabOperatorAdjoint(self.range, self.domain, True)
    
    @property
    def inverse(self):
        op = self
        
        class PyShearlabOperatorInverse(odl.Operator):
            def _call(self, x):
                x = np.moveaxis(x, 0, -1)
                return pyshearlab.SLshearrec2D(x, op.shearletSystem)
            
            @property
            def adjoint(self):
                return op
            
        return PyShearlabOperatorInverse(self.range, self.domain, True)
