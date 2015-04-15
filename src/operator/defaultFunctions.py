# -*- coding: utf-8 -*-
"""
operator.py -- functional analytic operators

Copyright 2014, 2015 Holger Kohr

This file is part of RL.

RL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RL.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import unicode_literals, print_function, division, absolute_import
from future.builtins import object, zip
from future import standard_library
standard_library.install_aliases()

from numbers import Number
from abc import ABCMeta, abstractmethod, abstractproperty
import RL.operator.space as space
from RL.operator.function import *
from RL.space.set import RealNumbers

class RegisteringFunction(object):
    def __init__(self):
        self.registeredFunctions = {}

    def applyImpl(self, *args):
        types = tuple(type(x) for x in args)
        function = self.registeredFunctions[types]
        if (function is None):
            raise NotImplementedError("Function not implemented for this type")

    def register(self, function, *types):
        self.registeredFunctions[types] = function


abs = RegisteringFunction()

def defaultAbs(vectorIn):
    vectorOut = vectorIn.space.empty()
    abs(vectorIn,vectorOut)
    return vectorOut

abs.register(defaultAbs, space.LinearSpace.Vector)


sign = RegisteringFunction()

def defaultSign(vectorIn):
    vectorOut = vectorIn.space.empty()
    sign(vectorIn,vectorOut)
    return vectorOut

sign.register(defaultSign, space.LinearSpace.Vector)


addScalar = RegisteringFunction()

def defaultAddScalar(vectorIn):
    vectorOut = vectorIn.space.empty()
    addScalar(vectorIn,vectorOut)
    return vectorOut

addScalar.register(defaultSign, space.LinearSpace.Vector, Number)


max = RegisteringFunction()