# -*- coding: utf-8 -*-
"""
simple_test_astra.py -- a simple test script

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
from __future__ import division, print_function, unicode_literals, absolute_import
from future import standard_library
standard_library.install_aliases()

class storePartial(object):
    """ Simple object for storing all partial results of the solvers
    """
    def __init__(self):
        self.results = []

    def send(self,result):
        self.results.append(result.copy())

    def __iter__(self):
        return self.results.__iter__()

class forEachPartial(object):
    """ Simple object for applying a function to each iterate
    """
    def __init__(self, function):
        self.function = function

    def send(self,result):
        self.function(result)

class printStatusPartial(object):
    """ Prints the interation count and current norm of each iterate
    """
    def __init__(self):
        self.iter = 0

    def send(self, result):
        print("iter = {}, norm = {}".format(self.iter, result.norm()))
        self.iter += 1

def landweber(operator, x, rhs, iterations=1, omega=1, partialResults=None):
    """ General and efficient implementation of Landweber iteration
    """

    #Reusable temporaries
    tmpRan = operator.range.empty()
    tmpDom = operator.domain.empty()

    for _ in range(iterations):
        operator.apply(x, tmpRan)                                   #tmpRan = Ax
        tmpRan -= rhs                                               #tmpRan = tmpRan - rhs
        operator.getDerivative(x).applyAdjoint(tmpRan, tmpDom)      #tmpDom = A^T tmpRan
        x.linComb(-omega, tmpDom)                                   #x = x - omega * tmpDom

        if partialResults is not None:
            partialResults.send(x)


def conjugateGradient(operator, x, rhs, iterations=1, partialResults=None):
    """ Optimized version of CGN, uses no temporaries etc.
    """

    d = operator(x)
    d.space.linComb(1, rhs, -1, d)       #d = rhs - A x
    p = operator.T(d)
    s = p.copy()
    q = operator.range.empty()
    normsOld = s.normSq()               #Only recalculate norm after update

    for _ in range(iterations):
        operator.apply(p, q)                                    #q = A p
        a = normsOld / q.normSq()
        x.linComb(a, p)                                         #x = x + a*p
        d.linComb(-a, q)                                        #d = d - a*q
        operator.getDerivative(p).applyAdjoint(d, s)            #s = A^T d

        normsNew = s.normSq()
        b = normsNew/normsOld
        normsOld = normsNew

        operator.domain.linComb(1, s, b, p)      #p = s + b * p

        if partialResults is not None:
            partialResults.send(x)