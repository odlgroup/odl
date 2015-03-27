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


def landweberBase(operator, x0, rhs, omega=1, iterations=1):
    """ Straightforward implementation of Landweber iteration
    """
    x = x0.copy()
    for _ in range(iterations):
        x = x - omega * operator.T(operator(x)-rhs)

        yield x


def landweber(operator, x0, rhs, omega=1, iterations=1):
    """ General and efficient implementation of Landweber iteration
    """
    x = x0.copy()

    #Reusable temporaries
    tmpRan = operator.range.empty()
    tmpDom = operator.domain.empty()
    for _ in range(iterations):
        #Optimized code (as an example)
        operator.apply(x, tmpRan)                   #tmpRan = Ax
        tmpRan -= rhs                               #tmpRan = tmpRan - rhs
        operator.applyAdjoint(tmpRan, tmpDom)       #tmpDom = A^T tmpRan
        x.linComb(-omega, tmpDom)                   #x = x - omega * tmpDom

        yield x
        

def conjugateGradientBase(op, x0, rhs, iterations=1):
    """ Non-optimized CGN
    """
    x = x0.copy()
    d = rhs - op(x)
    p = op.T(d)
    s = p.copy()

    for _ in range(iterations):
        q = op(p)                       
        norms2 = s.normSq()
        a = norms2 / q.normSq()
        x = x + a*p                    
        d = d - a*q                  
        s = op.T(d)
        b = s.normSq()/norms2
        p = s + b*p

        yield x


def conjugateGradient(op, x0, rhs, iterations=1):
    """ Optimized version of CGN, uses no temporaries etc.
    """
    x = x0.copy()
    d = op(x)
    d.space.linComb(1, rhs, -1, d)       #d = rhs - A x
    p = op.T(d)
    s = p.copy()
    q = op.range.empty()
    normsOld = s.normSq()               #Only recalculate norm after update

    for _ in range(iterations):
        op.apply(p, q)                   #q = A p
        a = normsOld / q.normSq()
        x.linComb(a, p)                  #x = x + a*p
        d.linComb(-a, q)                 #d = d - a*q
        op.applyAdjoint(d, s)            #s = A^T d

        normsNew = s.normSq()
        b = normsNew/normsOld
        normsOld = normsNew

        op.domain.linComb(1, s, b, p)      #p = s + b * p

        yield x
