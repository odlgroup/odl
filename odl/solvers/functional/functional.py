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
from builtins import super

import numpy as np
from numbers import Number

from odl.operator.operator import (
    Operator, OperatorComp, OperatorLeftScalarMult, OperatorRightScalarMult,
    OperatorRightVectorMult, OperatorSum)
from odl.set.space import LinearSpaceVector


__all__ = ('Functional',)


class Functional(Operator):

    """Quick hack for a functional class."""

    def __init__(self, domain, linear=False, smooth=False, concave=False, convex=False, grad_lipschitz=np.inf):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `Set`
            The domain of this operator, i.e., the set of elements to
            which this operator can be applied
        linear : `bool`
            If `True`, the operator is considered as linear. In this
            case, ``domain`` and ``range`` have to be instances of
            `LinearSpace`, or `Field`.
        domain : `LinearSpace`
            Set of elements on which the functional can be evaluated
        smooth : `bool`, optional
            If `True`, assume that the functional is continuously differentiable
        convex : `bool`, optional
            If `True`, assume that the functional is convex
        concave : `bool`, optional
            If `True`, assume that the functional is concave
        """

        self._is_smooth = bool(smooth)
        
        self._is_convex = bool(convex)
        
        self._is_concave = bool(concave)

        self._grad_lipschitz = float(grad_lipschitz)

        #super()
        Operator.__init__(self, domain=domain, range=domain.field, linear=linear)

    @property
    def gradient(self):
        """Gradient operator of the functional.

        Parameters
        ----------
        x : domain element-like
            Point in which to evaluate the gradient
        out : domain element, optional
            Element into which the result is written

        Returns
        -------
        out : `Operator`
            Gradient operator of this functional.
        """
        raise NotImplementedError

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the functional.

        Parameters
        ----------
        sigma : positive float, optional
            Regularization parameter of the proximal operator  

        Returns
        -------
        out : Operator 
            Domain and range equal to domain of functional 
        """
        raise NotImplementedError

    @property
    def conjugate_functional(self):
        """Convex conjugate functional of the functional.

        Parameters
        ----------
        none
        
        Returns
        -------
        out : Functional 
            Domain equal to domain of functional 
        """
        raise NotImplementedError


    def derivative(self, point):
        functional = self
        
        
        class DerivativeOperator(Functional):
            def __init__(self):
                super().__init__(functional.domain, linear=True)

            self.point=point
            
            #self.gradi=functional.gradient()

            def _call(self, x):
                return x.inner(functional.gradient(point)) #gradi(point))
        
        return DerivativeOperator()


    def __mul__(self, other):
        """Return ``self * other``.

        If ``other`` is an operator, this corresponds to
        operator composition:

            ``op1 * op2 <==> (x --> op1(op2(x))``

        If ``other`` is a scalar, this corresponds to right
        multiplication of scalars with operators:

            ``op * scalar <==> (x --> op(scalar * x))``

        If ``other`` is a vector, this corresponds to right
        multiplication of vectors with operators:

            ``op * vector <==> (x --> op(vector * x))``

        Note that left and right multiplications are generally
        different.

        Parameters
        ----------
        other : {`Operator`, `LinearSpaceVector`, scalar}
            `Operator`:
                The `Operator.domain` of ``other`` must match this
                operator's `Operator.range`.

            `LinearSpaceVector`:
                ``other`` must be an element of this operator's
                `Operator.domain`.

            scalar:
                The `Operator.domain` of this operator must be a
                `LinearSpace` and ``other`` must be an
                element of the ``field`` of this operator's
                `Operator.domain`.

        Returns
        -------
        mul : `Functional`
            Multiplication result

            If ``other`` is an `Operator`, ``mul`` is a
            `FunctionalComp`.

            If ``other`` is a scalar, ``mul`` is a
            `FunctionalRightScalarMult`.

            If ``other`` is a vector, ``mul`` is a
            `FunctionalRightVectorMult`.

        """
        if isinstance(other, Operator):
            return FunctionalComp(self, other)
        elif isinstance(other, Number):
            # Left multiplication is more efficient, so we can use this in the
            # case of linear operator.
            raise NotImplementedError
            if self.is_linear:
                return FunctionalLeftScalarMult(self, other)
            else:
                return OperatorRightScalarMult(self, other)
        elif isinstance(other, LinearSpaceVector) and other in self.domain:
            raise NotImplementedError
            return OperatorRightVectorMult(self, other.copy())
        else:
            return NotImplemented



    def __rmul__(self, other):

        if isinstance(other, Operator):
            return OperatorComp(other, self)
        elif isinstance(other, Number):
            return FunctionalLeftScalarMult(self, other)
        #elif other in self.range:
        #    return OperatorLeftVectorMult(self, other.copy())
        elif (isinstance(other, LinearSpaceVector) and
              other.space.field == self.range):
            return FunctionalLeftVectorMult(self, other.copy())
        else:
            return NotImplemented



    @property
    def is_smooth(self):
        """`True` if this operator is continuously differentiable."""
        return self._is_smooth        

    @property
    def is_concave(self):
        """`True` if this operator is concave."""
        return self._is_concave    

    @property
    def is_convex(self):
        """`True` if this operator is convex."""
        return self._is_convex   

    @property
    def grad_lipschitz(self):
        """Lipschitz constant for the gradient of the functional"""
        return self._grad_lipschitz   



class FunctionalLeftScalarMult(Functional, OperatorLeftScalarMult):

    """Scalar multiplication a functional."""

    def __init__(self, func, scalar):

        """Initialize a new instance.

        Parameters
        ----------
        scal : `Scalar`
            Scalar argument
        func : `Functional`
            The right ("inner") functional
        """

        if not isinstance(func, Functional):
            raise TypeError('functional {!r} is not a Functional instance.'
                            ''.format(func))

        scalar=func.range.element(scalar)
      
        #Functional.__init__(self, domain=func.domain)        
        
        if scalar>0:
            Functional.__init__(self, domain=func.domain, linear=func.is_linear, smooth=func.is_smooth, concave=func.is_concave, convex=func.is_convex, grad_lipschitz=scalar*func.grad_lipschitz)
        elif scalar==0:
            Functional.__init__(self, domain=func.domain, linear=True, smooth=True, concave=True, convex=True, grad_lipschitz=0)
        else:
            Functional.__init__(self, domain=func.domain, linear=func.is_linear, smooth=func.is_smooth, concave=func.is_convex, convex=func.is_concave, grad_lipschitz=-scalar*func.grad_lipschitz)


        OperatorLeftScalarMult.__init__(self, op=func, scalar=scalar)


        self._func = func
        self._scalar = scalar
                
        
 #   def _call(self, x):
 #       return self._scalar*self._func(x)     

    @property
    def gradient(self, x):
        return self._scalar * self._func.gradient(x)

        



class FunctionalComp(Functional, OperatorComp):

    """Composition of a functional with an operator."""

    def __init__(self, func, op, tmp1=None, tmp2=None):
        """Initialize a new instance.

        Parameters
        ----------
        func : `Functional`
            The left ("outer") operator
        op : `Operator`
            The right ("inner") operator. Its range must coincide with the
            domain of ``func``.
        tmp1 : `element` of the range of ``op``, optional
            Used to avoid the creation of a temporary when applying ``op``
        tmp2 : `element` of the range of ``op``, optional
            Used to avoid the creation of a temporary when applying the
            gradient of ``func``
        """
        if not isinstance(func, Functional):
            raise TypeError('functional {!r} is not a Functional instance.'
                            ''.format(func))

        OperatorComp.__init__(self, left=func, right=op, tmp=tmp1)

        if tmp2 is not None and tmp2 not in self._left.domain:
            raise TypeError('second temporary {!r} not in the domain '
                            '{!r} of the functional.'
                            ''.format(tmp2, self._left.domain))
        self._tmp2 = tmp2

    def gradient(self, x, out=None):
        """Gradient of the compositon according to the chain rule.

        Parameters
        ----------
        x : domain element-like
            Point in which to evaluate the gradient
        out : domain element, optional
            Element into which the result is written

        Returns
        -------
        out : domain element
            Result of the gradient calcuation. If ``out`` was given,
            the returned object is a reference to it.
        """
        if out is None:
            return self._right.derivative(x).adjoint(
                self._left.gradient(self._right(x)))
        else:
            if self._tmp is not None:
                tmp_op_ran = self._right(x, out=self._tmp)
            else:
                tmp_op_ran = self._right(x)

            if self._tmp2 is not None:
                tmp_dom = self._left.gradient(tmp_op_ran, out=self._tmp2)
            else:
                tmp_dom = self._left.gradient(tmp_op_ran)

            self._right.derivative(x).adjoint(tmp_dom, out=out)


class FunctionalSum(Functional, OperatorSum):

    """Expression type for the sum of functionals.

    ``FunctionalSum(func1, func2) <==> (x --> func1(x) + func2(x))``
    """

    def __init__(self, func1, func2, tmp_dom=None):
        """Initialize a new instance.

        Parameters
        ----------
        func1, func2 : `Functional`
            The summands of the functional sum. Their `Operator.domain`
            and `Operator.range` must coincide.
        tmp_dom : `Operator.domain` `element`, optional
            Used to avoid the creation of a temporary when applying the
            gradient.
        """
        if not isinstance(func1, Functional):
            raise TypeError('functional 1 {!r} is not a Functional instance.'
                            ''.format(func1))
        if not isinstance(func2, Functional):
            raise TypeError('functional 2 {!r} is not a Functional instance.'
                            ''.format(func2))

        OperatorSum.__init__(self, func1, func2, tmp_ran=None,
                             tmp_dom=tmp_dom)

    def gradient(self, x, out=None):
        """Evaluate the gradient of the functional sum."""
        if out is None:
            return self._op1.gradient(x) + self._op2.gradient(x)
        else:
            tmp = (self._tmp_dom if self._tmp_dom is not None
                   else self.domain.element())
            self._op1.gradient(x, out=out)
            self._op2.gradient(x, out=tmp)
            out += tmp
            return out
