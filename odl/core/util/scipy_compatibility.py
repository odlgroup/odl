# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Scipy compatibility module"""

import warnings
from os import environ
if 'SCIPY_ARRAY_API' in environ and environ['SCIPY_ARRAY_API']=='1':
        pass
else:
    warnings.warn("The environment variable SCIPY_ARRAY_API must be set to 1."
                + " It should be by default when importing odl, but it seems that scipy was imported before odl."
                + " If not set, the array API support of scipy will be disabled,"
                + " meaning that function calls such as ``xlogy`` on GPU will error"
                + " and throw back pytorch Type errors."
                + " Please add the following lines before your first scipy import. \n"
                + 'from os import environ \n' \
                + 'environ["SCIPY_ARRAY_API"]=="1" \n ' \
                + '********End of Warning********',
              stacklevel=2)

import scipy
import numpy

__all__ = (
    'lambertw',
    'scipy_lambertw',
    'xlogy',
    )

def _helper(operation:str, x1, x2=None, out=None, namespace=scipy.special, **kwargs):

def _helper(operation: str, x1, x2=None, out=None, namespace=scipy.special, **kwargs):
    """Internal helper to pass

    Args:
        operation (str): Name of the operation
        x1  (Tensor): _description_
        x2  (Tensor, optional): _description_. Defaults to None.
        out (Tensor, optional): Out argument for in-place operations. Defaults to None.
        namespace: scipy namespace to get the operation from. Defaults to scipy.special.

    Returns:
        Tensor
    """
    return x1.space._elementwise_num_operation(
        operation=operation, x1=x1, x2=x2, out=out, namespace=namespace, **kwargs)

def lambertw(x, k=0, tol=1e-8):
    """
    Lambert W function.

    The Lambert W function W(z) is defined as the inverse function of w * exp(w). In other words, the value of W(z) is such that z = W(z) * exp(W(z)) for any complex number z.

    The Lambert W function is a multivalued function with infinitely many branches. Each branch gives a separate solution of the equation z = w exp(w). Here, the branches are indexed by the integer k.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lambertw.html#scipy.special.lambertw
    """
    return _helper('lambertw', x, k=k, tol=tol)


def scipy_lambertw(x, k=0, tol=1e-8):
    """
    Lambert W function.

    The Lambert W function W(z) is defined as the inverse function of w * exp(w). In other words, the value of W(z) is such that z = W(z) * exp(W(z)) for any complex number z.

    The Lambert W function is a multivalued function with infinitely many branches. Each branch gives a separate solution of the equation z = w exp(w). Here, the branches are indexed by the integer k.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lambertw.html#scipy.special.lambertw

    Note:
        This function is a direct call to scipy.special.lambertw on a Numpy Array!
    """
    assert isinstance(
        x, numpy.ndarray
    ), "Can only call scipy_lambertw on nd_array. For ODL Tensors, please use the function scipy_compatibility.lambertw"
    return scipy.special.lambertw(x, k, tol)


def xlogy(x1, x2, out=None):
    """Compute x*log(y) so that the result is 0 if x = 0.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.xlogy.html
    """
    return _helper('xlogy', x1=x1, x2=x2, out=out)
