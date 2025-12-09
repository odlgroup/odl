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

__all__ = (
    'lambertw',
    'scipy_lambertw',
    'xlogy',
    )

def _helper(operation:str, x1, x2=None, out=None, namespace=scipy.special, **kwargs):
    return x1.space._elementwise_num_operation(
        operation=operation, x1=x1, x2=x2, out=out, namespace=namespace, **kwargs)

def lambertw(x, k=0, tol=1e-8):
    return _helper('lambertw', x, k=k, tol=tol)

def scipy_lambertw(x, k=0, tol=1e-8):
    return scipy.special.lambertw(x, k, tol)

def xlogy(x1, x2, out=None):
    return _helper('xlogy', x1=x1, x2=x2, out=out)