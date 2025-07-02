# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

__all__ = (
    'cumulative_prod',
    'cumulative_sum',
    'max',
    'mean',
    'min',
    'prod',
    'std',
    'sum',
    'var'   
)

# TODO: add kwargs handling
# TODO: rename module to 'statistical' to be array API compliant
def _apply_reduction(operation: str, x):
    return x.space._element_reduction(operation=operation, x=x)

def cumulative_prod(x):
    """Calculates the cumulative product of elements in the input array x."""
    return _apply_reduction('cumulative_prod', x)

def cumulative_sum(x):
    """Calculates the cumulative sum of elements in the input array x."""
    return _apply_reduction('cumulative_sum', x)

def max(x):
    """Calculates the maximum value of the input array x."""
    return _apply_reduction('max', x)

def mean(x):
    """Calculates the arithmetic mean of the input array x."""
    return _apply_reduction('mean', x)

def min(x):
    """Calculates the minimum value of the input array x."""
    return _apply_reduction('min', x)

def prod(x):
    "Calculates the product of input array x elements."
    return _apply_reduction('prod', x)

def std(x):
    """Calculates the standard deviation of the input array x."""
    return _apply_reduction('std', x)

def sum(x):
    """Calculates the sum of the input array x."""
    return _apply_reduction('sum', x)

def var(x):
    """Calculates the variance of the input array x."""
    return _apply_reduction('var', x)
