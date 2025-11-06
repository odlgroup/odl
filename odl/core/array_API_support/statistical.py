# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Satistical functions expected by the python array API
Internally, all functions apply a reduction operation on a LinearSpaceElement.

Args:
    x (LinearSpaceElement): LinearSpaceElement on which to apply the reduction. 

Returns:
    x (float | array-like): Output of the reduction. 

Notes:
    1) The actual implementation of the reduction is in the LinearSpace of this element.
    2) These functions can return python Numbers or backend-specific array (when calling with keepdims=True for instance), but they will not return odl objects.

"""

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

def _apply_reduction(operation: str, x, **kwargs):
    """
    Examples
    >>> e1 = odl.rn(3).element((1,2,3))
    >>> odl.cumulative_prod(e1) == [1,2,6]
    array([ True,  True,  True], dtype=bool)
    >>> odl.cumulative_sum(e1) == [1,3,6]
    array([ True,  True,  True], dtype=bool)
    >>> odl.max(e1) == 3
    True
    >>> odl.mean(e1) == 2
    True
    >>> odl.min(e1) == 1
    True
    >>> odl.prod(e1) == 6
    True
    >>> odl.std(e1) == np.std([1,2,3])
    True
    >>> odl.sum(e1) == 6
    True
    >>> odl.var(e1) == np.var([1,2,3])
    True
    """
    return x.space._element_reduction(operation=operation, x=x, **kwargs)

def cumulative_prod(x, axis=None, dtype=None, include_initial=False):
    """
    Calculates the cumulative product of elements in the input array x.
    Note: This function might not be doing what you expect. If you want to return an array (np.ndarray, torch.Tensor...), you are in the right place. However, you cannot use it to create a new LinearSpaceSelement.
    """
    return _apply_reduction('cumulative_prod', x, axis=axis, dtype=dtype, include_initial=include_initial)

def cumulative_sum(x, axis=None, dtype=None, include_initial=False):
    """
    Calculates the cumulative sum of elements in the input array x.
    Note: This function might not be doing what you expect. If you want to return an array (np.ndarray, torch.Tensor...), you are in the right place. However, you cannot use it to create a new LinearSpaceSelement.
    """
    return _apply_reduction('cumulative_sum', x, axis=axis, dtype=dtype, include_initial=include_initial)

def max(x, axis=None, keepdims=False):
    """
    Calculates the maximum value of the input array x.
    Note: This function might not be doing what you expect. If you want to return an array (np.ndarray, torch.Tensor...), you are in the right place. However, you cannot use it to create a new LinearSpaceSelement.
    """
    return _apply_reduction('max', x, axis=axis, keepdims=keepdims)

def mean(x, axis=None, keepdims=False):
    """
    Calculates the arithmetic mean of the input array x.
    Note: This function might not be doing what you expect. If you want to return an array (np.ndarray, torch.Tensor...), you are in the right place. However, you cannot use it to create a new LinearSpaceSelement.
    """
    return _apply_reduction('mean', x, axis=axis, keepdims=keepdims)

def min(x, axis=None, keepdims=False):
    """
    Calculates the minimum value of the input array x.
    Note: This function might not be doing what you expect. If you want to return an array (np.ndarray, torch.Tensor...), you are in the right place. However, you cannot use it to create a new LinearSpaceSelement.
    """
    return _apply_reduction('min', x, axis=axis, keepdims=keepdims)

def prod(x, axis=None, dtype=None, keepdims=False):
    """
    Calculates the product of input array x elements.
    Note: This function might not be doing what you expect. If you want to return an array (np.ndarray, torch.Tensor...), you are in the right place. However, you cannot use it to create a new LinearSpaceSelement.
    """
    return _apply_reduction('prod', x, axis=axis, dtype=dtype, keepdims=keepdims)

def std(x, axis=None, correction=0.0, keepdims=False):
    """
    Calculates the standard deviation of the input array x.
    Note: This function might not be doing what you expect. If you want to return an array (np.ndarray, torch.Tensor...), you are in the right place. However, you cannot use it to create a new LinearSpaceSelement.
    """
    return _apply_reduction('std', x, axis=axis, correction=correction, keepdims=keepdims)

def sum(x, axis=None, dtype=None, keepdims=False):
    """
    Calculates the sum of the input array x.
    Note: This function might not be doing what you expect. If you want to return an array (np.ndarray, torch.Tensor...), you are in the right place. However, you cannot use it to create a new LinearSpaceSelement.
    """
    return _apply_reduction('sum', x, axis=axis, dtype=dtype, keepdims=keepdims)

def var(x, axis=None, correction=0.0, keepdims=False):
    """    
    Calculates the variance of the input array x.
    Note: This function might not be doing what you expect. If you want to return an array (np.ndarray, torch.Tensor...), you are in the right place. However, you cannot use it to create a new LinearSpaceSelement.
    """
    return _apply_reduction('var', x, axis=axis, correction=correction, keepdims=keepdims)
