# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Satistical functions expected by the python array API"""

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
    """Helper function to apply a reduction operation on a LinearSpaceElement.

    Note:
    The actual implementation of the reduction is in the LinearSpace of this element.
    Args:
        operation (str): Identifier of the function. 
        x (LinearSpaceElement): LinearSpaceElement on which to apply the reduction. 

    Returns:
        x (float | array-like): Output of the reduction. 
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
