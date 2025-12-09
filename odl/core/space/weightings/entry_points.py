# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

# Necessary for pytorch device string formatting
# pylint: disable=unnecessary-dunder-call
from odl.core.array_API_support import get_array_and_backend, lookup_array_backend
from .weighting import ConstWeighting, ArrayWeighting, CustomInner, CustomNorm, CustomDist

def space_weighting(impl: str, device='cpu', **kwargs):
    """
    Notes:
        To instantiate a weigthing, one can use a variety of mutually exclusive parameters
        1) inner (callable): the inner product between two elements of the space
        2) norm (callable): the norm of an element of the space
            -> sqrt(inner(x,x).real)
        3) dist (callable): the distance between two elements of the space
            -> norm(x1-x2)
        4) weight (float | ArrayLike): Scalar or element-wise weighting of the space elements

        In case a weight was provided, additionally the following is supported:
        4A) exponent (float): exponent of the summands in the norm, used for Banach spaces like L¹
        If the exponent is 2, the weight is then used for defining an inner product and the
        other operations, whereas for other exponents only the norm and distance are enabled.

        For a custom inner-product space, the exponent must be 2 (the default). The inner product
        also implies a norm and distance then.
        A custom norm defines a distance but will disable the inner product. A custom distance
        disables all other operations.
    """

    if 'exponent' in kwargs:
        # Pop the kwarg
        exponent = kwargs['exponent']
        assert not set(['norm', 'dist']).issubset(kwargs)
        # Assign the attribute
        if exponent != 2:
            assert 'inner' not in kwargs
    else:
        exponent = 2

    if 'inner' in kwargs:
        # Pop the kwarg
        inner = kwargs.pop('inner')
        # check the kwarg
        assert callable(inner)
        # Check the consistency
        assert exponent == 2

        for arg in ['norm', 'dist', 'weight']:
            if arg in kwargs:
                raise ValueError(f"If a custom inner product is specified, the weighting cannot also have custom {arg}={kwargs[arg]}.")
        
        return CustomInner(inner, device=device, impl=impl)
        
    elif 'norm' in kwargs:
        # Pop the kwarg
        array_norm = kwargs.pop('norm')
        # check the kwarg
        assert callable(array_norm)
        # Check the consistency
        for arg in ['exponent', 'inner', 'dist', 'weight']:
            if arg in kwargs:
                raise ValueError(f"If a custom norm is specified, the weighting cannot also have custom {arg}={kwargs[arg]}.")

        return CustomNorm(array_norm, device=device, impl=impl)
    
    elif 'dist' in kwargs:
        # Pop the kwarg
        dist  = kwargs.pop('dist')
        # check the kwarg
        assert callable(dist)
        # Check the consistency
        for arg in ['exponent', 'inner', 'norm', 'weight']:
            if arg in kwargs:
                raise ValueError(f"If a custom distance is specified, the weighting cannot also have custom {arg}={kwargs[arg]}.")
        
        
        return CustomDist(dist, device=device, impl=impl)
    
    elif 'weight' in kwargs:
        # Pop the kwarg
        weight = kwargs.pop('weight')
        # Check the consistency
        for arg in ['inner', 'norm', 'dist']:
            if arg in kwargs:
                raise ValueError(f"If a custom weight is specified, the weighting cannot also have custom {arg}={kwargs[arg]}.")
        
        if isinstance(weight, (int, float)):
            if 0 < weight and weight != float('inf'):
                weight = float(weight)
            else:
                raise ValueError("If the weight is a scalar, it must be positive")
            return ConstWeighting(const=weight, impl=impl, device=device, exponent=exponent)
        
        elif isinstance(weight, (tuple, list)):
            array_backend = lookup_array_backend(impl)
            weight = array_backend.array_constructor(weight, device=device)
            if array_backend.array_namespace.any(weight < 0):
                raise ValueError("If the weight is an array, all its elements must be positive")
                    
        elif hasattr(weight, '__array__'):
            weight, backend = get_array_and_backend(weight)
            if backend.array_namespace.all(0 < weight):
                assert (
                    device == weight.device.__str__()
                ), (f"The weighing is expecting the device {device}, but
                 + f" the array provided for the weight has a device {weight.device}."
                 +  " Please make sure that the two devices are consistent")
            else:
                raise ValueError(
                    "If the weight is an array, all its elements must be positive")

        else:
            raise ValueError(
                "A weight can only be a positive __array__, or a positive float.")

        return ArrayWeighting(array=weight, impl=impl, device=device, exponent=exponent)

    elif kwargs == {}:
        # TODO handle boolean case
        return ConstWeighting(const=1.0, impl=impl, device=device)

    elif kwargs == {'exponent': exponent}:
        return ConstWeighting(const=1.0, exponent=exponent, impl=impl, device=device)

    raise TypeError(f"got unknown keyword arguments {kwargs}")
