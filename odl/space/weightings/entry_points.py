import numpy as np
from numpy.typing import ArrayLike

from odl.space.weighting import Weighting, ConstWeighting, ArrayWeighting, CustomInner, CustomNorm, CustomDist

def space_weighting(
        impl : str,
        device = 'cpu',
        **kwargs
    ):
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
        
        elif hasattr(weight, 'odl_tensor'):
            if np.all(0 < weight.data):
                assert impl == weight.impl
                weight = weight.data
                assert device == weight.device
            else:
                raise ValueError("If the weight is an ODL Tensor, all its entries must be positive")
            
        elif hasattr(weight, '__array__'):
            if np.all(0 < weight):
                pass
                assert device == weight.device
            else:
                raise ValueError("If the weight is an array, all its elements must be positive")          

        else:
            raise ValueError(f"A weight can only be a positive __array__, a positive float or a positive ODL Tensor")      

        return ArrayWeighting(array=weight, impl=impl, device=device, exponent=exponent)

    elif kwargs == {}:
        # TODO handle boolean case
        return ConstWeighting(const=1.0, impl=impl, device=device)

    elif kwargs == {'exponent': exponent}:
        return ConstWeighting(const=1.0, exponent=exponent, impl=impl, device=device)

    raise TypeError('got unknown keyword arguments {}'.format(kwargs))
