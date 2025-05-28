from numpy.typing import ArrayLike

from .numpy_weighting import NumpyWeighting

WEIGHTING_IMPLS = {
    'numpy': NumpyWeighting,
    }

def space_weighting(
        impl : str,
        device = 'cpu',
        **kwargs
    ):
    """
    Notes: 
        To instanciate a weigthing, one can use a variety of mutually exclusive parameters
        1) inner (callable): the inner product between two elements of the space
        2) norm (callable): the norm of an element of the space
            -> sqrt(inner(x,x).real)
        3) dist (callable): the distance between two elements of the space
            -> norm(x1-x2)
        4) weight (float | ArrayLike): Scalar or element-wise weighting of the space elements
        5) exponent (float): exponent of the norm
    """
    ########## Parsing the input parameters ##########
    dist   : callable = kwargs.get("dist", None)
    norm   : callable = kwargs.get("norm", None)
    inner  : callable = kwargs.get("inner", None)
    weight : float | ArrayLike = kwargs.get("weight", None)
    exponent : float = kwargs.get("exponent", 2.0)
    ########## Performing checks ##########
    # Parsing implementation
    assert impl in WEIGHTING_IMPLS, f"impl arg must be in {WEIGHTING_IMPLS} but {impl} was provided"
    # We do not allow the use of callables if the exponent is not equal to 2
    if exponent != 2.0 and any(x is not None for x in (dist, norm, inner)):
            raise ValueError(
                f"cannot use any of `dist`, `norm` or `inner` for exponent != 2, but {exponent} was provided"
            )
    # Check validity of option combination (0 or 1 may be provided)
    num_extra_args = sum(a is not None for a in (dist, norm, inner, weight))
    if num_extra_args > 1:
        raise ValueError(
            "invalid combination of options `weighting`, "
            "`dist`, `norm` and `inner`"
        )
    # Check the dtype of the weight
    if weight is not None:
        if not hasattr(weight, '__array__') and (not isinstance(weight, float)):
            raise TypeError(f"If provided, the weight must be a positive float or an array with positive entries or an odl Tensor with positive data, but a weight of type {type(weight)} was provided.")
    # Choosing the implementation
    weighting_impl = WEIGHTING_IMPLS[impl]
    return weighting_impl(device, **kwargs)