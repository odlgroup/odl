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
    # Parsing implementation
    assert impl in WEIGHTING_IMPLS, f"impl arg must be in {WEIGHTING_IMPLS} but {impl} was provided"
    # Choosing the implementation
    weighting_impl = WEIGHTING_IMPLS[impl]
    return weighting_impl(device, **kwargs)