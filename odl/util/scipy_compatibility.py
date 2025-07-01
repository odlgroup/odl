import scipy

__all__ = (
    'lambertw',
    'xlogy',
    )

def _helper(operation:str, x1, x2=None, out=None, namespace=scipy.special, **kwargs):
    return x1.space._elementwise_num_operation(
        operation=operation, x1=x1, x2=x2, out=out, namespace=namespace, **kwargs)

def lambertw(x, k=0, tol=1e-8):
    return _helper('lambertw', x, k=k, tol=tol)

def xlogy(x1, x2, out=None):
    return _helper('xlogy', x1=x1, x2=x2, out=out)