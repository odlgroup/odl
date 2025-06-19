from .base_weighting import Weighting

import array_api_compat.numpy as xp

THRESHOLD_MEDIUM = 50000
REAL_DTYPES = [xp.float32, xp.float64]

class NumpyWeighting(Weighting):
    def __init__(self, device:str, **kwargs):
        
        super(NumpyWeighting, self).__init__(device, **kwargs)

    @property
    def array_namespace(self):
        return xp
    
    @property
    def impl(self):
        return 'numpy'
    
    @property
    def array_type(self):
        return xp.ndarray
    
    def _inner_default(self, x1, x2):
        assert x1.shape == x2.shape
        if x1.dtype in REAL_DTYPES:
            if x1.size > THRESHOLD_MEDIUM:
                # This is as fast as BLAS dotc
                result = xp.tensordot(x1, x2, [range(x1.ndim)] * 2)
            else:
                # Several times faster for small arrays
                result = xp.dot(x1.ravel(), x2.ravel())
            return result.astype(float)
        else:
            # x2 as first argument because we want linearity in x1
            return xp.vdot(x2.ravel(), x1.ravel()).astype(complex)
        
    def _norm_default(self, x):
        if isinstance(self.weight, (int, float)):
            if self.exponent == 2.0:
                return float(xp.sqrt(self.weight) * xp.linalg.norm(x.data.ravel(), ord = self.exponent))
            elif self.exponent == float('inf'):
                return float(self.weight * xp.linalg.norm(x.data.ravel(), ord = self.exponent))
            else:
                return float((self.weight ** (1 / self.exponent) *
                            xp.linalg.norm(x.data.ravel(), ord = self.exponent)))
        elif isinstance(self.weight, self.array_type):
            if self.exponent == 2.0:
                norm_squared = self.inner(x, x).real  # TODO: optimize?!
                if norm_squared < 0:
                    norm_squared = 0.0  # Compensate for numerical error
                return float(xp.sqrt(norm_squared))
            else:
                return float(self._pnorm_diagweight(x))
            
    def _dist_default(self, x1, x2):
        return self._norm_default(x1-x2)
        
    def _pnorm_diagweight(self,x):
        """Diagonally weighted p-norm implementation."""

        # This is faster than first applying the weights and then summing with
        # BLAS dot or nrm
        x_p = xp.abs(x.data.ravel())
        if self.exponent == float('inf'):
            x_p *= self.weight.ravel()
            return xp.max(x_p)
        else:
            x_p = xp.power(x_p, self.exponent, out=x_p)
            x_p *= self.weight.ravel()
            return xp.sum(x_p) ** (1/self.exponent)
            