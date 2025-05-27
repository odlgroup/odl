from .base_weighting import Weighting

import array_api_compat.numpy as xp

class NumpyWeighting(Weighting):
    def __init__(self, **kwargs):
        Weighting.__init__(self, **kwargs)
    
    @property
    def array_namespace(self):
        return xp
    
    @property
    def impl(self):
        return 'numpy'