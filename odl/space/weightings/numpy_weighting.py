from .base_weighting import Weighting

import array_api_compat.numpy as xp

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