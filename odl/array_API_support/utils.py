__all__ = ('check_device',)

AVAILABLE_DEVICES = {
    'numpy' : ['cpu'],
    # 'pytorch' : ['cpu'] +  [f'cuda:{i}' for i in range(torch.cuda.device_count())]
}

def check_device(impl:str, device:str):
    """
    Checks the device argument 
    This checks that the device requested is available and that its compatible with the backend requested
    """
    assert device in AVAILABLE_DEVICES[impl], f"For {impl} Backend, devices {AVAILABLE_DEVICES[impl]} but {device} was provided."
    
if __name__ =='__main__':
    check_device('numpy', 'cpu')