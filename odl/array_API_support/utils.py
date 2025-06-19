__all__ = (
    'AVAILABLE_DEVICES',
    'IMPL_DEVICE_PAIRS',
    'check_device',)

AVAILABLE_DEVICES = {
    'numpy' : ['cpu'],
    # 'pytorch' : ['cpu'] +  [f'cuda:{i}' for i in range(torch.cuda.device_count())]
}

IMPL_DEVICE_PAIRS = []
for impl in AVAILABLE_DEVICES.keys():
    for device in AVAILABLE_DEVICES[impl]:
        IMPL_DEVICE_PAIRS.append((impl, device))

def check_device(impl:str, device:str):
    """
    Checks the device argument 
    This checks that the device requested is available and that its compatible with the backend requested
    """
    assert device in AVAILABLE_DEVICES[impl], f"For {impl} Backend, devices {AVAILABLE_DEVICES[impl]} but {device} was provided."
    
if __name__ =='__main__':
    check_device('numpy', 'cpu')