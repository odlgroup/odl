__all__ = ('vecdot',)

def vecdot(x1, x2, axis=-1, out = None):
    """Computes the (vector) dot product of two arrays."""
    assert x1.space.shape == x2.space.shape, f"The shapes of x1 {x1.space.shape} and x2 {x2.space.shape} differ, cannot perform vecdot"
    assert x1.space.device == x2.space.device, f"The devices of x1 {x1.space.device} and x2 {x2.space.device} differ, cannot perform vecdot"
    if out is not None:
        assert x1.space.shape == out.space.shape, f"The shapes of x1 {x1.space.shape} and out {out.space.shape} differ, cannot perform vecdot"
        assert x1.space.device == out.space.device, f"The devices of x1 {x1.space.device} and out {out.space.device} differ, cannot perform vecdot"
        out = out.data
        result = x1.array_namespace.linalg.vecdot(x1.data, x2.data, out=out)
    else:
        result = x1.array_namespace.linalg.vecdot(x1.data, x2.data)

    return result