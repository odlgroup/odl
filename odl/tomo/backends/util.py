__all__ = tuple()


def _add_default_complex_impl(fn):
    """Wrapper to call a class method twice when it is complex.

    This function helps `RayTransform` implementations by splitting
    complex-valued forward/backward function calls into two real-valued calls.

    Parameters
    ----------
    fn : Callable
        Function with signature ``fn(self, x, out=None, **kwargs)``.
        ``self`` must be an object instance having ``self.vol_space`` and
        ``self.proj_space``. These spaces must be both real or both complex.

    Returns
    -------
    Callable
        A wrapper function to be executed by a Python decorator.
    """

    def wrapper(self, x, out=None, **kwargs):
        if self.vol_space.is_real and self.proj_space.is_real:
            return fn(self, x, out, **kwargs)
        elif self.vol_space.is_complex and self.proj_space.is_complex:
            result_parts = [
                fn(self, x.real, getattr(out, 'real', None), **kwargs),
                fn(self, x.imag, getattr(out, 'imag', None), **kwargs)
            ]

            if out is None:
                out = self.proj_space.element()
                out.real = result_parts[0]
                out.imag = result_parts[1]
            return out
        else:
            raise RuntimeError('Domain and range need to be both real '
                               'or both complex.')

    return wrapper
