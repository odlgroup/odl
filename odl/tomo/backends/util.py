__all__ = tuple()


def _add_default_complex_impl(fn):
    """Wrapper to call a class method twice when it is complex."""

    def wrapper(self, x, out, **kwargs):
        if self.reco_space.is_real and self.proj_space.is_real:
            fn(self, x, out, **kwargs)
            return out
        elif self.reco_space.is_complex and self.proj_space.is_complex:
            result_parts = [
                fn(x.real, getattr(out, 'real', None), **kwargs),
                fn(x.imag, getattr(out, 'imag', None), **kwargs)
            ]

            if out is None:
                out = range.element()
                out.real = result_parts[0]
                out.imag = result_parts[1]
            return out
        else:
            raise RuntimeError('Domain and range need to be both real '
                               'or both complex.')

    return wrapper
