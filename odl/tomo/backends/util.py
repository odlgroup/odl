__all__ = ('complexify',)


def complexify(fn, range):
    """Wrapper to call a function twice when it is complex"""

    if not range.is_real and not range.is_complex:
        raise RuntimeError('Range needs to be real or complex{!r}'
                           .format(range))

    # No need to do anything when the range is real
    if range.is_real:
        return fn

    # Function takes a possibly complex `x` and delivers complex `out`
    def complex_fn(x, out, **kwargs):
        result_parts = [
            complex_fn(x.real, getattr(out, 'real', None), **kwargs),
            complex_fn(x.imag, getattr(out, 'imag', None), **kwargs)]

        if out is None:
            out = range.element()
            out.real = result_parts[0]
            out.imag = result_parts[1]

        return out

    return complex_fn
