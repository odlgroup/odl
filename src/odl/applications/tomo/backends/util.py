# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for tomo backends."""

from functools import wraps

__all__ = ()


def _add_default_complex_impl(fn):
    """Wrapper to call a real-valued class method twice when input is complex.

    This function helps `RayTransform` implementations with extending methods
    that work for real-valued elements to complex-valued elements, by splitting
    complex calls into two individual real calls.

    The wrapper will only work for methods of which the class provides a
    `vol_space` and `proj_space`. Typically, this will then work as a decorator
    on the method, e.g. ::

        @_add_default_complex_impl
        def call_forward(self, x, out=None, **kwargs):
            # Code that should run for real input and output

    Parameters
    ----------
    fn : Callable
        Function with signature ``fn(self, x, out=None, **kwargs)``.
        ``self`` must be an object instance having ``self.vol_space`` and
        ``self.proj_space``.

    Returns
    -------
    Callable
        A wrapper function to be executed by a Python decorator.
    """

    @wraps(fn)
    def wrapper(self, x, out=None, **kwargs):
        if self.vol_space.is_real and self.proj_space.is_real:
            return fn(self, x, out, **kwargs)
        elif self.vol_space.is_complex and self.proj_space.is_complex:
            if out is None:
                if x in self.vol_space:
                    range = self.proj_space
                else:
                    range = self.vol_space

                out = range.zero()
            
            fn(self, x.real, out.real, **kwargs)
            fn(self, x.imag, out.imag, **kwargs)

            return out
        else:
            raise RuntimeError(
                'domain and range need to be both real or both complex'
            )

    return wrapper
