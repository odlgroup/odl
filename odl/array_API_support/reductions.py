# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

__all__ = (
    'sum',
    'prod',
    'min',
    'max',
)


def _apply_reduction(operation: str, x):
    return x.space._element_reduction(operation=operation, x=x)

def sum(x):
    return _apply_reduction('sum', x)

def prod(x):
    return _apply_reduction('prod', x)

def min(x):
    return _apply_reduction('min', x)

def max(x):
    return _apply_reduction('max', x)
