# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify odl
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Functions for generating standardized examples in spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range, zip

import warnings
import numpy as np
from itertools import product

from odl.set.sets import RealNumbers, ComplexNumbers
from odl.set.space import LinearSpace
from odl.set.pspace import ProductSpace
from odl.space.base_ntuples import FnBase
from odl.discr.lp_discr import DiscreteLp


__all__ = ('scalar_examples', 'vector_examples', 'samples')


def scalar_examples(field):
    """Generate example scalars in ``field``.

    Parameters
    ----------
    field : `Field`
        The field to generate examples from

    Returns
    -------
    examples : `generator`
        Yields elements in ``field``
    """
    if field == RealNumbers():
        return [-1.0, 0.5, 0.0, 0.01, 1.0]
    elif field == ComplexNumbers():
        return [-1.0, 0.5, 0.0 + 2.0j, 0.0, 0.01, 1.0 + 1.0j, 1.0j, 1.0]
    else:
        raise NotImplementedError('field not supported')


def vector_examples(space):
    """Generate example vectors in ``space``.

    Parameters
    ----------
    space : `LinearSpace`
        The space to generate examples from

    Returns
    -------
    examples : `generator`
        Yields tuples (`string`, `LinearSpaceVector`)
        where ``string`` is a short description of the vector
    """

    if isinstance(space, ProductSpace):
        for examples in product(*[vector_examples(spc) for spc in space]):
            name = ', '.join(name for name, _ in examples)
            vector = space.element([vec for _, vec in examples])
            yield (name, space.element(vector))
        return

    # All spaces should yield the zero element
    yield ('Zero', space.zero())

    try:
        yield ('One', space.one())
    except NotImplementedError:
        pass

    if isinstance(space, DiscreteLp):
        # Get the points and calculate some statistics on them
        points = space.points()
        mins = space.grid.min()
        maxs = space.grid.max()
        means = (maxs + mins) / 2.0
        stds = np.apply_along_axis(np.std, axis=0, arr=points)

        def element(fun):
            return space.element(fun)

        # Indicator function in first dimension
        def _step_fun(x):
            z = np.zeros(space.shape, dtype=space.dtype)
            z[:space.shape[0] // 2, ...] = 1
            return z

        yield ('Step', element(_step_fun))

        # Indicator function on hypercube
        def _cube_fun(x):
            inside = np.ones(space.shape, dtype=bool)
            for points, mean, std in zip(x, means, stds):
                inside = np.logical_and(inside, points < mean + std)
                inside = np.logical_and(inside, mean - std < points)

            return inside.astype(space.dtype, copy=False)

        yield ('Cube', element(_cube_fun))

        # Indicator function on hypersphere
        if space.ndim > 1:  # Only if ndim > 1, don't duplicate cube
            def _sphere_fun(x):
                r = np.zeros(space.shape)

                for points, mean, std in zip(x, means, stds):
                    r += (points - mean) ** 2 / std ** 2
                return (r < 1.0).astype(space.dtype, copy=False)

            yield ('Sphere', element(_sphere_fun))

        # Gaussian function
        def _gaussian_fun(x):
            r2 = np.zeros(space.shape)

            for points, mean, std in zip(x, means, stds):
                r2 += (points - mean) ** 2 / ((std / 2) ** 2)

            return np.exp(-r2)

        yield ('Gaussian', element(_gaussian_fun))

        # Gradient in each dimensions
        for dim in range(space.ndim):
            def _gradient_fun(x):
                s = np.zeros(space.shape)
                s += (x[dim] - mins[dim]) / (maxs[dim] - mins[dim])

                return s

            yield ('grad {}'.format(dim),
                   element(_gradient_fun))

        # Gradient in all dimensions
        if space.ndim > 1:  # Only if ndim > 1, don't duplicate grad 0
            def _all_gradient_fun(x):
                s = np.zeros(space.shape)

                for points, minv, maxv in zip(x, mins, maxs):
                    s += (points - minv) / (maxv - minv)

                return s

            yield ('Grad all', element(_all_gradient_fun))

    elif isinstance(space, FnBase):
        rand_state = np.random.get_state()
        np.random.seed(1337)

        yield ('Linspaced', space.element(np.linspace(0, 1, space.size)))

        yield ('Random noise', space.element(np.random.rand(space.size)))

        yield ('Normally distributed random noise',
               space.element(np.random.randn(space.size)))

        np.random.set_state(rand_state)
    else:
        warnings.warn('No known examples in this space')


def samples(*sets):
    """Generate some samples from the given sets.

    Currently supports vectors according to `vector_examples`
    and scalars according to `scalar_examples`.

    Parameters
    ----------
    *sets : `Set` instance(s)

    Returns
    -------
    samples : `generator`
        Generator that yields tuples of examples from the sets.
    """
    if len(sets) == 1:
        if isinstance(sets[0], LinearSpace):
            for vec in vector_examples(sets[0]):
                yield vec
        else:
            for scal in scalar_examples(sets[0]):
                yield scal
    else:
        generators = [vector_examples(set_) if isinstance(set_, LinearSpace)
                      else scalar_examples(set_) for set_ in sets]
        for examples in product(*generators):
            yield examples

if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
