# Copyright 2014, 2015 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
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

import warnings
import numpy as np
from itertools import product

from odl.set.sets import RealNumbers, ComplexNumbers
from odl.set.space import LinearSpace
from odl.set.pspace import ProductSpace
from odl.space.base_ntuples import FnBase
from odl.discr.lp_discr import DiscreteL2


__all__ = ('scalar_examples', 'vector_examples', 'samples')


def _arg_shape(*args):
    if len(args) == 1:
        return args[0].shape
    else:
        return np.broadcast(*args).shape


def scalar_examples(field):
    if field == RealNumbers():
        return [-1.0, 0.5, 0.0, 0.01, 1.0]
    if field == ComplexNumbers():
        return [-1.0, 0.5, 0.0+2.0j, 0.0, 0.01, 1.0 + 1.0j, 1.0j, 1.0]


def vector_examples(space):
    # All spaces should yield the zero element
    yield ('Zero', space.zero())

    if isinstance(space, ProductSpace):
        for examples in product(*[vector_examples(spc) for spc in space]):
            name = ', '.join(name for name, _ in examples)
            vector = space.element([vec for _, vec in examples])
            yield (name, space.element(vector))

    elif isinstance(space, DiscreteL2):
        uspace = space.uspace

        # Get the points and calculate some statistics on them
        points = space.points()
        mins = space.grid.min()
        maxs = space.grid.max()
        means = (maxs+mins)/2.0
        stds = np.apply_along_axis(np.std, axis=0, arr=points)

        # Indicator function in first dimension
        def _step_fun(*args):
            z = np.zeros(_arg_shape(*args))
            z[:space.grid.shape[0] // 2, ...] = 1
            return z

        yield ('Step', space.element(uspace.element(_step_fun)))

        # Indicator function on hypercube
        def _cube_fun(*args):
            inside = np.ones(_arg_shape(*args), dtype=bool)
            for points, mean, std in zip(args, means, stds):
                inside = np.logical_and(inside, points < mean+std)
                inside = np.logical_and(inside, mean-std < points)

            return inside.astype(space.dtype)

        yield ('Cube', space.element(uspace.element(_cube_fun)))

        # Indicator function on hypersphere
        if space.grid.ndim > 1:  # Only if ndim > 1, don't duplicate cube
            def _sphere_fun(*args):
                r = np.zeros(_arg_shape(*args))

                for points, mean, std in zip(args, means, stds):
                    r += (points - mean)**2 / std**2
                return (r < 1.0).astype(space.dtype)

            yield ('Sphere', space.element(uspace.element(_sphere_fun)))

        # Gaussian function
        def _gaussian_fun(*args):
            r2 = np.zeros(_arg_shape(*args))

            for points, mean, std in zip(args, means, stds):
                r2 += (points - mean)**2 / ((std/2)**2)

            return np.exp(-r2)

        yield ('Gaussian', space.element(uspace.element(_gaussian_fun)))

        # Gradient in each dimensions
        for dim in range(space.grid.ndim):
            def _gradient_fun(*args):
                s = np.zeros(_arg_shape(*args))
                s += (args[dim]-mins[dim]) / (maxs[dim]-mins[dim])

                return s

            yield ('grad {}'.format(dim),
                   space.element(uspace.element(_gradient_fun)))

        # Gradient in all dimensions
        if space.grid.ndim > 1:  # Only if ndim > 1, don't duplicate grad 0
            def _all_gradient_fun(*args):
                s = np.zeros(_arg_shape(*args))

                for points, minv, maxv in zip(args, mins, maxs):
                    s += (points - minv) / (maxv-minv)

                return s

            yield ('Grad all', space.element(uspace.element(_all_gradient_fun)))

    elif isinstance(space, FnBase):
        rand_state = np.random.get_state()
        np.random.seed(1337)

        yield ('Linspaced', space.element(np.linspace(0, 1, space.size)))

        yield ('Ones', space.element(np.ones(space.size)))

        yield ('Random noise', space.element(np.random.rand(space.size)))

        yield ('Normally distributed random noise',
               space.element(np.random.randn(space.size)))

        np.random.set_state(rand_state)
    else:
        warnings.warn('No known examples in this space')


def samples(*sets):
    if len(sets) == 1:
        if isinstance(sets[0], LinearSpace):
            for vec in vector_examples(sets[0]):
                yield vec
        else:
            for scal in scalar_examples(sets[0]):
                yield scal
    else:
        generators = [vector_examples(set) if isinstance(set, LinearSpace)
                      else scalar_examples(set) for set in sets]
        for examples in product(*generators):
            yield examples

if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
