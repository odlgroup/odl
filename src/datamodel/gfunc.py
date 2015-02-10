# -*- coding: utf-8 -*-
"""
gfunc.py -- n-dimensional functions on a grid

Copyright 2014, 2015 Holger Kohr

This file is part of RL.

RL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RL.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import object
from builtins import int
from builtins import zip
from builtins import range
from builtins import super
from future import standard_library
standard_library.install_aliases()

import numpy as np
from scipy.interpolate import interpn
from copy import deepcopy

from RL.datamodel.ugrid import Ugrid
from RL.datamodel.coord import Coord
from RL.utility.utility import flat_tuple, errfmt, InputValidationError


def asgfunc(obj):
    """Convert `obj` to a ´Gfunc` obj. If type(obj)==ugrid.Ugrid, use
    empty fvals. If type(obj)==numpy.ndarray, assume standard grid.
    """

    if isinstance(obj, Gfunc):
        return obj
    elif isinstance(obj, np.ndarray):
        return Gfunc(obj)
    elif isinstance(obj, Ugrid):
        return Gfunc(None, obj.shape, obj.center, obj.spacing)
    else:  # TODO: think about other cases
        raise TypeError("{} not convertible to Gfunc".format(type(obj)))


def frommapping(grid, mapping):
    """Initialize a `gfunc` from a `mapping` on a `grid`. The `mapping`
    needs to accept a (grid.ntotal, grid.dim) array or (grid.dim) arrays
    (grid.ntotal,) and must map it to a (grid.ntotal,) array."""
    gfun = asgfunc(grid)
    gfun.fvals = mapping(gfun.coord.asarr())
    return gfun


class Gfunc(Ugrid):
    """Grid function class. For initialization, `fvals` or `shape` are
    required. If `shape` is None, the shape will be determined from
    `fvals`. If both are given, `fvals` must be broadcastable to an array
    with `shape`.
    TODO: write properly
    """

    # TODO: use variable args and determine number and types
    def __init__(self, fvals=None, shape=None, center=None, spacing=None):

        if shape is None and fvals is None:
            raise ValueError("Either `shape` or `fvals` must be specified.")

        if shape is None:
            super().__init__(fvals.shape, center, spacing)
            self._fvals[...] = np.asarray(fvals)
        else:
            super().__init__(shape, center, spacing)
            self._fvals = np.empty(shape)

            # Reshape or broadcast fvals
            # TODO: deal with datatype mismatch
            if fvals is not None:
                try:
                    self._fvals = np.asarray(fvals).reshape(shape)
                except ValueError:
                    self._fvals[...] = np.asarray(fvals)

    # Essential properties

    @property
    def fvals(self):
        return self._fvals

    @fvals.setter
    def fvals(self, new_fvals):
        """Set the fvals array. `None` means an empty array. Otherwise, the
        argument must be broadcastable to the object's `shape`."""
        if new_fvals is None:
            self._fvals = np.empty(self.shape)
        else:
            # See __init__
            try:
                self._fvals = np.asarray(new_fvals).reshape(self.shape)
            except ValueError:  # a single value given
                self._fvals[...] = np.asarray(new_fvals)

    # Derived properties

    # Magic methods

    def __call__(self, vec):
        # TODO: Cythonize
        vec = np.asarray(vec)
        if vec.shape != (self.dim,):
            raise InputValidationError('vec.shape', (self.dim,))
        if np.any(vec < self.xmin) or np.any(vec > self.xmax):
            raise LookupError("Vector outside grid.")

        vec_ind_flt = (vec - self.xmin) / self.spacing
        vec_ind = vec_ind_flt.astype(int)
        weight_u = vec_ind_flt - vec_ind
        weight_l = 1. - weight_u

        # Cut out the interpolation cell
        cell_slc = [np.s_[vec_ind[i]:vec_ind[i] + 2] for i in range(self.dim)]
        cell = self.fvals[cell_slc]

        # Due to reduction, we can always slice along the first axis
        slc_l = [0] + [np.s_[:]] * (self.dim - 1)
        slc_r = [1] + [np.s_[:]] * (self.dim - 1)
        for axis in range(self.dim):
            cell[slc_l] *= weight_l[axis]
            cell[slc_r] *= weight_u[axis]
            cell = np.sum(cell, axis=0)
            slc_l.pop()
            slc_r.pop()

        return cell

    def __getitem__(self, slc):
        if isinstance(slc, slice):
            slc = np.index_exp[slc]
        else:
            slc = flat_tuple(slc)

        newct = []
        newsp = []

        for i in range(self.dim):
            if hasattr(slc[i], 'start'):
                iarr = np.arange(self.shape[i])[slc[i]]
                sta, sto = iarr[0], iarr[-1]
                imid = (sta + sto) / 2.
                newct.append(self.xmin[i] + imid * self.spacing[i])

                ste = slc[i].step if slc[i].step else 1
                newsp.append(ste * self.spacing[i])

        return Gfunc(self.fvals[slc], None, newct, newsp)

    def __setitem__(self, slc, other):
        raise NotImplementedError  # TODO: do

    def __eq__(self, other):
        # FIXME: this compares Ugrid with Gfunc - avoid!
        return super().__eq__(other) and np.all(self.fvals == other.fvals)

    # TODO: implement this properly; right now it works for
    # type(other)==numpy.ndarray
    def __mul__(self, other):
        self_copy = self.copy()
        self_copy.fvals = self.fvals.__mul__(other)  # TODO: reduce overhead
        return self_copy

    def __rmul__(self, other):
        self_copy = self.copy()
        self_copy.fvals = self.fvals.__rmul__(other)  # TODO: reduce overhead
        return self_copy

    def __imul__(self, other):
        self.fvals = self.fvals.__imul__(other)
        return self

    # Public methods

    def copy(self):
        """Return a (deep) copy."""
        return deepcopy(self)

    def asgraph(self):  # TODO: necessary?
        coord_arr = self.coord.asarr()
        return coord_arr, self.fvals.flatten()

    def display(self, method='', figsize=None, saveto='', **kwargs):
        """For dim in (1,2) make a graph plot. No generic way otherwise.
        """
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        args_re = []
        args_im = []
        dsp_kwargs = {}
        sub_kwargs = {}
        arrange_subplots = (121, 122)  # horzontal arrangement

        if self.dim == 1:  # TODO: maybe a plotter class would be better
            if not method:
                method = 'plot'

            if method == 'plot':
                args_re += [self.coord.vecs[0], self.fvals.real]
                args_im += [self.coord.vecs[0], self.fvals.imag]
            else:
                raise ValueError("""Unknown display method '{}'
                """.format(method))

        elif self.dim == 2:
            if not method:
                method = 'imshow'

            if method == 'imshow':
                from matplotlib.cm import gray
                args_re = [self.fvals.real.T]
                args_im = [self.fvals.imag.T]
                extent = [self.xmin[0], self.xmax[0],
                          self.xmin[1], self.xmax[1]]
                aspect = self.tsize[1] / self.tsize[0]
                dsp_kwargs.update({'interpolation': 'none', 'cmap': gray,
                                   'extent': extent, 'aspect': aspect})
            elif method == 'scatter':
                coo_arr = self.coord.asarr()
                args_re = [coo_arr[:, 0], coo_arr[:, 1], self.fvals.real]
                args_im = [coo_arr[:, 0], coo_arr[:, 1], self.fvals.imag]
                sub_kwargs.update({'projection': '3d'})
            elif method in ('wireframe', 'plot_wireframe'):
                method = 'plot_wireframe'
                xm, ym = np.meshgrid(self.coord.vecs[0], self.coord.vecs[1],
                                     sparse=True, indexing='ij')
                args_re = [xm, ym, self.fvals.real]
                args_im = [xm, ym, self.fvals.imag]
                sub_kwargs.update({'projection': '3d'})
            else:
                raise ValueError("""Unknown display method '{}'
                """.format(method))

        else:
            print("""No generic way to display {}D data, sorry.
            """.format(self.dim))
            return

        # Additional keyword args are passed on to the display method
        dsp_kwargs.update(**kwargs)

        fig = plt.figure(figsize=figsize)
        if np.any(np.iscomplex(self.fvals)):
            sub_re = plt.subplot(arrange_subplots[0], **sub_kwargs)
            sub_re.set_title('Real part')
            sub_re.set_xlabel('x')
            sub_re.set_ylabel('y')
            display_re = getattr(sub_re, method)
            csub_re = display_re(*args_re, **dsp_kwargs)

            if method == 'imshow':
                minval_re = np.min(self.fvals.real)
                maxval_re = np.max(self.fvals.real)
                ticks_re = [minval_re, (maxval_re + minval_re) / 2.,
                            maxval_re]
                cbar_re = plt.colorbar(csub_re, orientation='horizontal',
                                       ticks=ticks_re, format='%.4g')

            sub_im = plt.subplot(arrange_subplots[1], **sub_kwargs)
            sub_im.set_title('Imaginary part')
            sub_im.set_xlabel('x')
            sub_im.set_ylabel('y')
            display_im = getattr(sub_im, method)
            csub_im = display_im(*args_im, **dsp_kwargs)

            if method == 'imshow':
                minval_im = np.min(self.fvals.imag)
                maxval_im = np.max(self.fvals.imag)
                ticks_im = [minval_im, (maxval_im + minval_im) / 2.,
                            maxval_im]
                cbar_im = plt.colorbar(csub_im, orientation='horizontal',
                                       ticks=ticks_im, format='%.4g')

        else:
            sub = plt.subplot(111, **sub_kwargs)
            sub.set_xlabel('x')
            sub.set_ylabel('y')
            try:
                sub.set_zlabel('z')
            except AttributeError:
                pass
            display = getattr(sub, method)
            csub = display(*args_re, **dsp_kwargs)

            if method == 'imshow':
                minval = np.min(self.fvals)
                maxval = np.max(self.fvals)
                ticks = [minval, (maxval + minval) / 2., maxval]
                cbar = plt.colorbar(csub, ticks=ticks, format='%.4g')

        plt.show()
        if saveto:
            fig.savefig(saveto)

    def interpolate(self, *vecs, **kwargs):
        """Interpolate the function at given points. The points are either
        provided as a large array with each row consisting of one point
        or as a tuple of vectors each defining one coordinate of the
        interpolation points. If `as_grid` is True, each combination of
        coordinates will be used (each point on the grid defined by the
        coordinates). Additional keyword args are passed to
        scipy.interpolate.interpn.
        TODO: write up properly"""

        as_grid = kwargs.get('as_grid', False)
        if as_grid:
            coo = Coord(*vecs)
            vecs = coo.asarr()

        method = kwargs.get('method', 'linear')
        bounds_error = kwargs.get('bounds_error', False)
        fill_value = kwargs.get('fill_value', 0.0)

        return interpn(self.coord.vecs, self.fvals, vecs, method=method,
                       bounds_error=bounds_error, fill_value=fill_value)

    def read(self, source):
        raise NotImplementedError  # TODO: do

    def write(self, dest):
        raise NotImplementedError  # TODO: do

    def downsample(self, factor, allow_interpolation=True):
        # FIXME: think again about the math, currently broken!
        # This is actually a problem of basis change. Think properly in this
        # way!
        raise NotImplementedError
        # TODO: this function needs some decent testing
        fac_cp = factor
        try:
            factor[0]
        except TypeError:
            factor = (factor,) * self.dim
        factor = np.asarray(factor)
        if not np.all(factor > 0):
            raise ValueError("factor: got {!s}, expected > 0.".format(fac_cp))
        if np.any(factor != factor.astype(int)) and not allow_interpolation:
            raise ValueError(errfmt("""\
            Downsampling by non-integer factor not possible without
            interpolation"""))

        if np.any(factor != factor.astype(int)):
            if not allow_interpolation:
                raise RuntimeError(errfmt('''\
                Interpolation not allowed but required for non-integer
                downsampling.'''))
            # Interpolate if one of the factors is non-integer
            # TODO: more efficient to only interpolate along the dimensions
            # where it's really necessary
            newshape = (self.shape / factor).astype(int)
            newspacing = self.spacing * factor
            new_grid = Ugrid(newshape, self.center, newspacing)
            self._fvals = self.interpolate(new_grid.coord.vecs,
                                          as_grid=True).reshape(newshape)
            self._shape = newshape
            self._spacing = newspacing
            self._update_coord()

        else:
            slc_lst = [np.s_[:]] * self.dim

            for axis in range(self.dim):
                # Depending on old shape and factor, the max/min of the
                # new grid will differ from the old ones. If the new ones
                # lie between the old grid points, the whole grid will be
                # off by a fraction a cell, so interpolation is necessary
                fact = factor[axis]
                oldshape = self.shape[axis]
                newshape = int(oldshape / fact)
                scale_rem = oldshape - fact * newshape
                old_odd = oldshape % 2
                new_odd = newshape % 2

                print('fact (k): ', fact)
                print('oldshape (N): ', oldshape)
                print('newshape (M): ', newshape)
                print('scale_rem (t): ', scale_rem)
                print('old_odd (r): ', old_odd)
                print('new_odd (s): ', new_odd)

                # FIXME: this is wrong!
                # oldmax - newmax in old cell units -> boundary distance
                bound_dist = (scale_rem + fact * (new_odd + 1) -
                              (old_odd + 1)) / 2
                print('bound_dist: ', bound_dist)
    #            bound_dist = (fact + oldshape % stride - 1.) / 2.
                last_out = int(bound_dist)
                new_off = bound_dist - int(bound_dist)
                print('last_out: ', last_out)
                print('new_off: ', new_off)

                must_interpolate = (new_off != 0 or fact != int(fact))
                if must_interpolate:
                    if not allow_interpolation:
                        raise RuntimeError(errfmt('''\
                        Interpolation not allowed but required in this case.
                        '''))
                    # Every new point has the same shift with respect to the
                    # old grid, therefore all interpolation weights are the
                    # same.
                    slc_lst_l = slc_lst[:]
                    slc_lst_r = slc_lst[:]

                    if last_out == 0:
                        slc_lst_l[axis] = np.s_[: -1]
                        slc_lst_r[axis] = np.s_[1:]
                    elif last_out == 1:
                        slc_lst_l[axis] = np.s_[1:-1]
                        slc_lst_r[axis] = np.s_[2:]
                    else:
                        slc_lst_l[axis] = np.s_[last_out:-last_out]
                        slc_lst_r[axis] = np.s_[last_out + 1:-last_out + 1]

                    print('left slice: ', slc_lst_l[axis])
                    print('right slice: ', slc_lst_r[axis])
                    self._fvals = ((1 - new_off) * self._fvals[slc_lst_l] +
                                  new_off * self._fvals[slc_lst_r])
                    print('intermediate shape: ', self._fvals.shape)
                    # Now the actual downsampling
                    down_lst = slc_lst[:]
                    down_lst[axis] = np.s_[::int(fact)]
                    self._fvals = self._fvals[down_lst]
                else:
                    first_in = int(bound_dist)
                    down_lst = slc_lst[:]
                    down_lst[axis] = np.s_[first_in:-first_in:int(fact)]
                    self._fvals = self._fvals[down_lst]

                self._shape[axis] = newshape
                self._spacing[axis] *= fact
                self._update_coord()

    def upsample(self, factor, allow_interpolation=True):
        try:
            factor[0]
        except TypeError:
            factor = (factor,) * self.dim

        if not np.all(np.asarray(factor) > 0):
            raise ValueError("factor: got {!s}, expected > 0.".format(factor))

#        for axis in range(self.dim):
#            fact = factor[axis]
#            newshape = int(fact) * self.shape[axis]
#            bound_dist = (scale_rem + fact * (new_odd + 1) -
#                          (old_odd + 1)) / 2

        raise NotImplementedError

    # TODO: the following functions can probably be combined into fewer
    # or even a single one
    def apply_ndmapping(self, mapping):
        raise NotImplementedError

    def apply_ndmapping_with_slice(self, mapping, slc):
        raise NotImplementedError

    def apply_0dmapping(self, mapping):
        """Apply a 0d `mapping` on the function values.
        TODO: good description"""

        self.fvals = mapping(self.fvals)

    def apply_0dmapping_with_1dslice(self, mapping, axis, ax_slc):
        """Apply a 0d `mapping` on a slice along `axis`.
        TODO: good description"""

        slc = [np.s_[:]] * self.dim
        slc[axis] = ax_slc
        self.fvals[slc] = mapping(self.fvals[slc])

    def apply_0dmapping_with_ndslice(self, mapping, slc):
        """Apply a 0d `mapping` on a slice along `axis`.
        TODO: good description"""

        self.fvals[slc] = mapping(self.fvals[slc])

    def apply_1dmapping(self, mapping, axis):
        """Apply a 1d `mapping` depending on the coorinate along `axis`.
        TODO: good description"""

        # Alternative: blow up 1d array with np.newaxis
        ax_coo = self.coord.vecs[axis]
        slc = [np.s_[:]] * self.dim

        for i in range(self.shape[axis]):
            slc[axis] = i
            self.fvals[slc] = mapping(ax_coo[i], self.fvals[slc])

    def apply_1dmapping_with_slice(self, mapping, axis, ax_slice):
        """Apply a 1d `mapping` depending on the coorinate along `axis`.
        TODO: good description"""

        ax_coo = self.coord.vecs[axis][ax_slice]
        ax_idc = np.arange(self.shape[axis])[ax_slice]
        slc = [np.s_[:]] * self.dim

        for i, x in zip(ax_idc, ax_coo):
            slc[axis] = i
            self.fvals[slc] = mapping(x, self.fvals[slc])


class GraphTransform(object):
    """Base class for function graph transforms.
    `mapping`: mapping(coordinate_array, values) -> tr_values
    The transform is executed by calling the class object and returns the
    transformed vector array.
    Subclasses can customize the initialization of `mapping` for special
    cases.
    TODO: write up properly
    """

    def __init__(self, mapping):
        self.mapping = mapping  # TODO: check signature?

    def __call__(self, *args):
        try:
            # input x, y, f; mapping(x, y, f) or input arr, f; mapping(arr, f)
            return self.mapping(*args)
        except TypeError:
            pass
        try:
            # input arr, f; mapping(x, y, f)
            arg_lst = [col for col in args[0].T] + [args[1]]
            return self.mapping(*arg_lst)
        except IndexError:
            # NOTE: forth case never happens
            raise ValueError("Wrong mapping type.")


class GraphTransformMultiply(GraphTransform):
    """Multiply with a function defined by `multiplier`. This is either a
    constant or accepts either an array (column-wise) or a list of coordinate
    vectors as arguments and return a column of the same length (one value
    for each row).
    The transform is executed by calling the class object and returns the
    transformed values.
    """
    def __init__(self, multiplier):

        def mapping(coord_arr, fvals):
            try:
                # TODO: check: does this overwrite?
                factor = float(multiplier)
                fvals *= factor
                return fvals
            except TypeError:
                pass
            try:
                fvals *= multiplier(coord_arr)
            except TypeError:
                fvals *= multiplier(*coord_arr.T)

            return fvals

        self.mapping = mapping
