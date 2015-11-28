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

"""Functions for graphical output."""


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()

# ODL
from odl.util.utility import is_real_dtype

# External module imports
from numbers import Integral
import numpy as np


__all__ = ('show_discrete_function',)


def show_discrete_function(dfunc, method='', title=None, indices=None,
                           **kwargs):
    """Display a discrete 1d or 2d function.

    Parameters
    ----------
    method : `str`, optional
        1d methods:

        'plot' : graph plot

        2d methods:

        'imshow' : image plot with coloring according to value,
        including a colorbar.

        'scatter' : cloud of scattered 3d points
        (3rd axis <-> value)

        'wireframe', 'plot_wireframe' : surface plot

    title : `str`, optional
        Set the title of the figure

    indices : index expression, optional
        Display a slice of the array instead of the full array. The
        index expression is most easily created with the `numpy.s_`
        constructur, i.e. supply ``np.s_[:, 1, :]`` to display the
        first slice along the second axis.

        For data with 3 or more dimensions, the 2d slice in the first
        two axes at the "middle" along the remaining axes is shown
        (semantically ``[:, :, shape[2:] // 2]``).

    kwargs : {'figsize', 'saveto', ...}
        Extra keyword arguments passed on to display method
        See the Matplotlib functions for documentation of extra
        options.

    See Also
    --------
    matplotlib.pyplot.plot : Show graph plot

    matplotlib.pyplot.imshow : Show data as image

    matplotlib.pyplot.scatter : Show scattered 3d points
    """
    # Importing pyplot takes ~2 sec, only import when needed.
    import matplotlib.pyplot as plt

    args_re = []
    args_im = []
    dsp_kwargs = {}
    sub_kwargs = {}
    arrange_subplots = (121, 122)  # horzontal arrangement

    # Default to showing x-y slice "in the middle"
    if indices is None and dfunc.ndim >= 3:
        indices = [np.s_[:]] * 2
        indices += [n // 2 for n in dfunc.space.grid.shape[2:]]

    if isinstance(indices, (Integral, slice)):
        indices = [indices]
    elif indices is None or indices == Ellipsis:
        indices = [np.s_[:]] * dfunc.ndim
    else:
        indices = list(indices)

    if Ellipsis in indices:
        # Replace Ellipsis with the correct number of [:] expressions
        pos = indices.index(Ellipsis)
        indices = (indices[:pos] +
                   [np.s_[:]] * (dfunc.ndim - len(indices) + 1) +
                   indices[pos + 1:])

    if len(indices) < dfunc.ndim:
        raise ValueError('too few axes ({} < {}).'.format(len(indices),
                                                          dfunc.ndim))
    if len(indices) > dfunc.ndim:
        raise ValueError('too many axes ({} > {}).'.format(len(indices),
                                                           dfunc.ndim))

    # Create axis labels which remember their original meaning
    if dfunc.ndim <= 3:
        axis_labels = ['x', 'y', 'z']
    else:
        axis_labels = ['x{}'.format(axis) for axis in range(dfunc.ndim)]
    squeezed_axes = [axis for axis in range(dfunc.ndim)
                     if not isinstance(indices[axis], Integral)]
    axis_labels = [axis_labels[axis] for axis in squeezed_axes]

    # Squeeze grid and values according to the index expression
    grid = dfunc.space.grid[indices].squeeze()
    values = dfunc.asarray()[indices].squeeze()

    dfunc_is_complex = not is_real_dtype(dfunc.space.dspace.dtype)
    figsize = kwargs.pop('figsize', None)
    saveto = kwargs.pop('saveto', None)

    if values.ndim == 1:  # TODO: maybe a plotter class would be better
        if not method:
            if dfunc.space.interp == 'nearest':
                method = 'step'
                dsp_kwargs['where'] = 'mid'
            elif dfunc.space.interp == 'linear':
                method = 'plot'
            else:
                method = 'plot'

        if method == 'plot' or method == 'step':
            args_re += [grid.coord_vectors[0], values.real]
            args_im += [grid.coord_vectors[0], values.imag]
        else:
            raise ValueError('display method {!r} not supported.'
                             ''.format(method))

    elif values.ndim == 2:
        if not method:
            method = 'imshow'

        if method == 'imshow':
            args_re = [np.rot90(values.real)]
            args_im = [np.rot90(values.imag)] if dfunc_is_complex else []

            extent = [grid.min()[0], grid.max()[0],
                      grid.min()[1], grid.max()[1]]

            if dfunc.space.interp == 'nearest':
                interpolation = 'nearest'
            elif dfunc.space.interp == 'linear':
                interpolation = 'bilinear'
            else:
                interpolation = 'none'

            dsp_kwargs.update({'interpolation': interpolation,
                               'cmap': 'bone',
                               'extent': extent,
                               'aspect': 'auto'})
        elif method == 'scatter':
            pts = grid.points()
            args_re = [pts[:, 0], pts[:, 1], values.ravel().real]
            args_im = ([pts[:, 0], pts[:, 1], values.ravel().imag]
                       if dfunc_is_complex else [])
            sub_kwargs.update({'projection': '3d'})
        elif method in ('wireframe', 'plot_wireframe'):
            method = 'plot_wireframe'
            xm, ym = grid.meshgrid()
            args_re = [xm, ym, np.rot90(values.real)]
            args_im = ([xm, ym, np.rot90(values.imag)] if dfunc_is_complex
                       else [])
            sub_kwargs.update({'projection': '3d'})
        else:
            raise ValueError('display method {!r} not supported.'
                             ''.format(method))

    else:
        raise NotImplementedError('no method for {}d display implemented.'
                                  ''.format(dfunc.ndim))

    # Additional keyword args are passed on to the display method
    dsp_kwargs.update(**kwargs)

    fig = plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)

    if dfunc_is_complex:
        sub_re = plt.subplot(arrange_subplots[0], **sub_kwargs)
        sub_re.set_title('Real part')
        sub_re.set_xlabel(axis_labels[0])
        if values.ndim == 2:
            sub_re.set_ylabel(axis_labels[1])
        else:
            sub_re.set_ylabel('value')
        display_re = getattr(sub_re, method)
        csub_re = display_re(*args_re, **dsp_kwargs)

        if method == 'imshow':
            minval_re = np.min(values.real)
            maxval_re = np.max(values.real)
            ticks_re = [minval_re, (maxval_re + minval_re) / 2.,
                        maxval_re]
            plt.colorbar(csub_re, orientation='horizontal',
                         ticks=ticks_re, format='%.4g')

        sub_im = plt.subplot(arrange_subplots[1], **sub_kwargs)
        sub_im.set_title('Imaginary part')
        sub_im.set_xlabel(axis_labels[0])
        if values.ndim == 2:
            sub_im.set_ylabel(axis_labels[1])
        else:
            sub_re.set_ylabel('value')
        display_im = getattr(sub_im, method)
        csub_im = display_im(*args_im, **dsp_kwargs)

        if method == 'imshow':
            minval_im = np.min(values.imag)
            maxval_im = np.max(values.imag)
            ticks_im = [minval_im, (maxval_im + minval_im) / 2.,
                        maxval_im]
            plt.colorbar(csub_im, orientation='horizontal',
                         ticks=ticks_im, format='%.4g')

    else:
        sub = plt.subplot(111, **sub_kwargs)
        sub.set_xlabel(axis_labels[0])
        if values.ndim == 2:
            sub.set_ylabel(axis_labels[1])
        else:
            sub.set_ylabel('value')
        try:
            # For 3d plots
            sub.set_zlabel('z')
        except AttributeError:
            pass
        display = getattr(sub, method)
        csub = display(*args_re, **dsp_kwargs)

        if method == 'imshow':
            minval = np.min(values)
            maxval = np.max(values)
            ticks = [minval, (maxval + minval) / 2., maxval]
            plt.colorbar(csub, ticks=ticks, format='%.4g')

    plt.show()
    if saveto is not None:
        fig.savefig(saveto)
