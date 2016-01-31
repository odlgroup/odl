# Copyright 2014-2016 The ODL development group
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

# External
import numpy as np

# ODL
from odl.util.utility import is_real_dtype


__all__ = ('show_discrete_data',)


def show_discrete_data(values, grid, method='', title=None,
                       show=False, fig=None, **kwargs):
    """Display a discrete 1d or 2d function.

    Parameters
    ----------
    values : `numpy.ndarray`
        The values to visualize
    grid : `RegularGrid`
        Grid of the values
    method : `str`, optional
        1d methods:

        'plot' : graph plot

        'scatter' : scattered 2d points
        (2nd axis <-> value)

        2d methods:

        'imshow' : image plot with coloring according to value,
        including a colorbar.

        'scatter' : cloud of scattered 3d points
        (3rd axis <-> value)

        'wireframe', 'plot_wireframe' : surface plot

    title : `str`, optional
        Set the title of the figure

    show : `bool`, optional
        If the plot should be showed now or deferred until later

    fig : ``matplotlib`` figure
        The figure to show in. Expected to be of same "style", as the figure
        given by this function. The most common usecase is that fig is the
        return value from an earlier call to this function.

    interp : {'nearest', 'linear'}
        Interpolation method to use.

    axis_labels : `str`
        Axis labels, default: ['x', 'y']

    kwargs : {'figsize', 'saveto', ...}
        Extra keyword arguments passed on to display method
        See the Matplotlib functions for documentation of extra
        options.

    Returns
    -------
    fig : ``matplotlib`` figure
        The resulting figure. It is also shown to the user.
    colorbar : ``matplotlib`` colorbar
        The colorbar

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

    # Create axis labels which remember their original meaning
    axis_labels = kwargs.pop('axis_labels', ['x', 'y'])

    values_are_complex = not is_real_dtype(values.dtype)
    figsize = kwargs.pop('figsize', None)
    saveto = kwargs.pop('saveto', None)
    interp = kwargs.pop('interp', 'nearest')

    if values.ndim == 1:  # TODO: maybe a plotter class would be better
        if not method:
            if interp == 'nearest':
                method = 'step'
                dsp_kwargs['where'] = 'mid'
            elif interp == 'linear':
                method = 'plot'
            else:
                method = 'plot'

        if method == 'plot' or method == 'step' or method == 'scatter':
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
            args_im = [np.rot90(values.imag)] if values_are_complex else []

            extent = [grid.min()[0], grid.max()[0],
                      grid.min()[1], grid.max()[1]]

            if interp == 'nearest':
                interpolation = 'nearest'
            elif interp == 'linear':
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
                       if values_are_complex else [])
            sub_kwargs.update({'projection': '3d'})
        elif method in ('wireframe', 'plot_wireframe'):
            method = 'plot_wireframe'
            x, y = grid.meshgrid()
            args_re = [x, y, np.rot90(values.real)]
            args_im = ([x, y, np.rot90(values.imag)] if values_are_complex
                       else [])
            sub_kwargs.update({'projection': '3d'})
        else:
            raise ValueError('display method {!r} not supported.'
                             ''.format(method))

    else:
        raise NotImplementedError('no method for {}d display implemented.'
                                  ''.format(values.ndim))

    # Additional keyword args are passed on to the display method
    dsp_kwargs.update(**kwargs)

    if fig is not None:
        # Reuse figure if given as input
        if not isinstance(fig, plt.Figure):
            raise TypeError('fig {} not a matplotlib figure'.format(fig))

        if not plt.fignum_exists(fig.number):
            # If figure does not exist, user either closed the figure or
            # is using IPython, in this case we need a new figure.

            fig = plt.figure(figsize=figsize)
            updatefig = False
        else:
            # Set current figure to given input
            plt.figure(fig.number)
            updatefig = True

            if values.ndim > 1:
                # If the figure is larger than 1d, we can clear it since we
                # dont reuse anything. Keeping it causes performance problems.
                fig.clf()
    else:
        fig = plt.figure(figsize=figsize)
        updatefig = False

    if values_are_complex:
        # Real
        if len(fig.axes) == 0:
            # Create new axis if needed
            sub_re = plt.subplot(arrange_subplots[0], **sub_kwargs)
            sub_re.set_title('Real part')
            sub_re.set_xlabel(axis_labels[0])
            if values.ndim == 2:
                sub_re.set_ylabel(axis_labels[1])
            else:
                sub_re.set_ylabel('value')
        else:
            sub_re = fig.axes[0]

        display_re = getattr(sub_re, method)
        csub_re = display_re(*args_re, **dsp_kwargs)

        if method == 'imshow' and len(fig.axes) < 2:
            # Create colorbar if none seems to exist
            minval_re = np.min(values.real)
            maxval_re = np.max(values.real)
            ticks_re = [minval_re, (maxval_re + minval_re) / 2., maxval_re]

            plt.colorbar(csub_re, orientation='horizontal',
                         ticks=ticks_re, format='%.4g')

        # Imaginary
        if len(fig.axes) < 3:
            sub_im = plt.subplot(arrange_subplots[1], **sub_kwargs)
            sub_im.set_title('Imaginary part')
            sub_im.set_xlabel(axis_labels[0])
            if values.ndim == 2:
                sub_im.set_ylabel(axis_labels[1])
            else:
                sub_im.set_ylabel('value')
        else:
            sub_im = fig.axes[2]

        display_im = getattr(sub_im, method)
        csub_im = display_im(*args_im, **dsp_kwargs)

        if method == 'imshow' and len(fig.axes) < 4:
            # Create colorbar if none seems to exist
            minval_im = np.min(values.imag)
            maxval_im = np.max(values.imag)
            ticks_im = [minval_im, (maxval_im + minval_im) / 2., maxval_im]

            plt.colorbar(csub_im, orientation='horizontal',
                         ticks=ticks_im, format='%.4g')

    else:
        if len(fig.axes) == 0:
            # Create new axis object if needed
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
        else:
            sub = fig.axes[0]

        display = getattr(sub, method)
        csub = display(*args_re, **dsp_kwargs)

        if method == 'imshow' and len(fig.axes) < 2:
            # Create colorbar if none seems to exist

            # Use clim from kwargs if given
            if 'clim' not in kwargs:
                minval = np.min(values)
                maxval = np.max(values)
            else:
                minval, maxval = kwargs['clim']

            ticks = [minval, (maxval + minval) / 2., maxval]
            if minval == maxval:
                decimals = 5
            else:
                decimals = max(4, int(1 + abs(np.log10(maxval - minval))))
            format = '%.{}f'.format(decimals)

            plt.colorbar(mappable=csub, ticks=ticks, format=format)

    # Fixes overlapping stuff at the expense of potentially squashed
    # subplots
    fig.tight_layout()

    if title is not None:
        if not values_are_complex:
            # Do not overwrite title for complex values
            plt.title(title)
        fig.canvas.manager.set_window_title(title)

    if show:
        if updatefig:
            plt.show(block=False)
        else:
            plt.show()

    if updatefig:
        plt.draw()
        plt.pause(0.01)

    if saveto is not None:
        fig.savefig(saveto)

    return fig
