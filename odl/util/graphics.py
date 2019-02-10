# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Functions for graphical output."""

from __future__ import print_function, division, absolute_import
import numpy as np
import warnings

from odl.util.testutils import run_doctests
from odl.util.utility import is_real_dtype


__all__ = ('show_discrete_data',)


def warning_free_pause():
    """Issue a matplotlib pause without the warning."""
    import matplotlib.pyplot as plt

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                message="Using default event loop until "
                                        "function specific to this GUI is "
                                        "implemented")
        plt.pause(0.0001)


def _safe_minmax(values):
    """Calculate min and max of array with guards for nan and inf."""

    # Nan and inf guarded min and max
    isfinite = np.isfinite(values)
    if np.any(isfinite):
        # Only use finite values
        values = values[isfinite]

    minval = np.min(values)
    maxval = np.max(values)

    return minval, maxval


def _colorbar_ticks(minval, maxval):
    """Return the ticks (values show) in the colorbar."""
    if not (np.isfinite(minval) and np.isfinite(maxval)):
        return [0, 0, 0]
    elif minval == maxval:
        return [minval]
    else:
        # Add eps to ensure values stay inside the range of the colorbar.
        # Otherwise they may occationally not display.
        eps = (maxval - minval) / 1e5
        return [minval + eps, (maxval + minval) / 2., maxval - eps]


def _digits(minval, maxval):
    """Digits needed to comforatbly display values in [minval, maxval]"""
    if minval == maxval:
        return 3
    else:
        return min(10, max(2, int(1 + abs(np.log10(maxval - minval)))))


def _colorbar_format(minval, maxval):
    """Return the format string for the colorbar."""
    if not (np.isfinite(minval) and np.isfinite(maxval)):
        return str(maxval)
    else:
        return '%.{}f'.format(_digits(minval, maxval))


def _axes_info(grid, npoints=5):
    result = []

    min_pt = grid.min()
    max_pt = grid.max()
    for axis in range(grid.ndim):
        xmin = min_pt[axis]
        xmax = max_pt[axis]

        points = np.linspace(xmin, xmax, npoints)
        indices = np.linspace(0, grid.shape[axis] - 1, npoints, dtype=int)
        tick_values = grid.coord_vectors[axis][indices]

        # Do not use corner point in case of a partition, use outer corner
        tick_values[[0, -1]] = xmin, xmax

        format_str = '{:.' + str(_digits(xmin, xmax)) + 'f}'
        tick_labels = [format_str.format(f) for f in tick_values]

        result += [(points, tick_labels)]

    return result


def show_discrete_data(values, grid, title=None, method='',
                       force_show=False, fig=None, **kwargs):
    """Display a discrete 1d or 2d function.

    Parameters
    ----------
    values : `numpy.ndarray`
        The values to visualize.

    grid : `RectGrid` or `RectPartition`
        Grid of the values.

    title : string, optional
        Set the title of the figure.

    method : string, optional
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

    force_show : bool, optional
        Whether the plot should be forced to be shown now or deferred until
        later. Note that some backends always displays the plot, regardless
        of this value.

    fig : `matplotlib.figure.Figure`, optional
        The figure to show in. Expected to be of same "style", as the figure
        given by this function. The most common usecase is that fig is the
        return value from an earlier call to this function.
        Default: New figure

    interp : {'nearest', 'linear'}, optional
        Interpolation method to use.
        Default: 'nearest'

    axis_labels : string, optional
        Axis labels, default: ['x', 'y']

    update_in_place : bool, optional
        Update the content of the figure in-place. Intended for faster real
        time plotting, typically ~5 times faster.
        This is only performed for ``method == 'imshow'`` with real data and
        ``fig != None``. Otherwise this parameter is treated as False.
        Default: False

    axis_fontsize : int, optional
        Fontsize for the axes. Default: 16

    colorbar : bool, optional
        Argument relevant for 2d plots using ``method='imshow'``. If ``True``,
        include a colorbar in the plot.
        Default: True

    kwargs : {'figsize', 'saveto', ...}, optional
        Extra keyword arguments passed on to display method
        See the Matplotlib functions for documentation of extra
        options.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The resulting figure. It is also shown to the user.

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
    axis_fontsize = kwargs.pop('axis_fontsize', 16)
    colorbar = kwargs.pop('colorbar', True)

    # Normalize input
    interp, interp_in = str(interp).lower(), interp
    method, method_in = str(method).lower(), method

    # Check if we should and can update the plot in-place
    update_in_place = kwargs.pop('update_in_place', False)
    if (update_in_place and
            (fig is None or values_are_complex or values.ndim != 2 or
             (values.ndim == 2 and method not in ('', 'imshow')))):
        update_in_place = False

    if values.ndim == 1:  # TODO: maybe a plotter class would be better
        if not method:
            if interp == 'nearest':
                method = 'step'
                dsp_kwargs['where'] = 'mid'
            elif interp == 'linear':
                method = 'plot'
            else:
                raise ValueError('`interp` {!r} not supported'
                                 ''.format(interp_in))

        if method == 'plot' or method == 'step' or method == 'scatter':
            args_re += [grid.coord_vectors[0], values.real]
            args_im += [grid.coord_vectors[0], values.imag]
        else:
            raise ValueError('`method` {!r} not supported'
                             ''.format(method_in))

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
                raise ValueError('`interp` {!r} not supported'
                                 ''.format(interp_in))

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
            x, y = grid.meshgrid
            args_re = [x, y, np.rot90(values.real)]
            args_im = ([x, y, np.rot90(values.imag)] if values_are_complex
                       else [])
            sub_kwargs.update({'projection': '3d'})
        else:
            raise ValueError('`method` {!r} not supported'
                             ''.format(method_in))

    else:
        raise NotImplementedError('no method for {}d display implemented'
                                  ''.format(values.ndim))

    # Additional keyword args are passed on to the display method
    dsp_kwargs.update(**kwargs)

    if fig is not None:
        # Reuse figure if given as input
        if not isinstance(fig, plt.Figure):
            raise TypeError('`fig` {} not a matplotlib figure'.format(fig))

        if not plt.fignum_exists(fig.number):
            # If figure does not exist, user either closed the figure or
            # is using IPython, in this case we need a new figure.

            fig = plt.figure(figsize=figsize)
            updatefig = False
        else:
            # Set current figure to given input
            fig = plt.figure(fig.number)
            updatefig = True

            if values.ndim > 1 and not update_in_place:
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
            sub_re.set_xlabel(axis_labels[0], fontsize=axis_fontsize)
            if values.ndim == 2:
                sub_re.set_ylabel(axis_labels[1], fontsize=axis_fontsize)
            else:
                sub_re.set_ylabel('value')
        else:
            sub_re = fig.axes[0]

        display_re = getattr(sub_re, method)
        csub_re = display_re(*args_re, **dsp_kwargs)

        # Axis ticks
        if method == 'imshow' and not grid.is_uniform:
            (xpts, xlabels), (ypts, ylabels) = _axes_info(grid)
            plt.xticks(xpts, xlabels)
            plt.yticks(ypts, ylabels)

        if method == 'imshow' and len(fig.axes) < 2:
            # Create colorbar if none seems to exist

            # Use clim from kwargs if given
            if 'clim' not in kwargs:
                minval_re, maxval_re = _safe_minmax(values.real)
            else:
                minval_re, maxval_re = kwargs['clim']

            ticks_re = _colorbar_ticks(minval_re, maxval_re)
            fmt_re = _colorbar_format(minval_re, maxval_re)

            plt.colorbar(csub_re, orientation='horizontal',
                         ticks=ticks_re, format=fmt_re)

        # Imaginary
        if len(fig.axes) < 3:
            sub_im = plt.subplot(arrange_subplots[1], **sub_kwargs)
            sub_im.set_title('Imaginary part')
            sub_im.set_xlabel(axis_labels[0], fontsize=axis_fontsize)
            if values.ndim == 2:
                sub_im.set_ylabel(axis_labels[1], fontsize=axis_fontsize)
            else:
                sub_im.set_ylabel('value')
        else:
            sub_im = fig.axes[2]

        display_im = getattr(sub_im, method)
        csub_im = display_im(*args_im, **dsp_kwargs)

        # Axis ticks
        if method == 'imshow' and not grid.is_uniform:
            (xpts, xlabels), (ypts, ylabels) = _axes_info(grid)
            plt.xticks(xpts, xlabels)
            plt.yticks(ypts, ylabels)

        if method == 'imshow' and len(fig.axes) < 4:
            # Create colorbar if none seems to exist

            # Use clim from kwargs if given
            if 'clim' not in kwargs:
                minval_im, maxval_im = _safe_minmax(values.imag)
            else:
                minval_im, maxval_im = kwargs['clim']

            ticks_im = _colorbar_ticks(minval_im, maxval_im)
            fmt_im = _colorbar_format(minval_im, maxval_im)

            plt.colorbar(csub_im, orientation='horizontal',
                         ticks=ticks_im, format=fmt_im)

    else:
        if len(fig.axes) == 0:
            # Create new axis object if needed
            sub = plt.subplot(111, **sub_kwargs)
            sub.set_xlabel(axis_labels[0], fontsize=axis_fontsize)
            if values.ndim == 2:
                sub.set_ylabel(axis_labels[1], fontsize=axis_fontsize)
            else:
                sub.set_ylabel('value')
            try:
                # For 3d plots
                sub.set_zlabel('z')
            except AttributeError:
                pass
        else:
            sub = fig.axes[0]

        if update_in_place:
            import matplotlib as mpl
            imgs = [obj for obj in sub.get_children()
                    if isinstance(obj, mpl.image.AxesImage)]
            if len(imgs) > 0 and updatefig:
                imgs[0].set_data(args_re[0])
                csub = imgs[0]

                # Update min-max
                if 'clim' not in kwargs:
                    minval, maxval = _safe_minmax(values)
                else:
                    minval, maxval = kwargs['clim']

                csub.set_clim(minval, maxval)
            else:
                display = getattr(sub, method)
                csub = display(*args_re, **dsp_kwargs)
        else:
            display = getattr(sub, method)
            csub = display(*args_re, **dsp_kwargs)

        # Axis ticks
        if method == 'imshow' and not grid.is_uniform:
            (xpts, xlabels), (ypts, ylabels) = _axes_info(grid)
            plt.xticks(xpts, xlabels)
            plt.yticks(ypts, ylabels)

        if method == 'imshow' and colorbar:
            # Add colorbar
            # Use clim from kwargs if given
            if 'clim' not in kwargs:
                minval, maxval = _safe_minmax(values)
            else:
                minval, maxval = kwargs['clim']

            ticks = _colorbar_ticks(minval, maxval)
            fmt = _colorbar_format(minval, maxval)
            if len(fig.axes) < 2:
                # Create colorbar if none seems to exist
                plt.colorbar(mappable=csub, ticks=ticks, format=fmt)
            elif update_in_place:
                # If it exists and we should update it
                csub.colorbar.set_clim(minval, maxval)
                csub.colorbar.set_ticks(ticks)
                if '%' not in fmt:
                    labels = [fmt] * len(ticks)
                else:
                    labels = [fmt % t for t in ticks]
                csub.colorbar.set_ticklabels(labels)
                csub.colorbar.draw_all()

    # Set title of window
    if title is not None:
        if not values_are_complex:
            # Do not overwrite title for complex values
            plt.title(title)
        fig.canvas.manager.set_window_title(title)

    # Fixes overlapping stuff at the expense of potentially squashed subplots
    if not update_in_place:
        fig.tight_layout()

    if updatefig or plt.isinteractive():
        # If we are running in interactive mode, we can always show the fig
        # This causes an artifact, where users of `CallbackShow` without
        # interactive mode only shows the figure after the second iteration.
        plt.show(block=False)
        if not update_in_place:
            plt.draw()
            warning_free_pause()
        else:
            try:
                sub.draw_artist(csub)
                fig.canvas.blit(fig.bbox)
                fig.canvas.update()
                fig.canvas.flush_events()
            except AttributeError:
                plt.draw()
                warning_free_pause()

    if force_show:
        plt.show()

    if saveto is not None:
        fig.savefig(saveto)
    return fig


if __name__ == '__main__':
    run_doctests()
