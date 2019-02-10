# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Callback objects for per-iterate actions in iterative methods."""

from __future__ import print_function, division, absolute_import
from builtins import object
import copy
import numpy as np
import os
import time
import warnings

from odl.util import signature_string

__all__ = ('Callback', 'CallbackStore', 'CallbackApply', 'CallbackPrintTiming',
           'CallbackPrintIteration', 'CallbackPrint', 'CallbackPrintNorm',
           'CallbackShow', 'CallbackSaveToDisk', 'CallbackSleep',
           'CallbackShowConvergence', 'CallbackPrintHardwareUsage',
           'CallbackProgressBar')


class Callback(object):

    """Abstract base class for handling iterates of solvers."""

    def __call__(self, iterate):
        """Apply the callback object to result.

        Parameters
        ----------
        iterate : `LinearSpaceElement`
            Partial result after n iterations.

        Returns
        -------
        None
        """

    def __and__(self, other):
        """Return ``self & other``.

        Compose callbacks, calls both in sequence.

        Parameters
        ----------
        other : callable
            The other callback to compose with.

        Returns
        -------
        result : `Callback`
            A callback whose `__call__` method calls both constituents
            `__call__`.

        Examples
        --------
        >>> store = CallbackStore()
        >>> iter = CallbackPrintIteration()
        >>> store & iter
        CallbackStore() & CallbackPrintIteration()
        """
        return _CallbackAnd(self, other)

    def __mul__(self, other):
        """Return ``self * other``.

        Compose callback with operator, calls the callback after calling the
        operator.

        Parameters
        ----------
        other : `Operator`
            The operator to compose with.

        Returns
        -------
        result : `Callback`
            A callback whose `__call__` method calls first the operator, and
            then applies the callback to the result.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> callback = odl.solvers.CallbackPrint()
        >>> operator = odl.ScalingOperator(r3, 2.0)
        >>> composed_callback = callback * operator
        >>> composed_callback([1, 2, 3])
        rn(3).element([ 2.,  4.,  6.])
        """
        return _CallbackCompose(self, other)

    def reset(self):
        """Reset the callback to its initial state.

        Should be overridden by subclasses.
        """
        pass

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}()'.format(self.__class__.__name__)


class _CallbackAnd(Callback):

    """Callback used for combining several callbacks."""

    def __init__(self, *callbacks):
        """Initialize a new instance.

        Parameters
        ----------
        callback1, ..., callbackN : callable
            Callables to be called in sequence as listed.
        """
        callbacks = [c if isinstance(c, Callback) else CallbackApply(c)
                     for c in callbacks]

        self.callbacks = callbacks

    def __call__(self, result):
        """Apply all callbacks to result."""
        for p in self.callbacks:
            p(result)

    def reset(self):
        """Reset all callbacks to their initial state."""
        for callback in self.callbacks:
            callback.reset()

    def __repr__(self):
        """Return ``repr(self)``."""
        return ' & '.join('{!r}'.format(p) for p in self.callbacks)


class _CallbackCompose(Callback):

    """Callback used for the composition of a callback with an operator."""

    def __init__(self, callback, operator):
        """Initialize a new instance.

        Parameters
        ----------
        callback : callable
            The callback to call.
        operator : `Operator`
            Operator to apply before calling the callback.
        """
        self.callback = callback
        self.operator = operator

    def __call__(self, result):
        """Apply the callback."""
        self.callback(self.operator(result))

    def reset(self):
        """Reset the internal callback to its initial state."""
        self.callback.reset()

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> callback = odl.solvers.CallbackPrint()
        >>> operator = odl.ScalingOperator(r3, 2.0)
        >>> callback * operator
        CallbackPrint() * ScalingOperator(rn(3), 2.0)
        """
        return '{!r} * {!r}'.format(self.callback, self.operator)


class CallbackStore(Callback):

    """Callback for storing all iterates of a solver.

    Can optionally apply a function, for example the norm or calculating the
    residual.

    By default, calls the ``copy()`` method on the iterates before storing.
    """

    def __init__(self, results=None, function=None, step=1):
        """Initialize a new instance.

        Parameters
        ----------
        results : list, optional
            List in which to store the iterates.
            Default: new list (``[]``)
        function : callable, optional
            Deprecated, use composition instead. See examples.
            Function to be called on all incoming results before storage.
            Default: copy
        step : int, optional
            Number of iterates between storing iterates.

        Examples
        --------
        Store results as-is:

        >>> callback = CallbackStore()

        Provide list to store iterates in:

        >>> results = []
        >>> callback = CallbackStore(results=results)

        Store the norm of the results:

        >>> norm_function = lambda x: x.norm()
        >>> callback = CallbackStore() * norm_function
        """
        self.results = [] if results is None else results
        self.function = function
        if function is not None:
            warnings.warn('`function` argument is deprecated and will be '
                          'removed in a future release. Use composition '
                          'instead. '
                          'See Examples in the documentation.',
                          DeprecationWarning)
        self.step = int(step)
        self.iter = 0

    def __call__(self, result):
        """Append result to results list."""
        if self.iter % self.step == 0:
            if self.function:
                self.results.append(self.function(result))
            else:
                self.results.append(copy.copy(result))

    def reset(self):
        """Clear the results list."""
        self.results = []
        self.iter = 0

    def __iter__(self):
        """Allow iteration over the results."""
        return iter(self.results)

    def __getitem__(self, index):
        """Return ``self[index]``.

        Get iterates by index.
        """
        return self.results[index]

    def __len__(self):
        """Number of results stored."""
        return len(self.results)

    def __repr__(self):
        """Return ``repr(self)``."""
        optargs = [('results', self.results, []),
                   ('function', self.function, None),
                   ('step', self.step, 1)]
        inner_str = signature_string([], optargs)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class CallbackApply(Callback):

    """Callback for applying a custom function to iterates."""

    def __init__(self, function, step=1):
        """Initialize a new instance.

        Parameters
        ----------
        function : callable
            Function to call on the current iterate.
        step : int, optional
            Number of iterates between applications of ``function``.

        Examples
        --------
        By default, the function is called on each iterate:

        >>> def func(x):
        ...     print(np.max(x))
        >>> callback = CallbackApply(func)
        >>> x = odl.rn(3).element([1, 2, 3])
        >>> callback(x)
        3.0
        >>> callback(x)
        3.0

        To apply only to each n-th iterate, supply ``step=n``:

        >>> callback = CallbackApply(func, step=2)
        >>> callback(x)
        3.0
        >>> callback(x)  # no output
        >>> callback(x)  # next output
        3.0
        """
        assert callable(function)
        self.function = function
        self.step = int(step)
        self.iter = 0

    def __call__(self, result):
        """Apply function to result."""
        if self.iter % self.step == 0:
            self.function(result)
        self.iter += 1

    def reset(self):
        """Set `iter` to 0."""
        self.iter = 0

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.function]
        optargs = [('step', self.step, 1)]
        inner_str = signature_string(posargs, optargs)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class CallbackPrintIteration(Callback):

    """Callback for printing the iteration count."""

    def __init__(self, fmt='iter = {}', step=1, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        fmt : string, optional
            Format string for the text to be printed. The text is printed as::

                print(fmt.format(cur_iter_num))

            where ``cur_iter_num`` is the current iteration number.
        step : positive int, optional
            Number of iterations between output.

        Other Parameters
        ----------------
        kwargs :
            Key word arguments passed to the print function.

        Examples
        --------
        Create simple callback that prints iteration count:

        >>> callback = CallbackPrintIteration()
        >>> callback(None)
        iter = 0
        >>> callback(None)
        iter = 1

        Create callback that every 2nd iterate prints iteration count with
        a custom string:

        >>> callback = CallbackPrintIteration(fmt='Current iter is {}.',
        ...                                   step=2)
        >>> callback(None)
        Current iter is 0.
        >>> callback(None)  # prints nothing
        >>> callback(None)
        Current iter is 2.
        """
        self.fmt = str(fmt)
        self.step = int(step)
        self.iter = 0
        self.kwargs = kwargs

    def __call__(self, _):
        """Print the current iteration."""
        if self.iter % self.step == 0:
            print(self.fmt.format(self.iter), **self.kwargs)

        self.iter += 1

    def reset(self):
        """Set `iter` to 0."""
        self.iter = 0

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> CallbackPrintIteration(fmt='Current iter is {}.', step=2)
        CallbackPrintIteration(fmt='Current iter is {}.', step=2)
        """
        optargs = [('fmt', self.fmt, 'iter = {}'),
                   ('step', self.step, 1)]
        inner_str = signature_string([], optargs)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class CallbackPrintTiming(Callback):

    """Callback for printing the time elapsed since the previous iteration."""

    def __init__(self, fmt='Time elapsed = {:<5.03f} s', step=1,
                 cumulative=False, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        fmt : string, optional
            Formating that should be applied. The time is printed as ::

                print(fmt.format(runtime))

            where ``runtime`` is the runtime since the last iterate.
        step : positive int, optional
            Number of iterations between prints.
        cumulative : boolean, optional
            Print the time since the initialization instead of the last call.

        Other Parameters
        ----------------
        kwargs :
            Key word arguments passed to the print function.
        """
        self.fmt = str(fmt)
        self.step = int(step)
        self.iter = 0
        self.cumulative = cumulative
        self.start_time = time.time()
        self.kwargs = kwargs

    def __call__(self, _):
        """Print time elapsed from the previous iteration."""
        if self.iter % self.step == 0:
            current_time = time.time()

            print(self.fmt.format(current_time - self.start_time),
                  **self.kwargs)

            if not self.cumulative:
                self.start_time = current_time

        self.iter += 1

    def reset(self):
        """Set `time` to the current time."""
        self.start_time = time.time()
        self.iter = 0

    def __repr__(self):
        """Return ``repr(self)``."""
        optargs = [('fmt', self.fmt, 'Time elapsed = {:<5.03f} s'),
                   ('step', self.step, 1),
                   ('cumulative', self.cumulative, False)]
        inner_str = signature_string([], optargs)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class CallbackPrint(Callback):

    """Callback for printing the current value."""

    def __init__(self, func=None, fmt='{!r}', step=1, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        func : callable, optional
            Deprecated, use composition instead. See examples.
            Functional that should be called on the current iterate before
            printing. Default: print current iterate.
        fmt : string, optional
            Formating that should be applied. Will be used as ::

                print(fmt.format(x))

            where ``x`` is the input to the callback.
        step : positive int, optional
            Number of iterations between prints.

        Other Parameters
        ----------------
        kwargs :
            Key word arguments passed to the print function.

        Examples
        --------
        Callback for simply printing the current iterate:

        >>> callback = CallbackPrint()
        >>> callback([1, 2])
        [1, 2]

        Apply function before printing via composition:

        >>> callback = CallbackPrint() * np.sum
        >>> callback([1, 2])
        3

        Format to two decimal points:

        >>> callback = CallbackPrint(fmt='{0:.2f}') * np.sum
        >>> callback([1, 2])
        3.00
        """
        self.func = func
        if func is not None:
            warnings.warn('`func` argument is deprecated and will be removed '
                          'in a future release. Use composition instead. '
                          'See Examples in the documentation.',
                          DeprecationWarning)
        if func is not None and not callable(func):
            raise TypeError('`func` must be `callable` or `None`')

        self.fmt = str(fmt)
        self.step = int(step)
        self.iter = 0
        self.kwargs = kwargs

    def __call__(self, result):
        """Print the current value."""
        if self.iter % self.step == 0:
            if self.func is not None:
                result = self.func(result)

            print(self.fmt.format(result), **self.kwargs)

        self.iter += 1

    def reset(self):
        """Set `iter` to 0."""
        self.iter = 0

    def __repr__(self):
        """Return ``repr(self)``."""
        optargs = [('func', self.func, None),
                   ('fmt', self.fmt, '{!r}'),
                   ('step', self.step, 1)]
        inner_str = signature_string([], optargs)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class CallbackPrintNorm(Callback):

    """Callback for printing the current norm."""

    def __call__(self, result):
        """Print the current norm."""
        print("norm = {}".format(result.norm()))

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}()'.format(self.__class__.__name__)


class CallbackShow(Callback):

    """Callback for showing iterates.

    See Also
    --------
    odl.discr.lp_discr.DiscreteLpElement.show
    odl.space.base_tensors.Tensor.show
    """

    def __init__(self, title=None, step=1, saveto=None, **kwargs):
        """Initialize a new instance.

        Additional parameters are passed through to the ``show`` method.

        Parameters
        ----------
        title : str, optional
            Format string for the title of the displayed figure.
            The title name is generated as ::

                title = title.format(cur_iter_num)

            where ``cur_iter_num`` is the current iteration number.
            For the default ``None``, the title format ``'Iterate {}'``
            is used.
        step : positive int, optional
            Number of iterations between plots.
        saveto : str or callable, optional
            Format string for the name of the file(s) where
            iterates are saved.

            If ``saveto`` is a string, the file name is generated as ::

                filename = saveto.format(cur_iter_num)

            where ``cur_iter_num`` is the current iteration number.

            If ``saveto`` is a callable, the file name is generated as ::

                filename = saveto(cur_iter_num)

            If the directory name does not exist, a ``ValueError`` is raised.
            If ``saveto is None``, the figures are not saved.

        Other Parameters
        ----------------
        kwargs :
            Optional keyword arguments passed on to ``x.show``.

        Examples
        --------
        Show the result of each iterate:

        >>> callback = CallbackShow()

        Show and save every fifth iterate in ``png`` format, overwriting the
        previous one:

        >>> callback = CallbackShow(step=5,
        ...                         saveto='my_path/my_iterate.png')

        Show and save each fifth iterate in ``png`` format, indexing the files
        with the iteration number:

        >>> callback = CallbackShow(step=5,
        ...                         saveto='my_path/my_iterate_{}.png')

        Pass additional arguments to ``show``:

        >>> callback = CallbackShow(step=5, clim=[0, 1])
        """
        if title is None:
            self.title = 'Iterate {}'
        else:
            self.title = str(title)
        self.title_formatter = self.title.format

        self.saveto = saveto
        self.saveto_formatter = getattr(self.saveto, 'format', self.saveto)

        self.step = step
        self.fig = kwargs.pop('fig', None)
        self.iter = 0
        self.space_of_last_x = None
        self.kwargs = kwargs

    def __call__(self, x):
        """Show the current iterate."""
        # Check if we should update the figure in-place
        x_space = x.space
        update_in_place = (self.space_of_last_x == x_space)
        self.space_of_last_x = x_space

        if self.iter % self.step == 0:
            title = self.title_formatter(self.iter)

            if self.saveto is None:
                self.fig = x.show(title, fig=self.fig,
                                  update_in_place=update_in_place,
                                  **self.kwargs)

            else:
                saveto = self.saveto_formatter(self.iter)
                self.fig = x.show(title, fig=self.fig,
                                  update_in_place=update_in_place,
                                  saveto=saveto, **self.kwargs)

        self.iter += 1

    def reset(self):
        """Set `iter` to 0 and create a new figure."""
        self.iter = 0
        self.fig = None
        self.space_of_last_x = None

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = []
        if self.title != 'Iterate {}':
            posargs.append(self.title)
        optargs = [('step', self.step, 1),
                   ('saveto', self.saveto, None)]
        for kwarg, value in self.kwargs.items():
            optargs.append((kwarg, value, None))
        inner_str = signature_string(posargs, optargs)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class CallbackSaveToDisk(Callback):

    """Callback for saving iterates to disk."""

    def __init__(self, saveto, step=1, impl='pickle', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        saveto : string
            Format string for the name of the file(s) where
            iterates are saved. The file name is generated as

                filename = saveto.format(cur_iter_num)

            where ``cur_iter_num`` is the current iteration number.
        step : positive int, optional
            Number of iterations between saves.
        impl : {'pickle', 'numpy', 'numpy_txt'}, optional
            The format to store the iterates in. Numpy formats are only usable
            if the data can be converted to an array via `numpy.asarray`.

        Other Parameters
        ----------------
        kwargs :
            Optional arguments passed to the save function.

        Examples
        --------
        Store each iterate:

        >>> callback = CallbackSaveToDisk('my_path/my_iterate')

        Save every fifth overwriting the previous one:

        >>> callback = CallbackSaveToDisk(saveto='my_path/my_iterate',
        ...                               step=5)

        Save each fifth iterate in ``numpy`` format, indexing the files with
        the iteration number:

        >>> callback = CallbackSaveToDisk(saveto='my_path/my_iterate_{}',
        ...                               step=5, impl='numpy')
        """
        self.saveto = saveto
        try:
            self.saveto_formatter = self.saveto.format
        except AttributeError:
            self.saveto_formatter = self.saveto

        self.step = int(step)
        self.impl = str(impl).lower()
        self.kwargs = kwargs
        self.iter = 0

    def __call__(self, x):
        """Save the current iterate."""
        if self.iter % self.step == 0:
            file_path = self.saveto_formatter(self.iter)
            folder_path = os.path.dirname(os.path.realpath(file_path))

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            if self.impl == 'pickle':
                import pickle
                with open(file_path, 'wb+') as f:
                    pickle.dump(x, f, **self.kwargs)
            elif self.impl == 'numpy':
                np.save(file_path, np.asarray(x), **self.kwargs)
            elif self.impl == 'numpy_txt':
                np.savetxt(file_path, np.asarray(x), **self.kwargs)
            else:
                raise RuntimeError('unknown `impl` {}'.format(self.impl))

        self.iter += 1

    def reset(self):
        """Set `iter` to 0."""
        self.iter = 0

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.saveto]
        optargs = [('step', self.step, 1),
                   ('impl', self.impl, 'pickle')]
        for kwarg, value in self.kwargs.items():
            optargs.append((kwarg, value, None))
        inner_str = signature_string(posargs, optargs)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class CallbackSleep(Callback):

    """Callback for sleeping for a specific time span."""

    def __init__(self, seconds=1.0):
        """Initialize a new instance.

        Parameters
        ----------
        seconds : float, optional
            Number of seconds to sleep, can be float for subsecond precision.

        Examples
        --------
        Sleep 1 second between consecutive iterates:

        >>> callback = CallbackSleep(seconds=1)

        Sleep 10 ms between consecutive iterate:

        >>> callback = CallbackSleep(seconds=0.01)
        """
        self.seconds = float(seconds)

    def __call__(self, x):
        """Sleep for a specified time."""
        time.sleep(self.seconds)

    def __repr__(self):
        """Return ``repr(self)``."""
        optargs = [('seconds', self.seconds, 1.0)]
        inner_str = signature_string([], optargs)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class CallbackShowConvergence(Callback):

    """Displays a convergence plot."""

    def __init__(self, functional, title='convergence', logx=False, logy=False,
                 **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        functional : callable
            Function that is called with the current iterate and returns the
            function value.
        title : str, optional
            Title of the plot.
        logx : bool, optional
            If true, the x axis is logarithmic.
        logx : bool, optional
            If true, the y axis is logarithmic.

        Other Parameters
        ----------------
        kwargs :
            Additional parameters passed to the scatter-plotting function.
        """
        self.functional = functional
        self.title = title
        self.logx = logx
        self.logy = logy
        self.kwargs = kwargs
        self.iter = 0

        import matplotlib.pyplot as plt
        self.fig = plt.figure(title)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('iteration')
        self.ax.set_ylabel('function value')
        self.ax.set_title(title)
        if logx:
            self.ax.set_xscale("log", nonposx='clip')
        if logy:
            self.ax.set_yscale("log", nonposy='clip')

    def __call__(self, x):
        """Implement ``self(x)``."""
        if self.logx:
            it = self.iter + 1
        else:
            it = self.iter
        self.ax.scatter(it, self.functional(x), **self.kwargs)
        self.iter += 1

    def reset(self):
        """Set `iter` to 0."""
        self.iter = 0

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}(functional={}, title={}, logx={}, logy={})'.format(
            self.__class__.__name__,
            self.functional,
            self.title,
            self.logx,
            self.logy)


class CallbackPrintHardwareUsage(Callback):

    """Callback for printing memory and CPU usage.

    This callback requires the ``psutil`` package.
    """

    def __init__(self, step=1, fmt_cpu='CPU usage (% each core): {}',
                 fmt_mem='RAM usage: {}', fmt_swap='SWAP usage: {}', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        step : positive int, optional
            Number of iterations between output.
        fmt_cpu : string, optional
            Formating that should be applied. The CPU usage is printed as ::

                print(fmt_cpu.format(cpu))

            where ``cpu`` is a vector with the percentage of current CPU usaged
            for each core. An empty format string disables printing of CPU
            usage.
        fmt_mem : string, optional
            Formating that should be applied. The RAM usage is printed as ::

                print(fmt_mem.format(mem))

            where ``mem`` is the current RAM memory usaged. An empty format
            string disables printing of RAM memory usage.
        fmt_swap : string, optional
            Formating that should be applied. The SWAP usage is printed as ::

                print(fmt_swap.format(swap))

            where ``swap`` is the current SWAP memory usaged. An empty format
            string disables printing of SWAP memory usage.

        Other Parameters
        ----------------
        kwargs :
            Key word arguments passed to the print function.

        Examples
        --------
        Print memory and CPU usage

        >>> callback = CallbackPrintHardwareUsage()

        Only print every tenth step

        >>> callback = CallbackPrintHardwareUsage(step=10)

        Only print the RAM memory usage in every step, and with a non-default
        formatting

        >>> callback = CallbackPrintHardwareUsage(step=1, fmt_cpu='',
        ...                                       fmt_mem='RAM {}',
        ...                                       fmt_swap='')
        """
        self.step = int(step)
        self.fmt_cpu = str(fmt_cpu)
        self.fmt_mem = str(fmt_mem)
        self.fmt_swap = str(fmt_swap)
        self.iter = 0

    def __call__(self, _):
        """Print the memory and CPU usage"""

        import psutil

        if self.iter % self.step == 0:
            if self.fmt_cpu:
                print(self.fmt_cpu.format(psutil.cpu_percent(percpu=True)),
                      **self.kwargs)
            if self.fmt_mem:
                print(self.fmt_mem.format(psutil.virtual_memory()),
                      **self.kwargs)
            if self.fmt_swap:
                print(self.fmt_swap.format(psutil.swap_memory()),
                      **self.kwargs)

        self.iter += 1

    def reset(self):
        """Set `iter` to 0."""
        self.iter = 0

    def __repr__(self):
        """Return ``repr(self)``."""
        optargs = [('step', self.step, 1),
                   ('fmt_cpu', self.fmt_cpu, 'CPU usage (% each core): {}'),
                   ('fmt_mem', self.fmt_mem, 'RAM usage: {}'),
                   ('fmt_swap', self.fmt_swap, 'SWAP usage: {}')]
        inner_str = signature_string([], optargs)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class CallbackProgressBar(Callback):

    """Callback for displaying a progress bar.

    This callback requires the ``tqdm`` package.
    """

    def __init__(self, niter, step=1, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        niter : positive int, optional
            Total number of iterations.
        step : positive int, optional
            Number of iterations between output.

        Other Parameters
        ----------------
        kwargs :
            Further parameters passed to ``tqdm.tqdm``.
        """
        self.niter = int(niter)
        self.step = int(step)
        self.kwargs = kwargs
        self.reset()

    def __call__(self, _):
        """Update the progressbar."""
        if self.iter % self.step == 0:
            self.pbar.update(self.step)

        self.iter += 1

    def reset(self):
        """Set `iter` to 0."""
        import tqdm
        self.iter = 0
        self.pbar = tqdm.tqdm(total=self.niter, **self.kwargs)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.niter]
        optargs = [('step', self.step, 1)]
        inner_str = signature_string(posargs, optargs)
        if self.kwargs:
            return '{}({}, **{})'.format(self.__class__.__name__,
                                         inner_str, self.kwargs)
        else:
            return '{}({})'.format(self.__class__.__name__,
                                   inner_str)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
