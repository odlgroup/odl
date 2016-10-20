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

"""Callback objects for per-iterate actions in iterative methods."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import time


__all__ = ('CallbackStore', 'CallbackApply',
           'CallbackPrintTiming', 'CallbackPrintIteration',
           'CallbackPrintNorm', 'CallbackShow')


class SolverCallback(object):

    """Abstract base class for handling iterates of solvers."""

    def __call__(self, result):
        """Apply the callback object to result.

        Parameters
        ----------
        result : `LinearSpaceElement`
            Partial result after n iterations

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
        result : `SolverCallback`
            A callback whose `__call__` method calls both constituents
            `__call__`.

        Examples
        --------
        >>> store = CallbackStore()
        >>> iter = CallbackPrintIteration()
        >>> both = store & iter
        >>> both
        CallbackStore() & CallbackPrintIteration()
        """
        return _CallbackAnd(self, other)

    def reset(self):
        """Reset the callback to its initial state.

        Should be overridden by subclasses.
        """
        pass

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}()'.format(self.__class__.__name__)


class _CallbackAnd(SolverCallback):

    """Callback used for combining several callbacks."""

    def __init__(self, *callbacks):
        """Initialize a new instance.

        Parameters
        ----------
        callback1, ..., callbackN : callable
            Callables to be called in sequence as listed.
        """
        callbacks = [c if isinstance(c, SolverCallback) else CallbackApply(c)
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
        return ' & '.join('{}'.format(p) for p in self.callbacks)


class CallbackStore(SolverCallback):

    """Simple object for storing all iterates of a solver.

    Can optionally apply a function, for example the norm or calculating the
    residual.

    By default, calls the `copy()` method on the iterates before storing.
    """

    def __init__(self, results=None, function=None):
        """Initialize a new instance.

        Parameters
        ----------
        results : list, optional
            List in which to store the iterates.
            Default: new list (``[]``)
        results : callable, optional
            Function to be called on all incoming results before storage.
            Default: copy

        Examples
        --------
        Store results as is

        >>> callback = CallbackStore()

        Provide list to store iterates in.

        >>> results = []
        >>> callback = CallbackStore(results=results)

        Store the norm of the results

        >>> norm_function = lambda x: x.norm()
        >>> callback = CallbackStore(function=norm_function)
        """
        self._results = [] if results is None else results
        self._function = function

    @property
    def results(self):
        """Sequence of partial results."""
        return self._results

    def __call__(self, result):
        """Append result to results list."""
        if self._function:
            self._results.append(self._function(result))
        else:
            self._results.append(result.copy())

    def reset(self):
        """Clear the `results` list."""
        self._results = []

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

    def __str__(self):
        """Return ``str(self)``."""
        resultstr = '' if self.results == [] else str(self.results)
        return 'CallbackStore({})'.format(resultstr)

    def __repr__(self):
        """Return ``repr(self)``."""
        resultrepr = '' if self.results == [] else repr(self.results)
        return 'CallbackStore({})'.format(resultrepr)


class CallbackApply(SolverCallback):

    """Simple object for applying a function to each iterate."""

    def __init__(self, function):
        """Initialize a new instance.

        Parameters
        ----------
        function : callable
            Function to call for each iteration
        """
        assert callable(function)
        self.function = function

    def __call__(self, result):
        """Apply function to result."""
        self.function(result)

    def __str__(self):
        """Return ``str(self)``."""
        return 'CallbackApply({})'.format(self.function)

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'CallbackApply({!r})'.format(self.function)


class CallbackPrintIteration(SolverCallback):

    """Print the iteration count."""

    _default_text = 'iter ='

    def __init__(self, text=None):
        """Initialize a new instance.

        Parameters
        ----------
        text : string
            Text to display before the iteration count. Default: 'iter ='
        """
        self.text = text if text is not None else self._default_text
        self.iter = 0

    def __call__(self, _):
        """Print the current iteration."""
        print("{} {}".format(self.text, self.iter))
        self.iter += 1

    def reset(self):
        """Set `iter` to 0."""
        self.iter = 0

    def __repr__(self):
        """Return ``repr(self)``."""
        textstr = '' if self.text == self._default_text else self.text
        return 'CallbackPrintIteration({})'.format(textstr)


class CallbackPrintTiming(SolverCallback):

    """Print the time elapsed since the previous iteration."""

    def __init__(self):
        """Initialize a new instance."""
        self.time = time.time()

    def __call__(self, _):
        """Print time elapsed from the previous iteration."""
        t = time.time()
        print("Time elapsed = {:<5.03f} s".format(t - self.time))
        self.time = t

    def reset(self):
        """Set `time` to the current time."""
        self.time = time.time()

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'CallbackPrintTiming()'


class CallbackPrintNorm(SolverCallback):

    """Print the current norm."""

    def __call__(self, result):
        """Print the current norm."""
        print("norm = {}".format(result.norm()))

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'CallbackPrintNorm()'


class CallbackShow(SolverCallback):

    """Show the iterates.

    See Also
    --------
    odl.discr.lp_discr.DiscreteLpElement.show
    odl.space.base_ntuples.NtuplesBaseVector.show
    """

    def __init__(self, *args, **kwargs):
        """Initialize a new instance.

        Additional parameters are passed through to the ``show`` method.

        Parameters
        ----------
        display_step : positive int, optional
            Number of iterations between plots. Default: 1

        Other Parameters
        ----------------
        kwargs :
            Optional arguments passed on to ``x.show``
        """
        self.args = args
        self.kwargs = kwargs
        self.fig = kwargs.pop('fig', None)
        self.display_step = kwargs.pop('display_step', 1)
        self.iter = 0

    def __call__(self, x):
        """Show the current iterate."""
        if (self.iter % self.display_step) == 0:
            self.fig = x.show(*self.args, fig=self.fig, **self.kwargs)

        self.iter += 1

    def reset(self):
        """Set `iter` to 0 and create a new figure."""
        self.iter = 0
        self.fig = None

    def __repr__(self):
        """Return ``repr(self)``."""

        return '{}(display_step={}, fig={}, *{!r}, **{!r})'.format(
            self.__class__.__name__,
            self.display_step,
            self.fig,
            self.args,
            self.kwargs)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
