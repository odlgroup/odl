# coding: utf-8

# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Standardized tests for `Operator`'s."""

from __future__ import print_function, division, absolute_import
from functools import partial
import numpy as np
import sys

from odl.diagnostics.examples import samples
from odl.operator.operator import Operator

__all__ = ('check_operator', 'check_operator_properties',
           'check_operator_norm', 'check_operator_linearity',
           'check_operator_adjoint', 'check_operator_derivative')

VERBOSITY_LEVELS = {'DEBUG': 0, 'INFO': 1, 'CHECK': 2, 'WARNING': 3,
                    'ERROR': 4, 'SECTION': 99, 'QUIET': 100}


#TODO: add `logfile` parameter


def _log(message, verbosity, level='DEBUG', file=sys.stderr):
    """Log a message depending on verbosity."""
    if not message:
        # Make sure an empty string is printed as well
        message = '\n'

    if level == verbosity == 'QUIET':
        # Special case, don't print 'QUIET :'
        print(message, file=file)

    elif VERBOSITY_LEVELS[level] >= VERBOSITY_LEVELS[verbosity]:
        message_lines = str(message).splitlines()
        message_lines = ['{:<7}: '.format(level) + line
                         for line in message_lines]
        print('\n'.join(message_lines), file=file)


def _log_linewidth():
    """Return the available log linewidth based on ``np.get_printoptions``."""
    start_len = max(len(k) for k in VERBOSITY_LEVELS) + len(': ')
    end_len = len(' NOT OK')
    return np.get_printoptions()['linewidth'] - start_len - end_len


def _check_cond(cond, cond_str, logger, level_true, level_false):
    """Check a condition and log, returning 1 on failure for ERROR."""
    log_linewidth = _log_linewidth()
    parts = [cond_str[i * log_linewidth: (i + 1) * log_linewidth]
             for i in range(int(np.ceil(len(cond_str) / log_linewidth)))]
    if cond:
        log_fmt = '{{:<{}}} OK'.format(log_linewidth)
        parts[-1] = log_fmt.format(parts[-1])
        level = level_true
        failed = 0
    else:
        log_fmt = '{{:<{}}} NOT OK'.format(log_linewidth)
        parts[-1] = log_fmt.format(parts[-1])
        level = level_false
        failed = 1 if level_false == 'ERROR' else 0

    for part in parts:
        logger(part, level=level)

    return failed


def _get_derivative(op, arg=None):
    """Return a tuple ``(has_deriv, deriv)`` from ``op``.

    Calls ``op.derivative(arg)`` if ``arg`` is given, otherwise uses
    ``op.domain.one()`` if possible, or ``op.domain.element()``.
    """
    if arg is None:
        try:
            arg = op.domain.one()
        except (AttributeError, NotImplementedError):
            arg = op.domain.element()
    try:
        deriv = op.derivative(arg)
        has_deriv = True
    except NotImplementedError:
        has_deriv = False
        deriv = None
    return has_deriv, deriv


def _get_inverse(op):
    """Return a tuple ``(has_inverse, inverse)`` from ``op``."""
    try:
        inverse = op.inverse
        has_inverse = True
    except NotImplementedError:
        has_inverse = False
        inverse = None
    return has_inverse, inverse


def _get_adjoint(op):
    """Return a tuple ``(has_adjoint, adjoint)`` from ``op``."""
    try:
        adjoint = op.adjoint
        has_adjoint = True
    except NotImplementedError:
        has_adjoint = False
        adjoint = None
    return has_adjoint, adjoint


def _get_opnorm(op, maxiter=2):
    """Return a tuple ``(norm_exact, norm_est)`` from ``op``.

    If ``op.norm(estimate=False)`` is not implemented, ``norm_exact`` is
    ``None``, likewise for ``norm_est`` and ``op.norm(estimate=True)``.
    The ``maxiter`` parameter is used for the norm estimate iteration.
    """
    try:
        norm_exact = op.norm(estimate=False)
    except NotImplementedError:
        norm_exact = None
    try:
        norm_est = op.norm(estimate=True, maxiter=maxiter)
    except (TypeError, ValueError, NotImplementedError):
        norm_est = None

    return norm_exact, norm_est


def print_inputs(args, kwargs, verbosity):
    """Print all function inputs for a certain verbosity level."""
    log = partial(_log, verbosity=verbosity)
    log_linewidth = _log_linewidth()

    log('', level='DEBUG')
    log('Inputs', level='DEBUG')
    log('-' * log_linewidth, level='DEBUG')
    for arg in args:
        log(repr(arg), level='DEBUG')
    for key, val in kwargs.items():
        log('{} = {!r}'.format(key, val), level='DEBUG')


def check_operator_properties(operator, verbosity='INFO', deriv_arg=None):
    """Check and return basic operator properties.

    This function checks whether ``derivative``, ``inverse`` and ``adjoint``
    are implemented.

    Parameters
    ----------
    operator : `Operator`
        The operator on which to run the check.
    verbosity : str, optional
        Level of output verbosity. Possible values and corresponding print
        outputs are:

        - ``'DEBUG'``: Everything
        - ``'INFO'``: Informational context, warnings and errors
        - ``'WARNING'``: Warnings and errors
        - ``'ERROR'``: Errors
        - ``'QUIET'``: Only a summary at the end

    deriv_arg : ``operator.domain`` element-like, optional
        Argument to ``operator.derivative``. For the default ``None``,
        ``operator.domain.one()`` is used if possible, else an uninitialized
        ``operator.domain.element()``.

    Returns
    -------
    result : dict
        Dictionary with the following keys:

        - ``num_failed(int)``: Number of failed checks.
        - ``deriv(Operator or None)``: The derivative at ``deriv_arg`` if
          implemented, else ``None``.
        - ``inverse(Operator or None)``: The inverse if implemented,
          else ``None``.
        - ``adjoint(Operator or None)``: The adjoint if implemented,
          else ``None``.
    """
    assert isinstance(operator, Operator), 'bad type {}'.format(type(operator))
    op = operator
    verbosity, verb_in = str(verbosity).upper(), verbosity
    assert verbosity in VERBOSITY_LEVELS, 'bad verbosity {!r}'.format(verb_in)

    num_failed = 0

    log = partial(_log, verbosity=verbosity)
    log_linewidth = _log_linewidth()

    log('', level='SECTION')
    log('Basic properties', level='SECTION')
    log('=' * log_linewidth, level='SECTION')

    print_inputs(
        args=[op],
        kwargs={'verbosity': verbosity, 'deriv_arg': deriv_arg},
        verbosity=verbosity)

    log('## Getting operator properties...', level='DEBUG')
    has_deriv, deriv = _get_derivative(op, arg=deriv_arg)
    has_inverse, inverse = _get_inverse(op)
    has_adjoint, adjoint = _get_adjoint(op)
    log('## Done.', level='DEBUG')

    # --- Default properties --- #

    log('op.domain = {!r}'.format(op.domain), level='INFO')
    log('op.range = {!r}'.format(op.range), level='INFO')
    log('op.is_linear is {}'.format(op.is_linear), level='INFO')
    log('operator.is_functional is {}'.format(op.is_functional),
        level='INFO')

    # --- Derivative --- #

    log('', level='SECTION')
    log('Derivative', level='SECTION')
    log('-' * log_linewidth, level='SECTION')

    if has_deriv:
        log('op.derivative implemented', level='INFO')
        log('op.derivative(x) = {!r}'.format(deriv), level='INFO')
        log('[x = {!r}]'.format(deriv_arg), level='DEBUG')
    else:
        log('op.derivative NOT implemented', level='INFO')

    # --- Inverse --- #

    log('', level='SECTION')
    log('Inverse', level='SECTION')
    log('-' * log_linewidth, level='SECTION')

    if has_inverse:
        log('op.inverse implemented', level='INFO')
        log('inverse = {!r}'.format(inverse), level='INFO')
    else:
        log('op.inverse NOT implemented', level='INFO')

    if has_inverse:
        num_failed += _check_cond(
            inverse.domain == op.range, 'op.inverse.domain == op.range',
            log, level_true='CHECK', level_false='ERROR')
        num_failed += _check_cond(
            inverse.range == op.domain, 'op.inverse.range == op.domain',
            log, level_true='CHECK', level_false='ERROR')

    # --- Adjoint --- #

    log('', level='SECTION')
    log('Adjoint', level='SECTION')
    log('-' * log_linewidth, level='SECTION')

    if has_adjoint:
        log('op.adjoint implemented', level='INFO')
        log('adjoint = {!r}'.format(adjoint), level='INFO')
    else:
        log('op.adjoint NOT implemented', level='INFO')

    # --- Summary --- #

    if verbosity == 'QUIET':
        log('properties: {} failed'.format(num_failed), level='QUIET')
    else:
        failed_level = 'INFO' if num_failed == 0 else 'ERROR'
        log('', level=failed_level)
        log('## Number of failed checks: {}'.format(num_failed),
            level=failed_level)

    return dict(num_failed=num_failed,
                deriv=deriv, inverse=inverse, adjoint=adjoint)


def check_operator_norm(operator, verbosity='INFO', tol=1e-5,
                        norm_kwargs=None):
    """Check and return the operator norm.

    This function checks whether ``norm()`` is available, with both
    ``estimate=True`` and ``estimate=False``. If both are available, it is
    verified that the estimate is less than or equal to the exact norm.

    Parameters
    ----------
    operator : `Operator`
        The operator on which to run the check.
    verbosity : str, optional
        Level of output verbosity. Possible values and corresponding print
        outputs are:

        - ``'DEBUG'``: Everything
        - ``'INFO'``: Informational context, warnings and errors
        - ``'WARNING'``: Warnings and errors
        - ``'ERROR'``: Errors
        - ``'QUIET'``: Only a summary at the end

    tol : float, optional
        Relative tolerance for the norm comparison.
    norm_kwargs : dict, optional
        Keyword arguments to be used as follows::

            operator.norm(estimate=False, **norm_kwargs)

        The default ``None`` is equivalent to ``{'maxiter': 10}``.

    Returns
    -------
    result : dict
        Dictionary with the following keys:

        - ``num_failed(int)``: Number of failed checks.
        - ``opnorm_exact(float or None)``: The exact operator norm if
          ``operator.norm(estimate=False)`` is implemented,
          otherwise ``None``.
        - ``opnorm_est(float or None)``: The estimate for the operator norm
          using a power iteration if applicable, otherwise ``None``.
    """
    assert isinstance(operator, Operator), 'bad type {}'.format(type(operator))
    op = operator
    verbosity, verb_in = str(verbosity).upper(), verbosity
    assert verbosity in VERBOSITY_LEVELS, 'bad verbosity {!r}'.format(verb_in)
    tol = float(tol)
    if norm_kwargs is None:
        norm_kwargs = {'maxiter': 10}

    num_failed = 0

    log = partial(_log, verbosity=verbosity)
    log_linewidth = _log_linewidth()

    log('', level='SECTION')
    log('Operator norm', level='SECTION')
    log('=' * log_linewidth, level='SECTION')

    print_inputs(
        args=[op],
        kwargs={'verbosity': verbosity,
                'tol': tol,
                'norm_kwargs': norm_kwargs},
        verbosity=verbosity)

    # --- Exact norm --- #

    try:
        norm_exact = op.norm(estimate=False)
        has_exact_norm = True
    except NotImplementedError:
        norm_exact = None
        has_exact_norm = False

    if has_exact_norm:
        log('Exact norm `op.norm(estimate=False)` implemented.',
            level='INFO')
        log('Exact norm: {}'.format(norm_exact), level='INFO')
    else:
        log('Exact norm `op.norm(estimate=False)` NOT implemented.',
            level='INFO')

    # --- Norm estimate --- #

    has_adjoint, _ = _get_adjoint(op)
    if has_adjoint:
        log('## Computing operator norm estimate...', level='DEBUG')
        norm_est = op.norm(estimate=True, **norm_kwargs)
        log('## Done.', level='DEBUG')
        log('Estimated norm: {}'.format(norm_est), level='INFO')
        if has_exact_norm:
            num_failed += _check_cond(
                norm_est <= norm_exact * (1 + tol),
                'estimated norm <= exact norm',
                log, level_true='CHECK', level_false='ERROR')
    else:
        log('Operator has no adjoint, skipping norm estimate.', level='INFO')
        norm_est = None

    # --- Summary --- #

    if verbosity == 'QUIET':
        log('norm: {} failed'.format(num_failed), level='QUIET')
    else:
        failed_level = 'INFO' if num_failed == 0 else 'ERROR'
        log('', level=failed_level)
        log('## Number of failed checks: {}'.format(num_failed),
            level=failed_level)

    return dict(num_failed=num_failed,
                norm_exact=norm_exact, norm_est=norm_est)


def check_operator_linearity(operator, verbosity='INFO', opnorm=None,
                             tol=1e-5):
    """Check whether the operator really is linear.

    This function verifies additivity ::

        A(x + y) = A(x) + A(y)

    and scale invariance ::

        A(s * x) = s * A(x)

    for vectors ``x``, ``y`` and scalars ``s``.

    Parameters
    ----------
    operator : `Operator`
        The operator on which to run the check.
    verbosity : str, optional
        Level of output verbosity. Possible values and corresponding print
        outputs are:

        - ``'DEBUG'``: Everything
        - ``'INFO'``: Informational context, warnings and errors
        - ``'WARNING'``: Warnings and errors
        - ``'ERROR'``: Errors
        - ``'QUIET'``: Only a summary at the end

    opnorm : float, optional
        Operator norm used to scale the error in order to make it
        scale-invariant. For ``None``, it is retrieved or computed on the fly.
    tol : float, optional
        Relative tolerance parameter for the error in the checks.

    Returns
    -------
    result : dict
        Dictionary with the following keys:

        - ``num_failed(int)``: Number of failed checks.
    """
    assert isinstance(operator, Operator), 'bad type {}'.format(type(operator))
    op = operator
    verbosity, verb_in = str(verbosity).upper(), verbosity
    assert verbosity in VERBOSITY_LEVELS, 'bad verbosity {!r}'.format(verb_in)
    tol = float(tol)

    num_failed = 0

    log = partial(_log, verbosity=verbosity)
    log_linewidth = _log_linewidth()

    log('', level='SECTION')
    log('Linearity', level='SECTION')
    log('=' * log_linewidth, level='SECTION')

    print_inputs(
        args=[op],
        kwargs={'verbosity': verbosity,
                'opnorm': opnorm,
                'tol': tol},
        verbosity=verbosity)

    if opnorm is None:
        try:
            opnorm = op.norm(estimate=False)
        except NotImplementedError:
            try:
                opnorm = op.norm(estimate=True, maxiter=2)
            except (TypeError, ValueError, NotImplementedError):
                pass

    if opnorm is None:
        log('unable to get or compute operator norm, using opnorm=1.0',
            level='WARNING')
        opnorm = 1.0
    elif opnorm == 0:
        log('opnorm = 0 given, using 1.0 instead', level='WARNING')
        opnorm = 1.0

    # --- Scale invariance --- #

    log('', level='SECTION')
    log('Scale invariance', level='SECTION')
    log('-' * log_linewidth, level='SECTION')
    log('err = ||op(s*x) - s * op(x)|| / (|s| * ||op|| * ||x||)',
        level='INFO')
    log('-' * log_linewidth, level='INFO')
    for (name_x, x), (_, s) in samples(op.domain, op.domain.field):
        s_op_x = s * op(x)
        op_s_x = op(s * x)

        denom = abs(s) * opnorm * x.norm()
        if denom == 0:
            denom = 1.0

        err = (op_s_x - s_op_x).norm() / denom
        num_failed += _check_cond(
            err <= tol,
            'x={:<20}  s={:< 7.2f}  err={:.1}'.format(name_x, s, err),
            log, level_true='CHECK', level_false='ERROR')

        # Compute only if necessary
        op_s_x_norm = op_s_x.norm() if verbosity == 'DEBUG' else 1.0
        s_op_x_norm = s_op_x.norm() if verbosity == 'DEBUG' else 1.0
        log('||op(s*x)||={:.3}'.format(op_s_x_norm), level='DEBUG')
        log('||s * op(x)||={:.3}'.format(s_op_x_norm), level='DEBUG')
        log('|s|*||op||*||x||={:.3}'.format(denom), level='DEBUG')

    log('-' * log_linewidth, level='INFO')

    # --- Additivity --- #

    log('', level='SECTION')
    log('Additivity', level='SECTION')
    log('-' * log_linewidth, level='SECTION')
    log('err = ||op(x + y) - op(x) - op(y)|| / (||op|| * (||x|| + ||y||))',
        level='INFO')
    log('-' * log_linewidth, level='INFO')

    for (name_x, x), (name_y, y) in samples(op.domain, op.domain):
        op_x = op(x)
        op_y = op(y)
        op_x_y = op(x + y)

        denom = opnorm * (x.norm() + y.norm())
        if denom == 0:
            denom = 1.0

        err = (op_x_y - op_x - op_y).norm() / denom
        num_failed += _check_cond(
            err <= tol,
            'x={:<20}  y={:<20}  err={:.1}'.format(name_x, name_y, err),
            log, level_true='CHECK', level_false='ERROR')

        # Compute only if necessary
        op_x_y_norm = op_x_y.norm() if verbosity == 'DEBUG' else 1.0
        op_x_op_y_norm = (op_x + op_y).norm() if verbosity == 'DEBUG' else 1.0
        log('||op(x + y)||={:.3}'.format(op_x_y_norm), level='DEBUG')
        log('||op(x) + op(y)||={:.3}'.format(op_x_op_y_norm), level='DEBUG')
        log('||op||*(||x||+||y||)={:.3}'.format(denom), level='DEBUG')

    log('-' * log_linewidth, level='INFO')

    # --- Summary --- #

    if verbosity == 'QUIET':
        log('linearity: {} failed'.format(num_failed), level='QUIET')
    else:
        failed_level = 'INFO' if num_failed == 0 else 'ERROR'
        log('', level=failed_level)
        log('## Number of failed checks: {}'.format(num_failed),
            level=failed_level)

    return dict(num_failed=num_failed)


def check_operator_adjoint(operator, verbosity='INFO', opnorm=None, tol=1e-5):
    """Check whether the adjoint satisfies its mathematical properties.

    This function verifies the adjointness property ::

        <A(x), y>_Y = <x, A^*(y)>_X

    and whether the adjoint of the adjoint is equivalent to the original
    operator.

    Parameters
    ----------
    operator : `Operator`
        The operator on which to run the check.
    verbosity : str, optional
        Level of output verbosity. Possible values and corresponding print
        outputs are:

        - ``'DEBUG'``: Everything
        - ``'INFO'``: Informational context, warnings and errors
        - ``'WARNING'``: Warnings and errors
        - ``'ERROR'``: Errors
        - ``'QUIET'``: Only a summary at the end

    opnorm : float, optional
        Operator norm used to scale the error in order to make it
        scale-invariant. For ``None``, it is retrieved or computed on the fly.
    tol : float, optional
        Relative tolerance parameter for the error in the checks.

    Returns
    -------
    result : dict
        Dictionary with the following keys:

        - ``num_failed(int)``: Number of failed checks.
    """
    assert isinstance(operator, Operator), 'bad type {}'.format(type(operator))
    op = operator
    verbosity, verb_in = str(verbosity).upper(), verbosity
    assert verbosity in VERBOSITY_LEVELS, 'bad verbosity {!r}'.format(verb_in)
    tol = float(tol)

    num_failed = 0

    log = partial(_log, verbosity=verbosity)
    log_linewidth = _log_linewidth()

    log('', level='SECTION')
    log('Adjoint', level='SECTION')
    log('=' * log_linewidth, level='SECTION')

    print_inputs(
        args=[op],
        kwargs={'verbosity': verbosity,
                'opnorm': opnorm,
                'tol': tol},
        verbosity=verbosity)

    if opnorm is None:
        try:
            opnorm = op.norm(estimate=False)
        except NotImplementedError:
            try:
                opnorm = op.norm(estimate=True, maxiter=2)
            except (TypeError, ValueError, NotImplementedError):
                pass

    if opnorm is None:
        log('unable to get or compute operator norm, using opnorm=1.0',
            level='WARNING')
        opnorm = 1.0
    elif opnorm == 0:
        log('opnorm = 0 given, using 1.0 instead', level='WARNING')
        opnorm = 1.0

    has_adjoint, adjoint = _get_adjoint(op)
    if not has_adjoint:
        log('Operator adjoint not implemented, skipping checks', level='INFO')
        return dict(num_failed=num_failed)

    # --- Basic properties --- #

    num_failed += _check_cond(
        adjoint.is_linear, 'op.adjoint.is_linear',
        log, level_true='CHECK', level_false='WARNING')
    num_failed += _check_cond(
        adjoint.domain == op.range, 'op.adjoint.domain == op.range',
        log, level_true='CHECK', level_false='ERROR')
    num_failed += _check_cond(
        adjoint.range == op.domain, 'op.adjoint.range == op.domain',
        log, level_true='CHECK', level_false='ERROR')

    # --- Adjoint definition --- #

    log('', level='SECTION')
    log('Adjoint definition', level='SECTION')
    log('-' * log_linewidth, level='SECTION')
    log('err = |<op(x), y> - <x, adj(y)>| / (||op|| * ||x|| * ||y||)',
        level='INFO')
    log('-' * log_linewidth, level='INFO')
    inner1_vals = []
    inner2_vals = []
    num_failed_def = 0
    for (name_x, x), (name_y, y) in samples(op.domain, op.domain):
        inner1 = op(x).inner(y)
        inner2 = x.inner(adjoint(y))
        inner1_vals.append(inner1)
        inner2_vals.append(inner2)

        denom = opnorm * x.norm() * y.norm()
        if denom == 0:
            denom = 1.0

        err = abs(inner1 - inner2) / denom
        num_failed_def += _check_cond(
            err <= tol,
            'x={:<20}  y={:<20}  err={:.1}'.format(name_x, name_y, err),
            log, level_true='CHECK', level_false='ERROR')

        log('<op(x), y>={:.3}'.format(inner1), level='DEBUG')
        log('<x, adj(y)>={:.3}'.format(inner2), level='DEBUG')
        log('||op||*||x||*||y||={:.3}'.format(denom), level='DEBUG')

    prop_level = 'DEBUG' if num_failed_def == 0 else 'ERROR'
    factor = np.polyfit(inner1_vals, inner2_vals, deg=1)[0]
    log('', level=prop_level)
    log('Proportionality constant: <x, adj(y)> ~ factor * <op(x), y>',
        level=prop_level)
    log('with factor = {:.3}'.format(factor), level=prop_level)

    num_failed += num_failed_def
    log('-' * log_linewidth, level='INFO')

    # --- Adjoint of adjoint --- #

    log('', level='SECTION')
    log('Adjoint of adjoint', level='SECTION')
    log('-' * log_linewidth, level='SECTION')
    log('err = ||op(x) - adj.adjoint(x)|| / (||op|| * ||x||)',
        level='INFO')
    log('-' * log_linewidth, level='INFO')

    for (name_x, x) in samples(op.domain):
        op_x = op(x)
        adj_adj_x = adjoint.adjoint(x)

        denom = opnorm * x.norm()
        if denom == 0:
            denom = 1.0

        err = (op_x - adj_adj_x).norm() / denom
        num_failed += _check_cond(
            err <= tol,
            'x={:<20}  err={:.1}'.format(name_x, err),
            log, level_true='CHECK', level_false='ERROR')

        # Compute only if necessary
        op_x_norm = op_x.norm() if verbosity == 'DEBUG' else 1.0
        adj_adj_x_norm = adj_adj_x.norm() if verbosity == 'DEBUG' else 1.0
        log('||op(x)||={:.3}'.format(op_x_norm), level='DEBUG')
        log('||adj.adjoint(x)||={:.3}'.format(adj_adj_x_norm), level='DEBUG')
        log('||op||*||x||={:.3}'.format(denom), level='DEBUG')

    log('-' * log_linewidth, level='INFO')

    # --- Summary --- #

    if verbosity == 'QUIET':
        log('adjoint: {} failed'.format(num_failed), level='QUIET')
    else:
        failed_level = 'INFO' if num_failed == 0 else 'ERROR'
        log('', level=failed_level)
        log('## Number of failed checks: {}'.format(num_failed),
            level=failed_level)

    return dict(num_failed=num_failed)


def check_operator_derivative(operator, verbosity='INFO', tol=1e-4):
    """Check whether the derivative satisfies its mathematical properties.

    This function verifies that the ``derivative`` can be approximated
    by finite differences in chosen directions (Gâteaux derivative) ::

        A'(x)(v) ~ [A(x + tv) - A(x)] / t   for t --> 0.

    Parameters
    ----------
    operator : `Operator`
        The operator on which to run the check.
    verbosity : str, optional
        Level of output verbosity. Possible values and corresponding print
        outputs are:

        - ``'DEBUG'``: Everything
        - ``'INFO'``: Informational context, warnings and errors
        - ``'WARNING'``: Warnings and errors
        - ``'ERROR'``: Errors
        - ``'QUIET'``: Only a summary at the end

    tol : float, optional
        Relative tolerance parameter for the error in the checks. Since
        derivative checking is prone to numerical instability, this tolerance
        needs to be larger than in other, more stable checks.

    Returns
    -------
    result : dict
        Dictionary with the following keys:

        - ``num_failed(int)``: Number of failed checks.
    """
    assert isinstance(operator, Operator), 'bad type {}'.format(type(operator))
    op = operator
    verbosity, verb_in = str(verbosity).upper(), verbosity
    assert verbosity in VERBOSITY_LEVELS, 'bad verbosity {!r}'.format(verb_in)
    tol = float(tol)

    num_failed = 0

    log = partial(_log, verbosity=verbosity)
    log_linewidth = _log_linewidth()

    log('', level='SECTION')
    log('Derivative', level='SECTION')
    log('=' * log_linewidth, level='SECTION')

    print_inputs(
        args=[op],
        kwargs={'verbosity': verbosity,
                'tol': tol},
        verbosity=verbosity)

    has_deriv, deriv = _get_derivative(op)
    if not has_deriv:
        log('Operator derivative not implemented, skipping checks',
            level='INFO')
        return dict(num_failed=num_failed)

    # --- Basic properties --- #

    num_failed += _check_cond(
        deriv.is_linear, 'op.derivative(x).is_linear',
        log, level_true='CHECK', level_false='WARNING')
    num_failed += _check_cond(
        deriv.domain == op.domain, 'op.derivative(x).domain == op.domain',
        log, level_true='CHECK', level_false='ERROR')
    num_failed += _check_cond(
        deriv.range == op.range, 'op.derivative(x).range == op.range',
        log, level_true='CHECK', level_false='ERROR')

    if op.is_linear:
        num_failed += _check_cond(
            deriv is op, 'op.is_linear and op is op.derivative(x)',
            log, level_true='CHECK', level_false='WARNING')

    # --- Directional (Gâteaux) derivative --- #

    log('', level='SECTION')
    log('Directional derivative', level='SECTION')
    log('-' * log_linewidth, level='SECTION')
    log('err = ||(op(x + c*dx) - op(x)) / c - deriv(x)(dx)|| / ', level='INFO')
    log('      (||deriv(x)|| * ||dx||)', level='INFO')
    log('-' * log_linewidth, level='INFO')

    for (name_x, x), (name_dx, dx) in samples(op.domain, op.domain):
        # Precompute some values
        deriv_x = op.derivative(x)
        deriv_x_dx = deriv_x(dx)

        num_failed += _check_cond(
            deriv_x.is_linear, 'op.derivative(x).is_linear',
            log, level_true='DEBUG', level_false='WARNING')
        num_failed += _check_cond(
            deriv_x.domain == op.domain,
            'op.derivative(x).domain == op.domain',
            log, level_true='DEBUG', level_false='ERROR')
        num_failed += _check_cond(
            deriv_x.range == op.range, 'op.derivative(x).range == op.range',
            log, level_true='DEBUG', level_false='ERROR')

        op_x = op(x)
        try:
            deriv_x_norm = deriv_x.norm(estimate=False)
        except NotImplementedError:
            try:
                deriv_x_norm = deriv_x.norm(estimate=True, maxiter=2)
            except (ValueError, TypeError, NotImplementedError):
                deriv_x_norm = 1.0

        denom = deriv_x_norm * dx.norm()
        if denom == 0:
            denom = 1.0

        # Compute finite difference with decreasing step size, where the
        # range depends on the data type precision

        # Start value; float32: c = 1e-3, float64: c = 1e-5
        c = np.cbrt(np.finfo(op.domain.dtype).resolution)
        deriv_ok = False

        cs = []
        diff_norms = []
        errs = []
        while c >= 10 * np.finfo(op.domain.dtype).resolution:
            finite_diff = (op(x + c * dx) - op_x) / c
            err = (finite_diff - deriv_x_dx).norm() / denom

            cs.append(c)
            errs.append(err)
            # Compute only if needed
            if verbosity == 'DEBUG':
                diff_norms.append(finite_diff.norm())

            if err < tol:
                deriv_ok = True
                break

            c /= 10.0

        num_failed += _check_cond(
            deriv_ok, 'x={:<20} dx={:<20} minerr={:.1}'
            ''.format(name_x, name_dx, min(errs)),
            log, level_true='CHECK', level_false='ERROR')

        if verbosity == 'DEBUG':
            deriv_x_dx_norm = deriv_x_dx.norm()

        log('||deriv(x)||*||dx||={:.3}'.format(denom), level='DEBUG')
        for c, err, diff_norm in zip(cs, errs, diff_norms):
            log('c={:.1}  err={:.3}'.format(c, err), level='DEBUG')
            log('||(op(x + c*dx) - op(x)) / c||={:.3}'.format(diff_norm),
                level='DEBUG')
            log('||deriv(x)(dx)||={:.3}'.format(deriv_x_dx_norm),
                level='DEBUG')

    log('-' * log_linewidth, level='INFO')

    # --- Summary --- #

    if verbosity == 'QUIET':
        log('derivative: {} failed'.format(num_failed), level='QUIET')
    else:
        failed_level = 'INFO' if num_failed == 0 else 'ERROR'
        log('', level=failed_level)
        log('## Number of failed checks: {}'.format(num_failed),
            level=failed_level)

    return dict(num_failed=num_failed)


def check_operator(operator, verbosity='INFO', checks=None, tol=1e-5,
                   deriv_arg=None, norm_kwargs=None):
    """Run a set of standard tests on the provided operator.

    Parameters
    ----------
    operator : `Operator`
        The operator on which to run the checks.
    verbosity : str, optional
        Level of output verbosity. Possible values and corresponding print
        outputs are:

        - ``'DEBUG'``: Everything
        - ``'INFO'``: Informational context, warnings and errors
        - ``'WARNING'``: Warnings and errors
        - ``'ERROR'``: Errors
        - ``'QUIET'``: Only a summary at the end

    checks : sequence of str, optional
        Checks that should be run. Available checks are:

        - ``'properties'``: Basic checks for domain, range etc., see
          `check_operator_properties`
        - ``'norm'``: Check exact vs. estimated operator norm if available,
          see `check_operator_norm`
        - ``'linearity'``: Test for scale invariance and additivity, see
          `check_operator_linearity`
        - ``'adjoint'``: Check adjointness properties, see
          `check_operator_adjoint`
        - ``'derivative'``: Verify the directional derivative using finite
          differences (note that this may be subject to numerical instability),
          see `check_operator_derivative`

        For the default ``None``, the first 4 checks are run if
        ``operator.is_linear``, otherwise the first and the last.

    tol : float, optional
        Tolerance parameter used as a base for the actual tolerance
        in the tests. Depending on the expected accuracy, the actual
        tolerance used in a test can be a factor times this number.
    deriv_arg : ``operator.domain`` element-like, optional
        Argument to ``operator.derivative`` for checking its presence. For
        the default ``None``, ``operator.domain.one()`` is used if possible,
        else an uninitialized ``operator.domain.element()``.
    norm_kwargs : dict, optional
        Keyword arguments to be used as follows::

            operator.norm(estimate=False, **norm_kwargs)

        The default ``None`` is equivalent to ``{'maxiter': 10}``.
    """
    assert isinstance(operator, Operator), 'bad type {}'.format(type(operator))
    op = operator
    verbosity, verb_in = str(verbosity).upper(), verbosity
    assert verbosity in VERBOSITY_LEVELS, 'bad verbosity {!r}'.format(verb_in)
    all_checks = {'properties', 'norm', 'linearity', 'adjoint', 'derivative'}
    if checks is None:
        if op.is_linear:
            checks = ('properties', 'norm', 'linearity', 'adjoint')
        else:
            checks = ('properties', 'derivative')
    checks, chk_in = tuple(str(c).lower() for c in checks), checks
    assert set(checks).issubset(all_checks), 'invalid checks {}'.format(chk_in)
    tol = float(tol)
    if norm_kwargs is None:
        norm_kwargs = {'maxiter': 10}

    log = partial(_log, verbosity=verbosity)

    log('', level='SECTION')
    log('Operator check', level='SECTION')
    log('==============', level='SECTION')
    log('==============', level='SECTION')

    print_inputs(
        args=[op],
        kwargs={'verbosity': verbosity,
                'checks': checks,
                'tol': tol,
                'deriv_arg': deriv_arg,
                'norm_kwargs': norm_kwargs},
        verbosity=verbosity)

    if 'properties' in checks:
        res_props = check_operator_properties(op, verbosity, deriv_arg)

    if 'norm' in checks:
        res_norm = check_operator_norm(op, verbosity, tol, norm_kwargs)
        norm_exact = res_norm['norm_exact']
        norm_est = res_norm['norm_est']
        opnorm = norm_exact if norm_exact is not None else norm_est
    else:
        opnorm = None

    if 'linearity' in checks:
        res_lin = check_operator_linearity(op, verbosity, opnorm, tol)

    if 'adjoint' in checks:
        res_adj = check_operator_adjoint(op, verbosity, opnorm, tol)

    if 'derivative' in checks:
        res_deriv = check_operator_derivative(op, verbosity, 10 * tol)

    # TODO: do stuff with results


if __name__ == '__main__':
    import odl
    space = odl.uniform_discr([0, 0], [1, 1], [3, 3])
    # Linear operator
    op = odl.ScalingOperator(space, 2.0)
    check_operator(op, verbosity='QUIET')

    # Nonlinear operator op(x) = x**4
    op = odl.PowerOperator(space, 4)
    check_operator(op, verbosity='QUIET')
