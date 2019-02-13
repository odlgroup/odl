# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Functions for folders and files."""

from __future__ import print_function
from builtins import super
import numpy as np
import odl
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt
from skimage.io import imsave

__all__ = ('total_variation', 'TotalVariationNonNegative', 'bregman',
           'save_image', 'save_signal', 'divide_1Darray_equally', 'Blur2D',
           'KullbackLeiblerSmooth')


def save_image(image, name, folder, fignum, cmap='gray', clim=None):
    matplotlib.rc('text', usetex=False)

    fig = plt.figure(fignum)
    plt.clf()

    image.show(name, cmap=cmap, fig=fig)
    fig.savefig('{}/{}_fig.png'.format(folder, name), bbox_inches='tight')

    if clim is None:
        x = image - np.min(image)
        if np.max(x) > 1e-4:
            x /= np.max(x)
    else:
        x = (image - clim[0]) / (clim[1] - clim[0])

    x = np.minimum(np.maximum(x, 0), 1)

    imsave('{}/{}.png'.format(folder, name), np.rot90(x, 1))


def save_signal(signal, name, folder, fignum):
    matplotlib.rc('text', usetex=False)

    fig = plt.figure(fignum)
    plt.clf()

    signal.show(name, fig=fig)
    fig.savefig('{}/{}_fig.png'.format(folder, name), bbox_inches='tight')


def bregman(f, v, subgrad):
    return (odl.solvers.FunctionalQuadraticPerturb(f, linear_term=-subgrad) -
            f(v) + subgrad.inner(v))


def partition_1d(arr, slices):
    return tuple(arr[slc] for slc in slices)


def partition_equally_1d(arr, nparts, order='interlaced'):
    if order == 'block':
        stride = int(np.ceil(arr.size / nparts))
        slices = [slice(i * stride, (i + 1) * stride) for i in range(nparts)]
    elif order == 'interlaced':
        slices = [slice(i, len(arr), nparts) for i in range(nparts)]
    else:
        raise ValueError

    return partition_1d(arr, tuple(slices))


def divide_1Darray_equally(ind, nsub):
    """Divide an array into equal chunks to be used for instance in OSEM.

    Parameters
    ----------
    ind : ndarray
        input array
    nsubsets : int
        number of subsets to be divided into

    Returns
    -------
    sub2ind : list
        list of indices for each subset
    ind2sub : list
        list of subsets for each index
    """

    n_ind = len(ind)
    sub2ind = partition_equally_1d(ind, nsub, order='interlaced')

    ind2sub = []
    for i in range(n_ind):
        ind2sub.append([])

    for i in range(nsub):
        for j in sub2ind[i]:
            ind2sub[j].append(i)

    return (sub2ind, ind2sub)


def total_variation(domain, grad=None):
    """Total variation functional.

    Parameters
    ----------
    domain : odlspace
        domain of TV functional
    grad : gradient operator, optional
        Gradient operator of the total variation functional. This may be any
        linear operator and thereby generalizing TV. default=forward
        differences with Neumann boundary conditions

    Examples
    --------
    Check that the total variation of a constant is zero

    >>> import odl.contrib.spdhg as spdhg, odl
    >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
    >>> tv = spdhg.total_variation(space)
    >>> x = space.one()
    >>> tv(x) < 1e-10
    """

    if grad is None:
        grad = odl.Gradient(domain, method='forward', pad_mode='symmetric')
        grad.norm = 2 * np.sqrt(sum(1 / grad.domain.cell_sides**2))
    else:
        grad = grad

    f = odl.solvers.GroupL1Norm(grad.range, exponent=2)

    return f * grad


class TotalVariationNonNegative(odl.solvers.Functional):
    """Total variation function with nonnegativity constraint and strongly
    convex relaxation.

    In formulas, this functional may represent

        alpha * |grad x|_1 + char_fun(x) + beta/2 |x|^2_2

    with regularization parameter alpha and strong convexity beta. In addition,
    the nonnegativity constraint is achieved with the characteristic function

        char_fun(x) = 0 if x >= 0 and infty else.

    Parameters
    ----------
    domain : odlspace
        domain of TV functional
    alpha : scalar, optional
        Regularization parameter, positive
    prox_options : dict, optional
        name: string, optional
            name of the method to perform the prox operator, default=FGP
        warmstart: boolean, optional
            Do you want a warm start, i.e. start with the dual variable
            from the last call? default=True
        niter: int, optional
            number of iterations per call, default=5
        p: array, optional
            initial dual variable, default=zeros
    grad : gradient operator, optional
        Gradient operator to be used within the total variation functional.
        default=see TV
    """

    def __init__(self, domain, alpha=1, prox_options={}, grad=None,
                 strong_convexity=0):
        """
        """

        self.strong_convexity = strong_convexity

        if 'name' not in prox_options:
            prox_options['name'] = 'FGP'
        if 'warmstart' not in prox_options:
            prox_options['warmstart'] = True
        if 'niter' not in prox_options:
            prox_options['niter'] = 5
        if 'p' not in prox_options:
            prox_options['p'] = None
        if 'tol' not in prox_options:
            prox_options['tol'] = None

        self.prox_options = prox_options

        self.alpha = alpha
        self.tv = total_variation(domain, grad=grad)
        self.grad = self.tv.right
        self.nn = odl.solvers.IndicatorBox(domain, 0, np.inf)
        self.l2 = 0.5 * odl.solvers.L2NormSquared(domain)
        self.proj_P = self.tv.left.convex_conj.proximal(0)
        self.proj_C = self.nn.proximal(1)

        super().__init__(space=domain, linear=False, grad_lipschitz=0)

    def __call__(self, x):
        """Evaluate functional.

        Examples
        --------
        Check that the total variation of a constant is zero

        >>> import odl.contrib.spdhg as spdhg, odl
        >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
        >>> tvnn = spdhg.TotalVariationNonNegative(space, alpha=2)
        >>> x = space.one()
        >>> tvnn(x) < 1e-10

        Check that negative functions are mapped to infty

        >>> import odl.contrib.spdhg as spdhg, odl, numpy as np
        >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
        >>> tvnn = spdhg.TotalVariationNonNegative(space, alpha=2)
        >>> x = -space.one()
        >>> np.isinf(tvnn(x))
        """

        nn = self.nn(x)

        if nn is np.inf:
            return nn
        else:
            out = self.alpha * self.tv(x) + nn
            if self.strong_convexity > 0:
                out += self.strong_convexity * self.l2(x)
            return out

    def proximal(self, sigma):
        """Prox operator of TV. It allows the proximal step length to be a
        vector of positive elements.

        Examples
        --------
        Check that the proximal operator is the identity for sigma=0

        >>> import odl.contrib.solvers.spdhg as spdhg, odl, numpy as np
        >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
        >>> tvnn = spdhg.TotalVariationNonNegative(space, alpha=2)
        >>> x = -space.one()
        >>> y = tvnn.proximal(0)(x)
        >>> (y-x).norm() < 1e-10

        Check that negative functions are mapped to 0

        >>> import odl.contrib.solvers.spdhg as spdhg, odl, numpy as np
        >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
        >>> tvnn = spdhg.TotalVariationNonNegative(space, alpha=2)
        >>> x = -space.one()
        >>> y = tvnn.proximal(0.1)(x)
        >>> y.norm() < 1e-10
        """

        if sigma == 0:
            return odl.IdentityOperator(self.domain)

        else:
            def tv_prox(z, out=None):

                if out is None:
                    out = z.space.zero()

                opts = self.prox_options

                sigma_ = np.copy(sigma)
                z_ = z.copy()

                if self.strong_convexity > 0:
                    sigma_ /= (1 + sigma * self.strong_convexity)
                    z_ /= (1 + sigma * self.strong_convexity)

                if opts['name'] == 'FGP':
                    if opts['warmstart']:
                        if opts['p'] is None:
                            opts['p'] = self.grad.range.zero()

                        p = opts['p']
                    else:
                        p = self.grad.range.zero()

                    sigma_sqrt = np.sqrt(sigma_)

                    z_ /= sigma_sqrt
                    grad = sigma_sqrt * self.grad
                    grad.norm = sigma_sqrt * self.grad.norm
                    niter = opts['niter']
                    alpha = self.alpha
                    out[:] = fgp_dual(p, z_, alpha, niter, grad, self.proj_C,
                                      self.proj_P, tol=opts['tol'])

                    out *= sigma_sqrt

                    return out

                else:
                    raise NotImplementedError('Not yet implemented')

            return tv_prox


def fgp_dual(p, data, alpha, niter, grad, proj_C, proj_P, tol=None, **kwargs):
    """Computes a solution to the ROF problem with the fast gradient
    projection algorithm.

    Parameters
    ----------
    p : np.array
        dual initial variable
    data : np.array
        noisy data / proximal point
    alpha : float
        regularization parameter
    niter : int
        number of iterations
    grad : instance of gradient class
        class that supports grad(x), grad.adjoint(x), grad.norm
    proj_C : function
        projection onto the constraint set of the primal variable,
        e.g. non-negativity
    proj_P : function
        projection onto the constraint set of the dual variable,
        e.g. norm <= 1
    tol : float (optional)
        nonnegative parameter that gives the tolerance for convergence. If set
        None, then the algorithm will run for a fixed number of iterations

    Other Parameters
    ----------------
    callback : callable, optional
        Function called with the current iterate after each iteration.
    """

    # Callback object
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'.format(callback))

    factr = 1 / (grad.norm**2 * alpha)

    q = p.copy()
    x = data.space.zero()

    t = 1.

    if tol is None:
        def convergence_eval(p1, p2):
            return False
    else:
        def convergence_eval(p1, p2):
            return (p1 - p2).norm() / p1.norm() < tol

    pnew = p.copy()

    if callback is not None:
        callback(p)

    for k in range(niter):
        t0 = t
        grad.adjoint(q, out=x)
        proj_C(data - alpha * x, out=x)
        grad(x, out=pnew)
        pnew *= factr
        pnew += q

        proj_P(pnew, out=pnew)

        converged = convergence_eval(p, pnew)

        if not converged:
            # update step size
            t = (1 + np.sqrt(1 + 4 * t0 ** 2)) / 2.

            # calculate next iterate
            q[:] = pnew + (t0 - 1) / t * (pnew - p)

        p[:] = pnew

        if converged:
            t = None
            break

        if callback is not None:
            callback(p)

    # get current image estimate
    x = proj_C(data - alpha * grad.adjoint(p))

    return x


class Blur2D(odl.Operator):
    """Blur operator"""

    def __init__(self, domain, kernel, boundary_condition='wrap'):
        """Initialize a new instance.
        """

        super().__init__(domain=domain, range=domain, linear=True)

        self.__kernel = kernel
        self.__boundary_condition = boundary_condition

    @property
    def kernel(self):
        return self.__kernel

    @property
    def boundary_condition(self):
        return self.__boundary_condition

    def _call(self, x, out):
        out[:] = scipy.signal.convolve2d(x, self.kernel, mode='same',
                                         boundary='wrap')

    @property
    def gradient(self):
        raise NotImplementedError('No yet implemented')

    @property
    def adjoint(self):
        adjoint_kernel = self.kernel.copy().conj()
        adjoint_kernel = np.fliplr(np.flipud(adjoint_kernel))

        return Blur2D(self.domain, adjoint_kernel, self.boundary_condition)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.kernel,
            self.boundary_condition)


class KullbackLeiblerSmooth(odl.solvers.Functional):

    r"""The smooth Kullback-Leibler divergence functional.

    Notes
    -----
    If the functional is defined on an :math:`\mathbb{R}^n`-like space, the
    smooth Kullback-Leibler functional :math:`\phi` is defined as

    .. math::
        \phi(x) = \sum_{i=1}^n \begin{cases}
                x + r - y + y * \log(y / (x + r))
                    & \text{if $x \geq 0$} \
                (y / (2 * r^2)) * x^2 + (1 - y / r) * x + r - b +
                    b * \log(b / r) & \text{else}
                                 \end{cases}

    where all variables on the right hand side of the equation have a subscript
    i which is omitted for readability.

    References
    ----------
    [CERS2017] A. Chambolle, M. J. Ehrhardt, P. Richtarik and C.-B. Schoenlieb,
    *Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling
    and Imaging Applications*. ArXiv: http://arxiv.org/abs/1706.04957 (2017).
    """

    def __init__(self, space, data, background):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `TensorSpace`
            Domain of the functional.
        data : ``space`` `element-like`
            Data vector which has to be non-negative.
        background : ``space`` `element-like`
            Background vector which has to be non-negative.
        """

        self.strong_convexity = 0

        if background.ufuncs.less_equal(0).ufuncs.sum() > 0:
            raise NotImplementedError('Background must be positive')

        super().__init__(space=space, linear=False,
                         grad_lipschitz=np.max(data / background ** 2))

        if data not in self.domain:
            raise ValueError('`data` not in `domain`'
                             ''.format(data, self.domain))

        self.__data = data
        self.__background = background

    @property
    def data(self):
        """The data in the Kullback-Leibler functional."""
        return self.__data

    @property
    def background(self):
        """The background in the Kullback-Leibler functional."""
        return self.__background

    def _call(self, x):
        """Return the KL-diveregnce in the point ``x``.

        If any components of ``x`` is non-positive, the value is positive
        infinity.
        """
        y = self.data
        r = self.background

        obj = self.domain.zero()

        # x + r - y + y * log(y / (x + r)) = x - y * log(x + r) + c1
        # with c1 = r - y + y * log y
        i = x.ufuncs.greater_equal(0)
        obj[i] = x[i] + r[i] - y[i]

        j = y.ufuncs.greater(0)
        k = i.ufuncs.logical_and(j)
        obj[k] += y[k] * (y[k] / (x[k] + r[k])).ufuncs.log()

        # (y / (2 * r^2)) * x^2 + (1 - y / r) * x + r - b + b * log(b / r)
        # = (y / (2 * r^2)) * x^2 + (1 - y / r) * x + c2
        # with c2 = r - b + b * log(b / r)
        i = i.ufuncs.logical_not()
        obj[i] += (y[i] / (2 * r[i]**2) * x[i]**2 + (1 - y[i] / r[i]) * x[i] +
                   r[i] - y[i])

        k = i.ufuncs.logical_and(j)
        obj[k] += y[k] * (y[k] / r[k]).ufuncs.log()

        return obj.inner(self.domain.one())

    @property
    def gradient(self):
        """Gradient operator of the functional.
        """
        raise NotImplementedError('No yet implemented')

    @property
    def proximal(self):
        """Return the `proximal factory` of the functional.
        """
        raise NotImplementedError('No yet implemented')

    @property
    def convex_conj(self):
        """The convex conjugate functional of the KL-functional."""
        return KullbackLeiblerSmoothConvexConj(self.domain, self.data,
                                               self.background)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.data, self.background)


class KullbackLeiblerSmoothConvexConj(odl.solvers.Functional):

    r"""The convex conj of the smooth Kullback-Leibler divergence functional.

    Notes
    -----
    If the functional is defined on an :math:`\mathbb{R}^n`-like space, the
    convex conjugate of the smooth Kullback-Leibler functional :math:`\phi^*`
    is defined as

    .. math::
        \phi^*(x) = \sum_{i=1}^n \begin{cases}
                r^2 / (2 * y) * x^2 + (r - r^2 / y) * x + r^2 / (2 * y) +
                    3 / 2 * y - 2 * r - y * log(y / r)
                    & \text{if $x < 1 - y / r$} \
                - r * x - y * log(1 - x)
                    & \text{if $1 - y / r <= x < 1} \
                + \infty
                    & \text{else}
                                 \end{cases}

    where all variables on the right hand side of the equation have a subscript
    :math:`i` which is omitted for readability.

    References
    ----------
    [CERS2017] A. Chambolle, M. J. Ehrhardt, P. Richtarik and C.-B. Schoenlieb,
    *Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling
    and Imaging Applications*. ArXiv: http://arxiv.org/abs/1706.04957 (2017).
    """

    def __init__(self, space, data, background):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `TensorSpace`
            Domain of the functional.
        data : ``space`` `element-like`
            Data vector which has to be non-negative.
        background : ``space`` `element-like`
            Background vector which has to be non-negative.
        """

        if background.ufuncs.less_equal(0).ufuncs.sum() > 0:
            raise NotImplementedError('Background must be positive')

        super().__init__(space=space, linear=False,
                         grad_lipschitz=np.inf)

        if data is not None and data not in self.domain:
            raise ValueError('`data` not in `domain`'
                             ''.format(data, self.domain))

        self.__data = data
        self.__background = background

        if np.min(self.data) == 0:
            self.strong_convexity = np.inf
        else:
            self.strong_convexity = np.min(self.background**2 / self.data)

    @property
    def data(self):
        """The data in the Kullback-Leibler functional."""
        return self.__data

    @property
    def background(self):
        """The background in the Kullback-Leibler functional."""
        return self.__background

    def _call(self, x):
        """Return the value in the point ``x``.

        If any components of ``x`` is larger than or equal to 1, the value is
        positive infinity.
        """

        # TODO: cover properly the case data = 0

        y = self.data
        r = self.background

        # if any element is greater or equal to one
        if x.ufuncs.greater_equal(1).ufuncs.sum() > 0:
            return np.inf

        obj = self.domain.zero()

        # out = sum(f)
        # f =
        #     if x < 1 - y / r:
        #         r^2 / (2 * y) * x^2 + (r - r^2 / y) * x + r^2 / (2 * y) +
        #             3 / 2 * y - 2 * r - y * log(y / r)
        #     if x >= 1 - y / r:
        #         - r * x - y * log(1 - x)
        i = x.ufuncs.less(1 - y / r)
        ry = r[i]**2 / y[i]
        obj[i] += (ry / 2 * x[i]**2 + (r[i] - ry) * x[i] + ry / 2 +
                   3 / 2 * y[i] - 2 * r[i])

        j = y.ufuncs.greater(0)
        k = i.ufuncs.logical_and(j)
        obj[k] -= y[k] * (y[k] / r[k]).ufuncs.log()

        i = i.ufuncs.logical_not()
        obj[i] -= r[i] * x[i]

        k = i.ufuncs.logical_and(j)
        obj[k] -= y[k] * (1 - x[k]).ufuncs.log()

        return obj.inner(self.domain.one())

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        raise NotImplementedError('No yet implemented')

    @property
    def proximal(self):

        space = self.domain
        y = self.data
        r = self.background

        class ProxKullbackLeiblerSmoothConvexConj(odl.Operator):
            """Proximal operator of the convex conjugate of the smooth
            Kullback-Leibler functional.
            """

            def __init__(self, sigma):
                """Initialize a new instance.

                Parameters
                ----------
                sigma : positive float
                    Step size parameter
                """
                self.sigma = float(sigma)
                self.background = r
                self.data = y
                super().__init__(domain=space, range=space, linear=False)

            def _call(self, x, out):

                s = self.sigma
                y = self.data
                r = self.background

                sr = s * r
                sy = s * y

                # out =
                #    if x  < 1 - y / r:
                #        (y * x - s * r * y + s * r**2) / (y + s * r**2)
                #    if x >= 1 - y / r:
                #        0.5 * (x + s * r + 1 -
                #            sqrt((x + s * r - 1)**2 + 4 * s * y)

                i = x.ufuncs.less(1 - y / r)
                # TODO: This may be faster without indexing on the GPU?
                out[i] = ((y[i] * x[i] - sr[i] * y[i] + sr[i] * r[i]) /
                          (y[i] + sr[i] * r[i]))

                i.ufuncs.logical_not(out=i)
                out[i] = (x[i] + sr[i] + 1 -
                          ((x[i] + sr[i] - 1) ** 2 + 4 * sy[i]).ufuncs.sqrt())
                out[i] /= 2

                return out

        return ProxKullbackLeiblerSmoothConvexConj

    @property
    def convex_conj(self):
        """The convex conjugate functional of the smooth KL-functional."""
        return KullbackLeiblerSmooth(self.domain, self.data,
                                     self.background)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.data, self.background)
