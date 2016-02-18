.. _proximal_operators:

##################
Proximal Operators
##################

Definition
----------

Let f be a proper convex function mapping the normed space X to the extended
real number line (-infinity, +infinity]. The proximal operators of the
functional f is mapping from X to X, denoted as prox_tau[f](x) (or
prox_{tau f}(x)) with x in X, and defined by

    prox_tau[f](x) = arg min_y f(y in X) + 1/(2 tau) ||x - y||_2^2

Properties
----------

Some properties which are useful to create or compose proximal operators:

**Separable sum**

    if f is separable across variables, i.e. f(x,y) = g(x) + h(y),
    then

        prox_tau[f](x, y) = (prox_tau[g](x), prox_tau[h](y))

**Post-composition**

    if g(x) = alpha f(x) + a, with alpha > 0, then

        prox_tau[g](x) = prox_{alpha tau}[f(x)](x)

**Pre-composition**

    if g(x) = f(beta x + b), with beta != 0, then

        prox_tau[g](x) = 1/beta (prox_{beta^2 tau}[f](beta x + b) - b)

**Moreau decomposition**

Also know as Moreau identity.

    x = prox_tau[f](x) + 1/tau prox_{1/tau}[f_cc](x / tau)

where f_cc is the convex conjugate of f and defined as

        f_cc(y) = sup_x <y,x> - f(x)

with inner product <.,.>. For more details see [R1970]_
or `Wikipedia <https://en.wikipedia.org/wiki/Convex_conjugate>`_.

For more details on proximal operators including how to evaluate the
proximal operator of a variety of functions see [PB2014]_.


Indicator function
------------------

Indicator functions are typically used to incorporate constraints. The
indicator function for a given set S is defined as

    ind_{S}(x) = {0 if x in S, infinity if x not in S}

**Special indicator functions**

Indicator for a box centered at origin and with width 2 a:

    ind_{box(a)}(x) = {0 if ||x||_infty <= a, infty if ||x||_infty > a}

where ||.||_infty denotes the maximum-norm.
