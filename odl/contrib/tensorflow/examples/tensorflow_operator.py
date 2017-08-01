"""Example of wrapping tensorflow to create an ODL operator."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import tensorflow as tf
import odl


sess = tf.InteractiveSession()

def G(x):
    if(np.array(x.shape).size==1):
        return 1./(np.pi**.5)*np.exp(-x**2/2.)
    elif(np.array(x.shape).size==2):
        return 1./(np.pi**.5)*np.exp(-np.sum(x**2,axis=1)/2.)
    else:
        print('Error: x should be an MxN array')
        return

def G_sigma_m_x_m(x,sigma_m,x_m):
    return G((x-x_m)/sigma_m)

def fp(Arg,x):
    # Arg = 4xM array
    # Arg = [[alpha_j],[sigma_j],[x1_j],[x2_j]]
    fp_temp = np.zeros(x.shape[0])
    M = Arg.shape[1]
    alpha_j = Arg[0]
    sigma_j = Arg[1]
    x_j = Arg[2:].T

    for m in np.arange(M):
        fp_temp = fp_temp + alpha_j[m]*G((x-x_j[m])/sigma_j[m])

    return fp_temp


def R_G(p):
    return tf.exp(-p ** 2. / 2.)


def R_G_sigma_m_x_m(theta, r, sigma_m, x_m, y_m):
    return R_G((r - (tf.cos(theta) * x_m + tf.sin(theta) * y_m)) / sigma_m)


def R_fp(alpha, sigma, x, y, theta, r):
    R_fp_temp = R_G_sigma_m_x_m(theta[:, None], r[:, None],
                                sigma[None, :], x[None, :], y[None, :])

    R_fp_temp = alpha[None, :] * R_fp_temp

    return tf.reduce_sum(R_fp_temp, axis=-1)


class GaussianRayTransform(odl.Operator):
    def __init__(self, domain, range, theta, p):
        size = domain[0].size
        self.alpha_ph = tf.placeholder(tf.float32, shape=[size])
        self.sigma_ph = tf.placeholder(tf.float32, shape=[size])
        self.x_ph = tf.placeholder(tf.float32, shape=[size])
        self.y_ph = tf.placeholder(tf.float32, shape=[size])
        theta = tf.constant(theta, dtype=tf.float32)
        p = tf.constant(p, dtype=tf.float32)

        self.y = tf.placeholder(tf.float32, shape=range.shape)

        self.result = R_fp(self.alpha_ph, self.sigma_ph, self.x_ph, self.y_ph, theta, p)
        self.gradient = tf.gradients(self.result,
                                     [self.alpha_ph, self.sigma_ph, self.x_ph, self.y_ph],
                                     [self.y])
        self.gradient = [gi * range.cell_volume for gi in self.gradient]

        odl.Operator.__init__(self, domain, range)

    def _call(self, x):
        result = sess.run(self.result,
                          feed_dict={self.alpha_ph: np.asarray(x[0]),
                                     self.sigma_ph: np.asarray(x[1]),
                                     self.x_ph: np.asarray(x[2]),
                                     self.y_ph: np.asarray(x[3])})

        return result.reshape(self.range.shape).T

    def derivative(self, x):
        op = self

        class GaussianRayTransformDerivative(odl.Operator):
            @property
            def adjoint(self):
                class GaussianRayTransformDerivativeAdjoint(odl.Operator):
                    def _call(self, y):
                        result = sess.run(op.gradient,
                          feed_dict={op.alpha_ph: np.asarray(x[0]),
                                     op.sigma_ph: np.asarray(x[1]),
                                     op.x_ph: np.asarray(x[2]),
                                     op.y_ph: np.asarray(x[3]),
                                     op.y: np.asarray(y).T})

                        return result

                return GaussianRayTransformDerivativeAdjoint(self.range,
                                                             self.domain,
                                                             linear=True)

        return GaussianRayTransformDerivative(self.domain, self.range,
                                              linear=True)


if __name__ == '__main__':
    n = 201
    x,y = np.meshgrid(np.linspace(-2,2,n)*5,np.linspace(-2,2,n)*5)
    X = np.row_stack((x.flatten(),y.flatten())).T

    M = 13
    r = np.linspace(0,2,M)*4
    x_theta = np.linspace(0,2.*np.pi,M)
    x_ini = r*np.array([np.cos(x_theta),np.sin(x_theta)])
    #plt.plot(x_ini[0],x_ini[1],'.')
    #plt.show()
    alpha_ini = np.ones(M)
    sigma_ini = np.ones(M)
    Arg_ini = np.row_stack((alpha_ini,sigma_ini,x_ini))

    theta,p = np.meshgrid(np.linspace(0,2.*np.pi,n),np.linspace(-20,20,n))
    Theta = theta.flatten()
    P = p.flatten()

    import odl
    rn = odl.rn(M)
    domain = rn ** 4

    ran = odl.uniform_discr([0, -20], [2 * np.pi, 20], [n, n])
    operator = GaussianRayTransform(domain, ran, Theta, P)

    x = domain.element([alpha_ini, sigma_ini, x_ini[0], x_ini[1]])

    y = operator(x)
    y.show()

    deriv = operator.derivative(x)

    adjy = deriv.adjoint(y)
