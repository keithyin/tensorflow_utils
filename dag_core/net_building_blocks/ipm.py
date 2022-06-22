import tensorflow as tf
import numpy as np

SQRT_CONST = 1e-10


def safe_sqrt(x, lbound=SQRT_CONST):
    """ Numerically safe version of TensorFlow sqrt """
    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))


def lindisc(X, p, t):
    """ Linear MMD """

    it = tf.where(t > 0)[:, 0]
    ic = tf.where(t < 1)[:, 0]

    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)

    mean_control = tf.reduce_mean(Xc, reduction_indices=0)
    mean_treated = tf.reduce_mean(Xt, reduction_indices=0)

    c = tf.square(2 * p - 1) * 0.25
    f = tf.sign(p - 0.5)

    mmd = tf.reduce_sum(tf.square(p * mean_treated - (1 - p) * mean_control))
    mmd = f * (p - 0.5) + safe_sqrt(c + mmd)

    return mmd


def mmd2_lin(X, t, p):
    with tf.name_scope("mmd2_lin"):
        """ Linear MMD """
        t = tf.cast(t, dtype=tf.float32)
        if len(t.shape) == 2:
            t = tf.squeeze(t, axis=1)
        treatment_indicator = tf.cast(tf.greater(t, 0.5), dtype=tf.float32)
        treatment_num = tf.reduce_sum(treatment_indicator) + 1e-6
        treatment_indicator = treatment_indicator / treatment_num

        control_indicator = tf.cast(tf.less(t, 0.5), dtype=tf.float32)
        control_num = tf.reduce_sum(control_indicator) + 1e-6
        control_indicator = control_indicator / control_num

        treatment_mean = tf.einsum("n,nd->d", treatment_indicator, X)
        control_mean = tf.einsum("n,nd->d", control_indicator, X)

        mmd = tf.reduce_sum(tf.square(2.0 * p * treatment_mean - 2.0 * (1.0 - p) * control_mean))
        mmd = tf.case({
            treatment_num > (tf.cast(tf.shape(X)[0], dtype=tf.float32) - 1.0): lambda: tf.constant(999.),
            treatment_num < 1.: lambda: tf.constant(999.)
        }, default=lambda: mmd, exclusive=True)

    return mmd


def mmd2_rbf(X, t, p, sig):
    """ Computes the l2-RBF MMD for X given t """

    it = tf.where(t > 0)[:, 0]
    ic = tf.where(t < 1)[:, 0]

    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)

    Kcc = tf.exp(-pdist2sq(Xc, Xc) / tf.square(sig))
    Kct = tf.exp(-pdist2sq(Xc, Xt) / tf.square(sig))
    Ktt = tf.exp(-pdist2sq(Xt, Xt) / tf.square(sig))

    m = tf.to_float(tf.shape(Xc)[0])
    n = tf.to_float(tf.shape(Xt)[0])

    mmd = tf.square(1.0 - p) / (m * (m - 1.0)) * (tf.reduce_sum(Kcc) - m)
    mmd = mmd + tf.square(p) / (n * (n - 1.0)) * (tf.reduce_sum(Ktt) - n)
    mmd = mmd - 2.0 * p * (1.0 - p) / (m * n) * tf.reduce_sum(Kct)
    mmd = 4.0 * mmd

    return mmd


def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * tf.matmul(X, tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
    ny = tf.reduce_sum(tf.square(Y), 1, keep_dims=True)
    D = (C + tf.transpose(ny)) + nx
    return D


def pdist2(X, Y):
    """ Returns the tensorflow pairwise distance matrix """
    return safe_sqrt(pdist2sq(X, Y))


def pop_dist(X, t):
    it = tf.where(t > 0)[:, 0]
    ic = tf.where(t < 1)[:, 0]
    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    M = pdist2(Xt, Xc)
    return M


def wasserstein(X, t, p, lam=10, its=10, sq=False, backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    it = tf.where(t > 0)[:, 0]
    ic = tf.where(t < 1)[:, 0]
    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt, Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt, Xc))

    ''' Estimate lambda and delta '''
    M_mean = tf.reduce_mean(M)
    M_drop = tf.nn.dropout(M, 10 / (nc * nt))
    delta = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam / M_mean)

    ''' Compute new distance matrix '''
    Mt = M
    row = delta * tf.ones(tf.shape(M[0:1, :]))
    col = tf.concat(0, [delta * tf.ones(tf.shape(M[:, 0:1])), tf.zeros((1, 1))])
    Mt = tf.concat(0, [M, row])
    Mt = tf.concat(1, [Mt, col])

    ''' Compute marginal vectors '''
    a = tf.concat(0, [p * tf.ones(tf.shape(tf.where(t > 0)[:, 0:1])) / nt, (1 - p) * tf.ones((1, 1))])
    b = tf.concat(0, [(1 - p) * tf.ones(tf.shape(tf.where(t < 1)[:, 0:1])) / nc, p * tf.ones((1, 1))])

    ''' Compute kernel matrix'''
    Mlam = eff_lam * Mt
    K = tf.exp(-Mlam) + 1e-6  # added constant to avoid nan
    U = K * Mt
    ainvK = K / a

    u = a
    for i in range(0, its):
        u = 1.0 / (tf.matmul(ainvK, (b / tf.transpose(tf.matmul(tf.transpose(u), K)))))
    v = b / (tf.transpose(tf.matmul(tf.transpose(u), K)))

    T = u * (tf.transpose(v) * K)

    if not backpropT:
        T = tf.stop_gradient(T)

    E = T * Mt
    D = 2 * tf.reduce_sum(E)

    return D, Mlam
