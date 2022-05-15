import tensorflow as tf


def dcn(x, num_layers, name_or_scope=None, reuse=False):
    """
    :param x: 2d tensor, [b, dim]
    :param num_layers: int
    :param name_or_scope: name
    :param reuse: reuse or not
    """
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="dcn", reuse=reuse):
        x0 = tf.expand_dims(x, dim=2)
        for i in range(num_layers):
            # [b, n, 1] * [b, 1, n] = [b, n, n]
            b = tf.expand_dims(x, dim=1)
            w_l = tf.get_variable(
                "w_l_{}".format(i), shape=[1, x.shape[1], 1],
                initializer=tf.initializers.glorot_normal)
            w_l = tf.tile(w_l, multiples=[tf.shape(x)[0], 1, 1])
            b_l = tf.get_variable(
                "b_l_{}".format(i), shape=[1, x.shape[1]],
                initializer=tf.initializers.zeros)
            # [b, n, 1] * [b, 1, n] = [b, n, n]
            # [b, n, n] * [b, n, 1] = [b, n, 1] --> b, n
            x = tf.squeeze(tf.matmul(tf.matmul(x0, b), w_l), axis=2) + b_l + x
    return x


def dcn_faster(x, num_layers, name_or_scope=None, reuse=False):
    """
        :param x: 2d tensor, [b, dim]
        :param num_layers: int
        :param name_or_scope: string
        :param reuse: reuse param or not
        """
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="DCN", reuse=reuse):
        x0 = tf.expand_dims(x, dim=2)
        for i in range(num_layers):
            # [b, n, 1] * [b, 1, n] = [b, n, n]
            b = tf.expand_dims(x, dim=1)
            w_l = tf.get_variable(
                "w_l_{}".format(i), shape=[1, x.shape[1], 1],
                initializer=tf.initializers.glorot_normal)
            w_l = tf.tile(w_l, multiples=[tf.shape(x)[0], 1, 1])
            b_l = tf.get_variable(
                "b_l_{}".format(i), shape=[1, x.shape[1]],
                initializer=tf.initializers.zeros)
            # [b, n, 1] * [b, 1, n] = [b, n, n]
            # [b, n, n] * [b, n, 1] = [b, n, 1] --> b, n
            x = tf.squeeze(tf.matmul(x0, tf.matmul(b, w_l)), axis=2) + b_l + x
    return x


def dcn_m(x, num_layers):
    """
    :param x: 2d tensor, [b, dim]
    :param num_layers: int
    """
    with tf.variable_scope(name_or_scope=None, default_name="dcn_m"):
        x0 = x
        for i in range(num_layers):
            w_l = tf.get_variable(
                "w_l_{}".format(i), shape=[x0.shape[1], x0.shape[1]],
                initializer=tf.initializers.glorot_normal)
            b_l = tf.get_variable(
                "b_l_{}".format(i), shape=[1, x0.shape[1]],
                initializer=tf.initializers.zeros)
            # [b, n] * [n, n] -> [b, n]
            x = x0 * (tf.matmul(x, w_l) + b_l) + x
    return x


def dcn_m_low_rank(x, num_layers, dim):
    """
    :param x: 2-D tensor
    :param num_layers: num cross layers
    :param dim: inner rank
    """

    if dim is None:
        dim = int(x.shape[1] / 4)
    assert dim > 0, "dim must be greater than 0, but actual dim={}".format(dim)

    with tf.variable_scope(name_or_scope=None, default_name="dcn_m_low_rank_version"):
        x0 = x
        for i in range(num_layers):
            u_l = tf.get_variable(
                "u_l_{}".format(i), shape=[dim, x.shape[1]],
                initializer=tf.initializers.glorot_normal)
            v_l = tf.get_variable(
                "v_l_{}".format(i), shape=[x0.shape[1], dim],
                initializer=tf.initializers.glorot_normal)
            b_l = tf.get_variable(
                "b_l_{}".format(i), shape=[1, x0.shape[1]],
                initializer=tf.initializers.zeros)
            x = x0 * tf.matmul(tf.matmul(x, v_l), u_l) + b_l + x
    return x


def dcn_m_moe(x, num_layers, num_experts, gate_func_units_list=[1], g_func=tf.nn.tanh, dim=None):
    """
    :param x: 2-D tensor
    :param num_layers: cross layer
    :param num_experts: num of experts
    :param gate_func_units_list: mlp units list of gate func, the last one must be 1
    :param g_func: g() in dcn-M paper in equation-4
    :param dim: inner rank
    """
    if dim is None:
        dim = int(x.shape[1] / 4)
    assert dim > 0, "dim must be greater than 0, but actual dim={}".format(dim)
    assert gate_func_units_list[-1] == 1, "last element of gate_func_units_list must be 1"

    with tf.variable_scope(name_or_scope=None, default_name="dcn_m_moe"):
        x0 = x
        x_l = x0
        for i in range(num_layers):
            experts = []
            for j in range(num_experts):
                u_l = tf.get_variable(
                    "u_l_{}_expert_{}".format(i, j), shape=[dim, x.shape[1]],
                    initializer=tf.initializers.glorot_normal)
                v_l = tf.get_variable(
                    "v_l_{}_expert_{}".format(i, j), shape=[x0.shape[1], dim],
                    initializer=tf.initializers.glorot_normal)
                b_l = tf.get_variable(
                    "b_l_{}_expert_{}".format(i, j), shape=[1, x0.shape[1]],
                    initializer=tf.initializers.zeros)
                x_ = x0 * g_func(tf.matmul(g_func(tf.matmul(x_l, v_l)), u_l)) + b_l
                tf.layers.dense(inputs=mlp(x_l, gate_func_units_list[:-1]) * x_,
                                units=1, activation=tf.sigmoid)
                experts.append(tf.layers.dense(
                    inputs=mlp(x_l, gate_func_units_list[:-1]) * x_,
                    units=1, activation=tf.sigmoid))
            x_l = sum(experts) + x_l
    return x_l


def mlp(x, hidden_sizes, activation=tf.nn.relu, use_bias=True):
    """
    :param x: 2-D tensor
    :param hidden_sizes: list of int
    :param activation: activation
    :param use_bias: use_bias or not
    """
    if len(hidden_sizes) == 0:
        return x
    assert isinstance(hidden_sizes, list), "hidden_sizes must be list"
    with tf.variable_scope(name_or_scope=None, default_name="mlp"):
        for units in hidden_sizes:
            x = tf.layers.dense(x, units=units, activation=activation, use_bias=use_bias)
    return x


if __name__ == '__main__':
    x = tf.random.normal(shape=[1, 3])
    x_ori = tf.identity(x)
    x1 = dcn(x, 5, name_or_scope="dcn")
    x2 = dcn_faster(x, 5, name_or_scope="dcn", reuse=True)

    with tf.Session() as sess:
        print(list(tf.global_variables()))
        sess.run(tf.global_variables_initializer())
        print(sess.run([x1, x2]))
