
from __future__ import print_function
import tensorflow as tf


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


def n_experts(x, hidden_sizes, num_experts, activation=tf.nn.relu, last_activation=tf.nn.sigmoid,
              use_bias=True, name_or_scope=None):
    """
    generate n experts net
    Args:
        x: input, [n, dim] or [n, dim, num_experts]
        hidden_sizes: list of hidden size
        num_experts: num_experts, if only 1 expert, using mlp() instead
        activation: activation function
        last_activation: activation of last layer
        use_bias: boolean,
        name_or_scope: string
    Returns:
        [n, hidden_size, num_experts]
    """
    assert len(x.shape) in (2, 3)
    if len(x.shape) == 2:
        x = tf.tile(tf.expand_dims(x, axis=-1), multiples=[1, 1, num_experts])
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="n_experts"):
        for i, units in enumerate(hidden_sizes):
            w = tf.get_variable("w_l_{}".format(i), shape=[x.shape[1], units, num_experts],
                                initializer=tf.initializers.glorot_normal)
            b = tf.get_variable("b_l_{}".format(i), shape=[1, units, num_experts],
                                initializer=tf.initializers.zeros) if use_bias else 0

            x = tf.einsum("nkg,khg->nhg", x, w) + b
            if i != (len(hidden_sizes) - 1):
                x = x if activation is None else activation(x)
            else:
                x = x if last_activation is None else last_activation(x)
    return x
