"""
Multi-Task Learning
"""

from __future__ import print_function
import tensorflow as tf
from .mlp import n_experts, n_experts_v2, n_experts_v3


# https://dl.acm.org/doi/pdf/10.1145/3219819.3220007    50000iter 10s
def mmoe(x, num_experts, num_tasks, expert_hidden_sizes, task_specific_hidden_sizes, name_or_scope=None):
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="mmoe"):
        gate_w = tf.get_variable(name="gate_w", shape=[x.shape[1], num_tasks, num_experts],
                                 initializer=tf.initializers.glorot_normal)
        # n, num_tasks, num_experts
        gate = tf.math.softmax(tf.einsum("ni,ijk->njk", x, gate_w), axis=-1)

        # [n, dim, num_experts]
        experts = n_experts(x, expert_hidden_sizes, num_experts)

        # [n, dim, num_tasks]
        x = tf.einsum("nte,nde->ndt", gate, experts)

        # [n, dim, num_tasks]
        x = n_experts(x, hidden_sizes=task_specific_hidden_sizes, num_experts=num_tasks, last_activation=tf.nn.sigmoid)
    return x, gate


# https://dl.acm.org/doi/pdf/10.1145/3219819.3220007   50000iter 9.94
def mmoe_v2(x, num_experts, num_tasks, expert_hidden_sizes, task_specific_hidden_sizes, name_or_scope=None):
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="mmoe"):
        gate_w = tf.get_variable(name="gate_w", shape=[x.shape[1], num_tasks, num_experts],
                                 initializer=tf.initializers.glorot_normal)
        # n, num_tasks, num_experts
        gate = tf.math.softmax(tf.einsum("ni,ijk->njk", x, gate_w), axis=-1)

        # [n, dim, num_experts]
        experts = n_experts_v2(x, expert_hidden_sizes, num_experts)

        # [n, dim, num_tasks]
        x = tf.einsum("nte,nde->ndt", gate, experts)

        # [n, dim, num_tasks]
        x = n_experts_v2(x, hidden_sizes=task_specific_hidden_sizes, num_experts=num_tasks, last_activation=tf.nn.sigmoid)
    return x, gate


# https://dl.acm.org/doi/pdf/10.1145/3219819.3220007   50000iter 9.45
def mmoe_v3(x, num_experts, num_tasks, expert_hidden_sizes, task_specific_hidden_sizes, name_or_scope=None):
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="mmoe"):
        gate_w = tf.get_variable(name="gate_w", shape=[num_experts, num_tasks, x.shape[1]],
                                 initializer=tf.initializers.glorot_normal)
        # n, num_experts, num_tasks
        gate = tf.math.softmax(tf.einsum("ni,eti->net", x, gate_w), axis=-1)

        # [n, num_experts, dim]
        experts = n_experts_v3(x, expert_hidden_sizes, num_experts)

        # [n, num_tasks, dim]
        x = tf.einsum("net,ned->ntd", gate, experts)

        # [n, num_tasks, dim]
        x = n_experts_v3(x, hidden_sizes=task_specific_hidden_sizes, num_experts=num_tasks, last_activation=tf.nn.sigmoid)
    return x, gate
