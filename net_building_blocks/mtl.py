"""
Multi-Task Learning
"""

from __future__ import print_function
import tensorflow as tf


# https://dl.acm.org/doi/pdf/10.1145/3219819.3220007
def mmoe(x, num_experts, num_tasks, expert_hidden_sizes, task_specific_hidden_sizes, name_or_scope=None):
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="mmoe"):
        gate_w = tf.get_variable(name="gate_w", shape=[x.shape[1], num_tasks, num_experts],
                                 initializer=tf.initializers.glorot_normal)
        # n, num_task, num_expert
        gate = tf.math.softmax(tf.einsum("ni,ijk->njk", x, gate_w), axis=-1)


    pass
