"""
Multi-Task Learning
"""

from __future__ import print_function
import tensorflow as tf
from .mlp import n_experts, n_experts_v2, n_experts_v3


# https://dl.acm.org/doi/pdf/10.1145/3219819.3220007    50000iter 10s
def mmoe(x, num_experts, num_tasks, expert_hidden_sizes, task_specific_hidden_sizes, name_or_scope=None):
    """
    mmoe
    Args:
        x: [b, dim]
        num_experts: int
        num_tasks:  int
        expert_hidden_sizes: list of int
        task_specific_hidden_sizes:  list of int
        name_or_scope:
    Returns:
        (x, gates)
            x: [n, dim, num_task]
            gates: [n, num_task, num_experts]
    """
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
    """
    mmoe
    Args:
        x: [b, dim]
        num_experts: int
        num_tasks:  int
        expert_hidden_sizes: list of int
        task_specific_hidden_sizes:  list of int
        name_or_scope:
    Returns:
        (x, gates)
            x: [n, dim, num_task]
            gates: [n, num_task, num_experts]
    """
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
    """
    mmoe
    Args:
        x: [b, dim]
        num_experts: int
        num_tasks:  int
        expert_hidden_sizes: list of int
        task_specific_hidden_sizes:  list of int
        name_or_scope:
    Returns:
        (x, gates)
            x: [n, num_tasks, dim]
            gates: [n, num_tasks, num_experts]
    """
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="mmoe"):
        gate_w = tf.get_variable(name="gate_w", shape=[num_experts, num_tasks, x.shape[1]],
                                 initializer=tf.initializers.glorot_normal)
        # n, num_experts, num_tasks
        gate = tf.math.softmax(tf.einsum("ni,eti->net", x, gate_w), axis=1)

        # [n, num_experts, dim]
        experts = n_experts_v3(x, expert_hidden_sizes, num_experts)

        # [n, num_tasks, dim]
        x = tf.einsum("net,ned->ntd", gate, experts)

        # [n, num_tasks, dim]
        x = n_experts_v3(x, hidden_sizes=task_specific_hidden_sizes, num_experts=num_tasks,
                         last_activation=None)
        gate = tf.einsum("net->nte", gate)
    return x, gate


def cgc(x, num_experts, num_tasks, expert_hidden_sizes, is_first_cgc_layer, is_last_cgc_layer, name_or_scope):
    """

    Args:
        x: [b, dim]
        num_experts:
        num_tasks:
        expert_hidden_sizes:
        is_first_cgc_layer:
        is_last_cgc_layer:
    Returns:

    """
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="cgc"):

        # [n, num_experts, dim]
        expert_groups = [n_experts_v3(x, expert_hidden_sizes, num_experts=num_experts) for _ in range(num_tasks + 1)]
        shared_experts = [expert_groups[0:1]] * num_tasks
        tasks_experts = expert_groups[1:]
        for task_and_shared_experts in zip(shared_experts, tasks_experts):
            # [n, num_e, dim]
            se = task_and_shared_experts[0]
            te = task_and_shared_experts[1]


    pass


def ple(x, num_experts, num_tasks, expert_hidden_sizes, task_specific_hidden_sizes, num_cgc_layers, name_or_scope=None):
    """

    Args:
        x:
        num_experts:
        num_tasks:
        expert_hidden_sizes:
        task_specific_hidden_sizes:
        num_cgc_layers:
        name_or_scope:

    Returns:

    """
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="progressive_layered_extraction"):

        pass
    pass
