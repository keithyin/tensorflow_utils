"""
Multi-Task Learning
"""

from __future__ import print_function
import tensorflow as tf
from tensorflow import layers
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
        gate_w = tf.get_variable(
            name="gate_w", shape=[x.shape[1], num_tasks, num_experts],
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
        gate_w = tf.get_variable(
            name="gate_w", shape=[x.shape[1], num_tasks, num_experts],
            initializer=tf.initializers.glorot_normal)
        # n, num_tasks, num_experts
        gate = tf.math.softmax(tf.einsum("ni,ijk->njk", x, gate_w), axis=-1)

        # [n, dim, num_experts]
        experts = n_experts_v2(x, expert_hidden_sizes, num_experts)

        # [n, dim, num_tasks]
        x = tf.einsum("nte,nde->ndt", gate, experts)

        # [n, dim, num_tasks]
        x = n_experts_v2(
            x, hidden_sizes=task_specific_hidden_sizes, num_experts=num_tasks,
            last_activation=tf.nn.sigmoid)
    return x, gate


# https://dl.acm.org/doi/pdf/10.1145/3219819.3220007   50000iter 9.45
def mmoe_v3(x, num_experts, num_tasks, expert_hidden_sizes, task_specific_hidden_sizes,
            task_specific_inputs=None, name_or_scope=None):
    """
    mmoe
    Args:
        x: [b, dim]
        num_experts: int
        num_tasks:  int
        expert_hidden_sizes: list of int
        task_specific_hidden_sizes:  list of int
        task_specific_inputs: concat to task_specific branch. tensor [b, num_task, some_dim]
        name_or_scope:
    Returns:
        (x, gates)
            x: [n, num_tasks, dim]
            gates: [n, num_tasks, num_experts]
    """
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="mmoe"):
        gate_w = tf.get_variable(
            name="gate_w", shape=[num_experts, num_tasks, x.shape[1]],
            initializer=tf.initializers.glorot_normal)
        # n, num_experts, num_tasks
        gate = tf.math.softmax(tf.einsum("ni,eti->net", x, gate_w), axis=1)

        # [n, num_experts, dim]
        experts = n_experts_v3(x, expert_hidden_sizes, num_experts)

        # [n, num_tasks, dim]
        x = tf.einsum("net,ned->ntd", gate, experts)
        if task_specific_inputs is not None:
            x = tf.concat([x, task_specific_inputs], axis=2)
        # [n, num_tasks, dim]
        x = n_experts_v3(
            x, hidden_sizes=task_specific_hidden_sizes, num_experts=num_tasks,
            last_activation=None)
        gate = tf.einsum("net->nte", gate)
    return x, gate


def cgc(input_list, num_experts, num_tasks, expert_hidden_sizes, name_or_scope=None):
    """

    Args:
        input_list: list of [b, dim] tensor, len(input_list) == num_tasks + 1. [shared_inp, t0_inp, t1_inp, ...]
        num_experts:
        num_tasks:
        expert_hidden_sizes:
        name_or_scope:
    Returns:
        list of tensors [b, dim].   [expert_output, t0_output, t1_output, ...]
    """
    assert len(input_list) == num_tasks + 1
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="CGC"):
        # [n, num_experts, dim]
        expert_groups = [
            n_experts_v3(input_list[i], expert_hidden_sizes, num_experts=num_experts) for i in range(num_tasks + 1)]
        shared_experts = [expert_groups[0]] * num_tasks
        tasks_experts = expert_groups[1:]

        # task output
        tasks_outputs = []
        for i, task_and_shared_experts in enumerate(zip(shared_experts, tasks_experts), start=1):
            # [n, 2*num_e, dim]
            e = tf.concat(task_and_shared_experts, axis=1)
            # [n, 2*num_e]
            gate = layers.dense(input_list[i], units=2 * num_experts, use_bias=False, activation=tf.nn.softmax)
            out = tf.einsum("ne,ned->nd", gate, e)
            tasks_outputs.append(out)

        # expert_out
        all_experts = tf.concat(expert_groups, axis=1)
        shared_exp_gates = layers.dense(input_list[0], units=(num_tasks + 1) * num_experts,
                                        activation=tf.nn.softmax, use_bias=False)
        expert_output = tf.einsum("ne,ned->nd", shared_exp_gates, all_experts)
    return [expert_output] + tasks_outputs


def ple(x, num_experts, num_tasks, expert_hidden_sizes, task_specific_hidden_sizes, num_cgc_layers,
        task_specific_inputs=None, name_or_scope=None):
    """

    Args:
        x: [b, dim]
        num_experts: int value
        num_tasks: int value
        expert_hidden_sizes: list
        task_specific_hidden_sizes: list
        num_cgc_layers: int
        task_specific_inputs: [b, num_t, dim]
        name_or_scope: string

    Returns:
        tensor [n, num_tasks]
    """
    x = [x] * (num_tasks + 1)
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="PLE"):
        for _ in range(num_cgc_layers):
            x = cgc(x, num_experts, num_tasks, expert_hidden_sizes)

        # [b, num_t, dim]
        x = tf.stack(x[1:], axis=1)
        if task_specific_inputs is not None:
            x = tf.concat([x, task_specific_inputs], axis=2)
        logit = n_experts_v3(x, hidden_sizes=task_specific_hidden_sizes, num_experts=num_tasks, last_activation=None)
        logit = tf.squeeze(logit, axis=-1)
    return logit
