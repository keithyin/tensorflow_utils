import tensorflow as tf
from ..net_building_blocks import mtl


class MmoeParam(object):
    def __init__(self, cfg):
        """
        Args:
            cfg:
                {
                    "num_experts": 3,
                    "num_tasks": 2,
                    "expert_hidden_sizes": [16, 16],
                    "task_specific_hidden_sizes": [12, 1]
                }
        """
        self.num_experts = cfg["num_experts"]
        self.num_tasks = cfg["num_tasks"]
        self.expert_hidden_sizes = cfg["expert_hidden_sizes"]
        self.task_specific_hidden_sizes = cfg["task_specific_hidden_sizes"]
        pass


def mmoe_op(x, param, name_or_scope):
    assert isinstance(param, MmoeParam)
    x = tf.concat(x, axis=1)
    x, gate = mtl.mmoe_v3(
        x=x, num_tasks=param.num_tasks, num_experts=param.num_experts,
        expert_hidden_sizes=param.expert_hidden_sizes,
        task_specific_hidden_sizes=param.task_specific_hidden_sizes,
        name_or_scope=name_or_scope)
    return {name_or_scope: x, "{}_gate".format(name_or_scope): gate}
