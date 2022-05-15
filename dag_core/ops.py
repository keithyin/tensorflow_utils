# coding=utf-8
from .mlp_op import mlp_op, MlpParam
from .mtl_ops import mmoe_op, MmoeParam
from .seq_ops import seq_mean_pooling_op, seq_mean_pooling_group_op
import inspect

"""
这里对于 OP 的定义的是一个比 tensorflow 更加高层抽象的 op，实际上其含义更像是 pytorch 的 module。
关于 op 的调用配置。
{
    "name": "",
    "op": "mlp",
    "params": {
        "hidden_sizes": [],
        "act_fns": [],
        "use_bias": [1, 1, 0]
    },
    "inputs": {"x": ["t1", "t2"]}
}

OP会被定义为一个方法。
    * 入参： 
        param (用来接受该op 的非 tensor 的参数的)。每个 op 的 param 对应与一个 类。用来解析 op 调用配置的 params 字段。
        name_or_score: 表明该 op 的 name_or_scope。咋 func op_call 中会使用 name 作为其 name_or_scope,
        context: 接受上下文的dict。方便传入其它上下文数据
        其它：其它的名字其实对应与 op 调用配置中的 inputs 中的 tensor 了。
    * 返回值：可以有两种返回值，一个是 dict，另外一个是 tensor
"""
OPS = {
    "mlp": {"op": mlp_op, "param_class": MlpParam},
    "mmoe": {"op": mmoe_op, "param_class": MmoeParam},
    "seq_mean_pooling": {"op": seq_mean_pooling_op, "param_class": lambda x: x},
    "seq_mean_pooling_group": {"op": seq_mean_pooling_group_op, "param_class": lambda x: x}
}


def op_call(context, cfg):
    """
    Args:
        context: dict, feature_name -> tensor。 并不会出现 inplace 修改
        cfg:
            {
                "name": "Candidate",
                "op": "mlp",
                "params": {"":  [1, 2, 3]},
                "inputs": {
                    "x": ["view_coupon_id",
                            "view_coupon_amount",
                            "view_coupon_limit",
                            "days_left",
                            "coupon_name",
                            "channel_template_id"
                        ],
                    ""
                }
                ]
            }
            name: 计算后放在context 中的名字
            op: op 的名字，该名字应该在 ops.py 中注册
            params: 非 tensor 的参数放在这里
            inputs: tensor 相关的配置放在这里

    Returns:
        context
    """
    assert isinstance(context, dict)
    context = context.copy()
    op_name = cfg["op"]
    out_name = cfg["name"]
    params = cfg.get("params", None)
    input_tensors = {}

    """
    如果 "inputs": {"x": ["t1", "t2"]} 是这种配置， 会构建一个 list 传给下游。
    如果 "inputs": {"x": "t1"} 是这种配置， 则就是一个 tensor 传给下游了。
    """
    for k, names in cfg["inputs"].items():
        if isinstance(names, list):
            input_tensors[k] = [context[name] for name in names]
        else:
            input_tensors[k] = context[names]

    params = OPS[op_name]["param_class"](params)

    op_func = OPS[op_name]["op"]
    op_kwargs = {"param": params, "name_or_scope": out_name, "context": context}

    # 这个是当前因为 seq_mean_pooling_op 需要 tensor 的名字，所以这里开了个口子。通过这种方式传过去
    if "feat_name_or_names" in inspect.getargspec(op_func).args:
        op_kwargs["feat_name_or_names"] = cfg["inputs"]["x"]

    op_kwargs.update(input_tensors)

    # op support two return type. tensor or dict[name->tensor]
    # op 的签名应该是，多个输入 tensor，一个 param 参数，一个 name_or_scope 参数
    # op 的返回值 dict of tensor or tensor
    res = op_func(**op_kwargs)

    if isinstance(res, dict):
        for name, tensor in res.items():
            if name in context:
                raise ValueError("duplicated name : {}".format(name))
            context[name] = tensor

    else:
        if out_name in context:
            raise ValueError("duplicated name : {}".format(out_name))
        context[out_name] = res

    return context



