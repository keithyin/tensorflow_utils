# coding=utf-8
from .mlp_op import mlp_op, MlpParam
from .mtl_ops import mmoe_op, MmoeParam


OPS = {
    "mlp": {"op": mlp_op, "param_class": MlpParam},
    "mmoe": {"op": mmoe_op, "param_class": MmoeParam}
}


def op_call(context, cfg):
    """
    Args:
        context: dict, feature_name -> tensor
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

    """
    op_name = cfg["op"]
    out_name = cfg["name"]
    params = cfg["params"]
    input_tensors = {}
    for k, names in cfg["inputs"].items():
        input_tensors[k] = [context[name] for name in names]

    params = OPS[op_name]["param_class"](params)
    op_kwargs = {"param": params, "name_or_score": out_name}
    op_kwargs.update(input_tensors)
    # op support two return type. tensor or dict[name->tensor]
    # op 的签名应该是，多个输入 tensor，一个 param 参数，一个 name_or_scope 参数
    # op 的返回值 dict of tensor or tensor
    res = OPS[op_name]["op"](**op_kwargs)

    if isinstance(res, dict):
        for name, tensor in res.items():
            if name in context:
                raise ValueError("duplicated name : {}".format(name))
            context[name] = tensor

    else:
        if out_name in context:
            raise ValueError("duplicated name : {}".format(out_name))
        context[out_name] = res

    return res



