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
                "inputs": [
                    "view_coupon_id",
                    "view_coupon_amount",
                    "view_coupon_limit",
                    "days_left",
                    "coupon_name",
                    "channel_template_id"
                ]
            }

    Returns:

    """
    op_name = cfg["op"]
    out_name = cfg["name"]
    params = cfg["params"]
    input_names = cfg["inputs"]

    inputs = [context[name] for name in input_names]
    params = OPS[op_name]["param_class"](params)

    # op support two return type. tensor or dict[name->tensor]
    res = OPS[op_name]["op"](x=inputs, param=params, name_or_score=out_name)

    if isinstance(res, dict):
        for k, v in res.items():
            if k in context:
                raise ValueError("duplicated name : {}".format(k))
            context[k] = v

    else:
        if out_name in context:
            raise ValueError("duplicated name : {}".format(out_name))
        context[out_name] = res

    return res



