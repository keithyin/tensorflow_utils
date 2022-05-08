from .ops import op_call


def call(context, dag_cfg):
    """
    dag core
    Args:
        context: dict. feature_name->tensor
        dag_cfg: list, list of op cfgs. the op must be registered in dag_core.ops.py
        [
            {
                "name": "UserProfile",
                "op": "mlp",
                "inputs": [
                    "gender",
                    "age",
                    "is_white_collar",
                    "income",
                    "resident_city"
                ]
            },
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
        ]
    Returns: context
    """

    for cfg in dag_cfg:
        context = op_call(context, cfg)
    return context
