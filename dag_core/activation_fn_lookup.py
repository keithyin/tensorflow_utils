import tensorflow as tf
import functools

ACTIVATION_FNS = {
    "relu": tf.nn.relu,
    "relu6": tf.nn.relu6,
    "None": lambda x: x
}


def get_activation_fn(act_fn_cfg):
    """
    get activation function
    Args:
        act_fn_cfg: "FN_NAME" or {"name": "FN_NAME", "params": {"arg1": "arg1v", "arg2": "arg2v"}}
    Returns: activation function

    """
    act_fn_params = None
    if isinstance(act_fn_cfg, dict):
        act_fn_name = act_fn_cfg["name"]
        act_fn_params = act_fn_cfg["params"]
    else:
        act_fn_name = act_fn_cfg
    return _get_activation_fn(act_fn_name, act_fn_params)


def _get_activation_fn(name, params=None):
    assert params is None or isinstance(params, dict)
    fn = ACTIVATION_FNS[name]
    if params is not None:
        fn = functools.partial(fn, **params)
    return fn
