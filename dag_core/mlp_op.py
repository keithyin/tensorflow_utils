import tensorflow as tf
from .activation_fn_lookup import get_activation_fn


class MlpParam(object):
    """
    {
        "name": "UserProfile",
        "op": "mlp",
        "params": {
            "hidden_sizes": [4, 6, 8],
            "act_fns": ["relu", "relu", "None"],
            "use_bias": [true, true, false]
            },
        "inputs": {
            "x": ["inp1", "inp2", ...]
        }
    }
    """
    def __init__(self, params):
        self.hidden_sizes = params["hidden_sizes"]
        self.activation_fns = [get_activation_fn(cfg) for cfg in params["act_fns"]]
        self.use_bias = params["use_bias"]
        assert len(self.hidden_sizes) == len(self.activation_fns)
        assert len(self.hidden_sizes) == len(self.use_bias)

    def layer_info(self):
        return zip(self.hidden_sizes, self.activation_fns, self.use_bias)


def mlp_op(x, param, name_or_scope, context=None):
    assert isinstance(param, MlpParam)
    with tf.variable_scope(name_or_scope="MLP_{}".format(name_or_scope)):
        if isinstance(x, list):
            x = tf.concat(x, axis=1)
        for hidden_size, fn, use_bias in param.layer_info():
            x = fn(tf.layers.dense(x, units=hidden_size, use_bias=use_bias))
    return x

