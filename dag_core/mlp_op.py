import tensorflow as tf
from .activation_fn_lookup import get_activation_fn


class MlpParam(object):
    """
    {
        "name": "",
        "op": "mlp",
        "params": {
            "hidden_sizes": [],
            "act_fns": [],
            "use_bias": [1, 1, 0]
            },
        "inputs": ["inp1", "inp2", ...]
    }
    """
    def __init__(self, params):
        self.hidden_sizes = params["hidden_sizes"]
        self.activation_fns = [get_activation_fn(cfg) for cfg in params["act_fns"]]
        assert len(self.hidden_sizes) == len(self.activation_fns)

    def layer_info(self):
        return zip(self.hidden_sizes, self.activation_fns)


def mlp_op(x, param, name_or_scope):
    assert isinstance(param, MlpParam)
    with tf.variable_scope(name_or_scope="MLP_{}".format(name_or_scope)):
        assert isinstance(x, list)
        x = tf.concat(x, axis=1)
        for hidden_size, fn in param.layer_info():
            x = fn(tf.layers.dense(x, units=hidden_size))
    return x

