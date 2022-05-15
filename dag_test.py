from __future__ import print_function

import tensorflow as tf
import unittest
import json
from dag_core import core


class Test(unittest.TestCase):
    def test_op(self):
        with open("cfgs/dag_cfg.json") as f:
            cfg = json.load(f)
        context = {
            "user_id": tf.constant([[1.0, 0.5], [2.1, -0.5]]),
            "user_name": tf.constant([[1.0, 0.5], [2.1, -0.5]]),
            "item_id": tf.constant([[1.0, 0.5], [2.1, -0.5]]),
            "item_name": tf.constant([[1.0, 0.5], [2.1, -0.5]]),

            "order_seq": tf.random_uniform(shape=[2, 4, 3]),
            "seq_fea_mask_tensor_dict": {},
            "seq_fea_len_tensor_dict": {"order_seq": tf.constant([2, 3])}
        }
        res = core.call(context, cfg["dag"])

        print(res["logit"], res["Order"])
