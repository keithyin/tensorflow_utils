from __future__ import print_function

import tensorflow as tf


def single_feature_value_str(single_feature):
    vals = []
    if len(single_feature.int64_list.value) != 0:
        for v in single_feature.int64_list.value:
            vals.append(v)
        return ",".join(map(str, vals)), "__int64"

    if len(single_feature.float_list.value) != 0:
        for v in single_feature.float_list.value:
            vals.append(v)
        return ",".join(map(str, vals)), "__float"
    return None, None


record_iterator = tf.python_io.tf_record_iterator(path="eval.data")

if __name__ == '__main__':

    for i, string_record in enumerate(record_iterator):
        example = tf.train.Example()
        example.ParseFromString(string_record)

        print(example)

        for k, v in example.features.feature.items():
            v, t = single_feature_value_str(v)
            print("{}{}:{}".format(k, t, v))

        if i > 3:
            break
