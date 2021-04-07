import tensorflow as tf
import re
import numpy as np
from datetime import datetime
import os


def static_vocab_table_with_kv_init(keys, num_oov_buckets=10):
    table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys, dtype=tf.string),
            values=tf.constant(list(range(len(keys))), dtype=tf.int64)), num_oov_buckets=num_oov_buckets)
    return table


def number_of_trainable_parameters(filter_pattern):
    num_params = 0
    for param in tf.trainable_variables():
        # begin with filter_pattern
        if re.match(filter_pattern, param.op.name):
            num_params += np.prod(param.shape)
    return num_params


def split_to_feature_dict(name, feature_tensor):
    """
    we sometimes group a lot of fields as one feature field.
    this function is used to tare there apart
    :param name: string
    :param feature_tensor: Tensor
    """
    assert len(feature_tensor.shape) == 2, "{}".format(feature_tensor)
    num_sub_fields = feature_tensor.shape[1]
    fea_dict = {}
    for i in range(num_sub_fields):
        fea_dict["{}_{}".format(name, i)] = feature_tensor[:, i:i + 1]
    return fea_dict


def dict_2_str(inp):
    assert isinstance(inp, dict)
    o_str = ""
    for k, v in inp.items():
        o_str += "{}: {}\n".format(k, v)
    return o_str


class PredictUtil(object):
    def __init__(self, job_name, task_index, estimator, input_fn, output_path, result_to_str_func,
                 model_name,
                 checkpoint_path=None,
                 num_row_per_file=1e4,
                 predict_keys=None,
                 message_pusher=None,
                 logging_interval=1e4):
        assert isinstance(estimator, tf.estimator.Estimator)
        self._estimator = estimator
        self._job_name = job_name
        # model fn use this param build predict graph
        self._estimator._params['export_model'] = model_name

        self._input_fn = input_fn
        self._checkpoint_path = checkpoint_path
        self._output_path = output_path
        self._task_index = task_index
        self._predict_keys = predict_keys
        self._message_pusher = message_pusher
        self._logging_interval = logging_interval
        self._num_row_per_file = num_row_per_file
        self._result_to_str_func = result_to_str_func

    def predict(self):
        assert isinstance(self._estimator, tf.estimator.Estimator)
        part_idx = 0
        output_path_fmt = self._output_path + "/job_name_{}_task_idx_{:02d}-part-{:05d}"

        writer = tf.gfile.Open(output_path_fmt.format(self._job_name, self._task_index, part_idx), mode='w')

        begin_time = datetime.now()
        tf.logging.info("PREDICT: begin_time: {}".format(begin_time.strftime("%Y-%m-%d %H:%M:%S")))
        info_fmt = "PREDICT: pred_num: {}, consumed_time: {:.2f}s, speed: {:.4f}(ins/s), output_path: {}"
        for i, v in enumerate(self._estimator.predict(self._input_fn,
                                                      predict_keys=self._predict_keys,
                                                      checkpoint_path=self._checkpoint_path)):
            if (i+1) % self._num_row_per_file == 0:
                writer.flush()
                writer.close()
                part_idx += 1
                writer = tf.gfile.Open(output_path_fmt.format(self._job_name, self._task_index, part_idx), mode='w')

            if (i + 1) % self._logging_interval == 0:
                time_delta = datetime.now() - begin_time
                if time_delta.total_seconds() > 60:
                    info = info_fmt.format(i + 1, time_delta.total_seconds(),
                                           (i + 1) / float(time_delta.total_seconds()),
                                           self._output_path)
                    tf.logging.info(info)

            writer.write(self._result_to_str_func(v) + "\n")

        writer.flush()
        writer.close()

        end_time = datetime.now()
        time_delta = end_time - begin_time
        if time_delta.total_seconds() < 1:
            return

        info = info_fmt.format(i + 1, time_delta.total_seconds(), (i + 1) / float(time_delta.total_seconds()),
                               self._output_path)
        tf.logging.info(info)
        if self._message_pusher is not None:
            self._message_pusher.push_text(info)


def get_first_ele(inp):
    if isinstance(inp, np.ndarray):
        return inp.flatten()[0]

    if isinstance(inp, list):
        return inp[0]
    return inp
