# coding=utf-8
import tensorflow as tf
import os


class HdfsPusher(object):
    """
    往路径 ${root_dir}/${prefix}_${job_name}_${task_index} 覆写数据
    """
    def __init__(self, job_name, task_index, root_dir, prefix):
        self._job_name = job_name
        self._task_index = task_index
        self._root_dir = root_dir
        self._prefix = prefix
        pass

    def push_text(self, text):
        o_filename = os.path.join(self._root_dir, "{prefix}_{job_name}_{task_index}".format(
            prefix=self._prefix, job_name=self._job_name, task_index=self._task_index))

        if tf.gfile.Exists(o_filename):
            tf.gfile.DeleteRecursively(o_filename)
            tf.logging.info("delete:[{}]".format(o_filename))

        if not tf.gfile.Exists(self._root_dir):
            raise ValueError("manually create dir [{}] first".format(self._root_dir))

        writer = tf.gfile.Open(o_filename, mode="w")
        writer.write(text)
        writer.close()
        tf.logging.info("write text to [{}] DONE".format(o_filename))
        tf.logging.info("text: ***********\n {} \n **************".format(text))