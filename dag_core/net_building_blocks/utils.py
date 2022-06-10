# coding=utf-8
import tensorflow as tf

# https://stackoverflow.com/questions/54047604/how-to-assign-custom-gradient-to-tensorflow-op-with-multiple-inputs
# https://stackoverflow.com/questions/43839431/tensorflow-how-to-replace-or-modify-gradient


@tf.custom_gradient
def gradient_clip_layer(x, clip_value_min, clip_value_max):
    """
    网络前向过程中调用该方法的话，x 反向的时候梯度会被剪切。
    注意，不哟啊 gradient_clip_layer(a, clip_value_min=0.1, clip_value_max=0.1) 这么调用。
        这么调用的会报错 The custom_gradient decorator currently supports keywords arguments only when eager execution
        is enabled.
    要 gradient_clip_layer(a, 0.1, 0.1)这么调用就可以了
    Args:
        x:
        clip_value_min:
        clip_value_max:

    Returns:

    """

    def grad(dy):
        return [tf.clip_by_value(dy, clip_value_min=clip_value_min, clip_value_max=clip_value_max),
                tf.constant(0.), tf.constant(0.)]
    return tf.identity(x), grad
