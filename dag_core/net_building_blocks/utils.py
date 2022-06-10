import tensorflow as tf


@tf.custom_gradient
def gradient_clip_layer(x, clip_min_value, clip_max_value):
    def grad(dy):
        return tf.clip_by_value(dy, clip_value_min=clip_min_value, clip_value_max=clip_max_value)
    return tf.identity(x), grad
