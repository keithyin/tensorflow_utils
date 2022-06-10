import tensorflow as tf


@tf.custom_gradient
def gradient_clip_layer(x):
    def grad(dy):
        return tf.clip_by_value(dy, clip_value_min=-0.1, clip_value_max=0.1)
    return tf.identity(x), grad
