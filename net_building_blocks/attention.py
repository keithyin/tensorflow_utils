import tensorflow as tf
from .batch_norm_layer import layer_norm_for_seq
# https://github.com/tensorflow/models/blob/6d458bcc40a378913f1dc949b49c9a38ec4c2465/official/nlp/modeling/layers/transformer.py
# https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/models


def multi_head_attn(q, k, v, mask, attn_units, num_heads, name_or_scope):
    """
    multi head attn
    Args:
        q: [b, T, dim]
        k: [b, T, dim]
        v: [b, T, dim]
        mask: [b, T]
        attn_units: int
        num_heads: int
        name_or_scope: string

    Returns:

    """
    ori_v = v
    tot_attn_units = num_heads * attn_units
    out_units = k.shape[2]
    bs = tf.shape(k)[0]
    timestep = k.shape[1]
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="MultiHeadAttn"):
        # [b, T, num_heads * attn_units]
        q = tf.layers.dense(q, units=tot_attn_units)
        k = tf.layers.dense(k, units=tot_attn_units)
        v = tf.layers.dense(v, units=tot_attn_units)

        # [b, T, num_heads, attn_units]
        q = tf.reshape(q, shape=[bs, timestep, num_heads, attn_units])
        k = tf.reshape(k, shape=[bs, timestep, num_heads, attn_units])
        v = tf.reshape(v, shape=[bs, timestep, num_heads, attn_units])
        # [b, num_head, T, T]
        weights = tf.einsum("bihd,bjhd->bhij") * (attn_units ** -0.5)
        # [b, T] valid timestep is 0, invalid timestep is -1e8
        ori_mask = mask
        mask = tf.reshape((mask - 1) * 1e8, shape=[bs, 1, 1, timestep])
        mask = tf.tile(mask, multiples=[1, num_heads, timestep, 1])
        weights = weights + mask
        # [b, num_heads, T, T]
        weights = tf.nn.softmax(weights, axis=-1)
        v = tf.einsum("bhij,bjhd->bihd", weights, v)
        v = tf.reshape(v, shape=[bs, timestep, tot_attn_units])
        v = tf.layers.dense(v, units=out_units)
        # residual
        v = v + ori_v
        v = tf.einsum("bt,btd->btd", ori_mask, v)
        # layer norm
        v = layer_norm_for_seq(v)
    return v


def dense_relu_dense(x, filter_size, output_size):
    with tf.variable_scope(name_or_scope=None, default_name="DRD"):
        x = tf.layers.dense(x, units=filter_size, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=output_size)
    return x


def attention(q, k, v, mask, attn_units, num_heads, ffn_filter_size, name_or_scope):
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="ATTN"):
        x = multi_head_attn(q, k, v, mask, attn_units, num_heads, name_or_scope=name_or_scope)
        x = x + dense_relu_dense(x, ffn_filter_size, output_size=x.shape[-1])
        x = layer_norm_for_seq(x)
    return x
