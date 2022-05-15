# coding=utf-8
import tensorflow as tf
from .batch_norm_layer import layer_norm_for_seq


# https://github.com/tensorflow/models/blob/6d458bcc40a378913f1dc949b49c9a38ec4c2465/official/nlp/modeling/layers/transformer.py
# https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/models


def multi_head_attn(q, k, v, v_mask, attn_units, output_units, num_heads, name_or_scope=None):
    """
    multi head attn
    Args:
        q: [b, T_q, dim]. k 和 v的 时间步需要是一致的，但是 q 可以和下面两个不一致。
            比如非 self-attn 的情况，q 有可能只是一个 时间步的 query
        k: [b, T, dim]
        v: [b, T, D_v]
        v_mask: [b, T], mask of k and v. [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
        attn_units: int
        output_units: 输出的维度
        num_heads: int
        name_or_scope: string,
    Returns:
        tuple:([b, T_q, output_units], [b, T_q, num_heads, T])
        [b, T_q, output_units] 这个并没有对于 q 的时间步 进行 mask 结果。
        [b, T_q, num_heads, T] 这个 shape 更容易观测 不同头的 不同 attn 位置。
    """
    tot_attn_units = num_heads * attn_units
    bs = tf.shape(k)[0]
    q_timestep = q.shape[1]
    timestep = k.shape[1]
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="MultiHeadAttn"):
        # [b, T_q, num_heads * attn_units]
        q = tf.layers.dense(q, units=tot_attn_units)
        # [b, T, num_heads * attn_units]
        k = tf.layers.dense(k, units=tot_attn_units)
        v = tf.layers.dense(v, units=tot_attn_units)

        # [b, T, num_heads, attn_units]
        q = tf.reshape(q, shape=[bs, q_timestep, num_heads, attn_units])
        k = tf.reshape(k, shape=[bs, timestep, num_heads, attn_units])
        v = tf.reshape(v, shape=[bs, timestep, num_heads, attn_units])
        # [b, num_head, q_timestep, T]
        weights = tf.einsum("bihd,bjhd->bhij", q, k) * (attn_units ** -0.5)
        # [b, T] valid timestep is 0, invalid timestep is -1e8
        v_mask = tf.reshape((v_mask - 1) * 1e8, shape=[bs, 1, 1, timestep])
        v_mask = tf.tile(v_mask, multiples=[1, num_heads, q_timestep, 1])
        weights = weights + v_mask
        # [b, num_heads, q_timestep, T]
        weights = tf.nn.softmax(weights, axis=-1)
        v = tf.einsum("bhij,bjhd->bihd", weights, v)
        v = tf.reshape(v, shape=[bs, q_timestep, tot_attn_units])
        # [b, q_timestep, output_units]
        v = tf.layers.dense(v, units=output_units)
        weights = tf.einsum("bhij->bihj", weights)
    return v, weights


def dense_relu_dense(x, filter_size, output_size):
    with tf.variable_scope(name_or_scope=None, default_name="DRD"):
        x = tf.layers.dense(x, units=filter_size, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=output_size)
    return x


def attention(q, k, v, v_mask, attn_units, num_heads, ffn_filter_size, q_mask=None, name_or_scope=None):
    """
    inp -> multi_head_attn -> residual + layer_norm -> ffn -> residual + layer_norm
    Args:
        q: [b, T_q, D_q]
        k: [b, T, d_k]
        v: [b, T, d_v]
        v_mask: [b, T].  k 和 v 的 mask。 [[1, 1, 0], [1, 0, 0]]
        attn_units: int
        num_heads: int
        ffn_filter_size: int
        q_mask: q 的 mask
        name_or_scope: string or scope
    Returns:
        [b, T, d_v]
    """
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="Attn"):
        x, attn_weights = multi_head_attn(
            q, k, v, v_mask, attn_units, output_units=v.shape[-1], num_heads=num_heads, name_or_scope=name_or_scope)
        x = v + x
        x = layer_norm_for_seq(x)
        x = x + dense_relu_dense(x, ffn_filter_size, output_size=x.shape[-1])
        x = layer_norm_for_seq(x)
        if q_mask is not None:
            x = tf.einsum("bt,btd->btd", q_mask, x)
    return x, attn_weights


def self_attention(x, mask, attn_units, num_heads, ffn_filter_size, name_or_scope=None):
    """
        Args:
            x: [b, T, D]
            mask: [b, T].  [[1, 1, 0], [1, 0, 0]]
            attn_units: int
            num_heads: int
            ffn_filter_size: int
            name_or_scope: string or scope
        Returns:
            tuple. ([b, T, D], [b, h, T, T])
        """
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="SelfAttn"):
        x, attn_weights = attention(
            x, x, x, v_mask=mask, attn_units=attn_units, num_heads=num_heads, ffn_filter_size=ffn_filter_size,
            q_mask=mask, name_or_scope=name_or_scope)
        return x, attn_weights

