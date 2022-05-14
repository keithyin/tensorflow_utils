# coding=utf-8
import tensorflow as tf


def get_seq_feature_mask(feature_name, seq_tensor, seq_fea_mask_tensor_dict, seq_fea_len_tensor_dict):
    """
    get_seq_feature_mask
    Args:
        feature_name: feature name of sequence feature
        seq_tensor: feature tensor of sequence feature
        seq_fea_mask_tensor_dict: dict
        seq_fea_len_tensor_dict: dict

    Returns:
        mask tensor. [b, T]. for example [[1, 1, 0], [1, 1, 1]]
    """
    assert feature_name in seq_fea_mask_tensor_dict or feature_name in seq_fea_len_tensor_dict
    if feature_name in seq_fea_len_tensor_dict:
        sequence_mask = tf.sequence_mask(
            seq_fea_len_tensor_dict[feature_name],
            maxlen=seq_tensor.shape[1],
            dtype=tf.float32)
    else:
        sequence_mask = tf.cast(seq_fea_mask_tensor_dict[feature_name], dtype=tf.float32)
    return sequence_mask


def seq_mean_pooling_op(x, param, name_or_scope, context=None, feat_name_or_names=None):
    """

    Args:
        x: tensor, [b, T, dim]
        param:
        name_or_scope: string
        context: dict of tensor
        feat_name_or_names: string
    Returns:

    """
    with tf.name_scope("SMP_{}".format(name_or_scope)):
        seq_fea_mask_tensor_dict = context["seq_fea_mask_tensor_dict"]
        seq_fea_len_tensor_dict = context["seq_fea_len_tensor_dict"]
        mask = get_seq_feature_mask(feature_name=feat_name_or_names, seq_tensor=x,
                                    seq_fea_mask_tensor_dict=seq_fea_mask_tensor_dict,
                                    seq_fea_len_tensor_dict=seq_fea_len_tensor_dict)
        mask = mask / mask.shape[1].value
        seq_tensor = tf.einsum("nt,ntd->nd", mask, x)
    return seq_tensor


def seq_mean_pooling_group_op(x, param, name_or_scope, context=None, feat_name_or_names=None):
    """
        多个序列特征做 mean pooling，然后结果 concat 一起。
        {
            "name": "SomeName",
            "op": "seq_mean_pooling_group",

        }
    Args:
        x: list of tensor
        param:
        name_or_scope:
        context: dict of tensor
        feat_name_or_names: list of string
    Returns:

    """
    with tf.name_scope("SMPG_{}".format(name_or_scope)):
        res = []
        for fea_tensor, fea_name in zip(x, feat_name_or_names):
            res.append(seq_mean_pooling_op(fea_tensor, param, name_or_scope, context, fea_name))
        res = tf.concat(res, axis=1)
    return res

