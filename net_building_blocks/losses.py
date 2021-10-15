import numpy as np
import tensorflow as tf


def exponential_nll_loss(lam, label):
    """
    Args:
        lam:[B, 1] or [B] lambda of exponential distribution. lam must be bigger than 0 !!!!
        label: [B, 1] or [B] x of exponential distribution
    Returns: [B, 1] or [B] loss
    """
    lam = tf.clip_by_value(lam, clip_value_min=1e-6, clip_value_max=1e6)
    return lam * label - tf.log(lam)


def inverse_gaussian_nll_loss(lam, mu, label):
    """
    https://zh.wikipedia.org/wiki/%E9%80%86%E9%AB%98%E6%96%AF%E5%88%86%E5%B8%83
    Args:
        lam: [B, 1] inverse gaussian distribution lambda
        mu: [B, 1] inverse gaussian distribution mu
        label:  [B, 1] inverse gaussian distribution
    lam>0, mu>0, label>0
    Returns:[B, 1]
    """
    lam = tf.clip_by_value(lam, clip_value_min=1e-6, clip_value_max=1e6)
    mu = tf.clip_by_value(mu, clip_value_min=1e-6, clip_value_max=1e6)
    return 0.5 * (lam * tf.square(label - mu) / (tf.square(mu) * label)
                  + tf.log(2 * np.pi * tf.pow(label, 3.) - tf.log(lam)))
