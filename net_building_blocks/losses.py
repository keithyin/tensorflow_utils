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