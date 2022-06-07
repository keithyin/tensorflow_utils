import tensorflow as tf


def compute_metric(metric_fn, inp_kv, summary_name):
    """
    compute metric & add update op to collection & summary metric
    Args:
        metric_fn: tf.metrics.mean
        inp_kv: kv of metric_fn's params
        summary_name: str

    Returns:

    """
    metric = metric_fn(**inp_kv)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, metric[1])
    tf.summary.scalar(summary_name, metric[0])
    return metric



