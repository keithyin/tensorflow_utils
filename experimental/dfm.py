# coding=utf-8
from __future__ import print_function
import numpy as np
import datetime
import pandas as pd
import tensorflow as tf

ONE_DAY = datetime.timedelta(days=1)


def data_generate(date_begin, date_end, per_day_records=1):
    """
    Args:
        date_begin: 20211009
        date_end: 20211031
    Returns: [[event_date, cvr, delayed_days, cvt_date], ]
    """
    # today, cvr, cvt_date
    cur_day = datetime.datetime.strptime(str(date_begin), "%Y%m%d")
    end_day = datetime.datetime.strptime(str(date_end), "%Y%m%d")
    sample = []
    while cur_day <= end_day:
        cur_dates = [cur_day.strftime("%Y%m%d")] * per_day_records
        converts = np.random.binomial(1, p=0.2, size=per_day_records).tolist()
        delayed_days = np.random.exponential(scale=4, size=per_day_records).astype(np.int).tolist()
        cvt_date = [(cur_day + d * ONE_DAY).strftime("%Y%m%d") for d in delayed_days]
        sample.extend(list(zip(cur_dates, converts, delayed_days, cvt_date)))
        cur_day += ONE_DAY
    sample = pd.DataFrame(sample, columns=['event_date', 'cvt', 'delayed_days', 'cvt_date'])
    return sample


def train_data_generate(sample, train_date):
    """
    Args:
        sample: data get from data_generate
        train_date: '20211010'
    Returns:
    """
    train_date = str(train_date)
    assert isinstance(sample, pd.DataFrame)
    sample = sample[sample['event_date'] <= train_date].reset_index()
    records = []
    # label, delayed_days
    for _, record in sample.iterrows():
        if record['cvt_date'] > train_date or record['cvt'] == 0:
            label = 0
            delayed_day = (datetime.datetime.strptime(str(train_date), "%Y%m%d") -
                           datetime.datetime.strptime(record['event_date'], "%Y%m%d")).days
            records.append([label, delayed_day])
        else:
            records.append([record['cvt'], record['delayed_days']])
    return pd.DataFrame(records, columns=['label', 'delayed_days'])


def get_dataset(train_data):
    assert isinstance(train_data, pd.DataFrame)
    train_data = train_data.values
    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(train_data, dtype=tf.float32))
    dataset = dataset.repeat(100000).shuffle(10000).batch(256).prefetch(20)
    train_iter = dataset.make_one_shot_iterator()
    next_val = train_iter.get_next()
    return next_val


def dfm(next_val):
    px_logit = tf.get_variable(name="px_logit", shape=[1], dtype=tf.float32)
    px = tf.sigmoid(px_logit)

    lam_logit = tf.get_variable(name="lam_logit", shape=[1], dtype=tf.float32)
    lam = tf.exp(lam_logit)

    label = next_val[:, 0:1]
    delayed = next_val[:, 1:2]

    loss = tf.reduce_mean(-(label * (tf.log(px) + lam_logit - lam * delayed)
                            + (1 - label) * tf.log(1 - px + px * tf.exp(-lam * delayed))))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    return train_op, px, lam


def dfm_v2(next_val):
    px_logit = tf.get_variable(name="px_logit", shape=[1], dtype=tf.float32,
                               initializer=tf.constant_initializer(value=-2))
    px = tf.sigmoid(px_logit)

    lam_logit = tf.get_variable(name="lam_logit", shape=[1], dtype=tf.float32,
                                initializer=tf.constant_initializer(value=2))
    lam = tf.exp(lam_logit)

    label = next_val[:, 0:1]
    delayed = next_val[:, 1:2]

    w_0 = tf.stop_gradient(tf.exp(-lam * delayed) * px / (tf.exp(-lam * delayed) * px + 1 - px))
    loss = tf.reduce_mean(-(label * (tf.log(px) + lam_logit - lam * delayed)
                            + (1 - label) * (
                                    (1 - w_0) * tf.log(1 - px) + w_0 * (tf.log(px) - lam * delayed)
                            )))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    return train_op, px, lam


def dfm_v3(next_val):
    def model(is_var_trainable=True):
        px_logit = tf.get_variable(name="px_logit", shape=[1], dtype=tf.float32,
                                   initializer=tf.constant_initializer(value=-2))
        px = tf.sigmoid(px_logit)

        lam_logit = tf.get_variable(name="lam_logit", shape=[1], dtype=tf.float32,
                                    initializer=tf.constant_initializer(value=2))
        lam = tf.exp(lam_logit)
        if is_var_trainable:
            return px, lam, lam_logit
        else:
            return px, lam, lam_logit

    def get_coordinate_op(target_vars, action_vars):
        ops = []
        for target, action in zip(target_vars, action_vars):
            ops.append(tf.assign(action, target))
        return tf.group(ops, name="action_target_coordinate")

    with tf.variable_scope("target_net"):
        target_px, target_lam, target_lam_logit = model()

    with tf.variable_scope("action_net"):
        action_px, action_lam, action_lam_logit = model()

    # 同步 op
    target_net_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope="target_net")
    action_net_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope="action_net")
    coordinate_op = get_coordinate_op(target_net_vars, action_net_vars)
    global_step = tf.train.get_or_create_global_step()

    coordinate_op = tf.cond(tf.equal(global_step % 4, 0), lambda: coordinate_op, lambda: tf.no_op())

    label = next_val[:, 0:1]
    delayed = next_val[:, 1:2]

    w_0 = tf.stop_gradient(tf.exp(-target_lam * delayed) * target_px / (tf.exp(-target_lam * delayed) * target_px + 1 - target_px))
    loss = tf.reduce_mean(-(label * (tf.log(action_px) + action_lam_logit - action_lam * delayed)
                            + (1 - label) * (
                                    (1 - w_0) * tf.log(1 - action_px) + w_0 * (tf.log(action_px) - action_lam * delayed)
                            )))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss, global_step=global_step)
    train_op = tf.group(coordinate_op, train_op)
    return train_op, action_px, action_lam


def test_dfm_v1():
    per_day_records = 10
    begin_date = "20211001"
    end_date = "20211011"
    sample = data_generate(begin_date, end_date, per_day_records=per_day_records)
    train_data = train_data_generate(sample, end_date)

    next_val = get_dataset(train_data)
    results = dfm(next_val)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            print(sess.run(results))


def test_dfm_v2():
    per_day_records = 10
    begin_date = "20211001"
    end_date = "20211011"
    sample = data_generate(begin_date, end_date, per_day_records=per_day_records)
    train_data = train_data_generate(sample, end_date)

    next_val = get_dataset(train_data)
    results = dfm_v2(next_val)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            print(sess.run(results))


def test_dfm_v3():
    per_day_records = 10
    begin_date = "20211001"
    end_date = "20211011"
    sample = data_generate(begin_date, end_date, per_day_records=per_day_records)
    train_data = train_data_generate(sample, end_date)

    next_val = get_dataset(train_data)
    results = dfm_v3(next_val)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            print(sess.run(results))


if __name__ == '__main__':
    test_dfm_v1()
