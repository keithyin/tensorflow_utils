from __future__ import print_function

import tensorflow as tf


def get_lr_with_cold_start_and_decay(cfg, global_step):
    cold_start_init_lr = cfg["cold_start_init_lr"]
    cold_start_end_lr = cfg["cold_start_end_lr"]
    cold_start_steps = cfg["cold_start_steps"]

    cur_lr = (
            (cold_start_end_lr - cold_start_init_lr) / cold_start_steps * tf.cast(global_step, dtype=tf.float32)
            + cold_start_init_lr)
    lr = tf.where(
        global_step <= cold_start_steps, cur_lr,
        tf.train.exponential_decay(
            learning_rate=cold_start_end_lr,
            global_step=global_step - cold_start_steps,
            decay_steps=cfg["decay_steps"],
            decay_rate=cfg["decay_rate"]
        ))
    return lr


if __name__ == '__main__':
    global_step = tf.train.get_or_create_global_step()
    update_global_step = tf.assign_add(global_step, 1)

    cfg = {
        "cold_start_init_lr": 0.01,
        "cold_start_end_lr": 0.1,
        "cold_start_steps": 10,
        "decay_steps": 10,
        "decay_rate": 0.1
    }

    lr = get_lr_with_cold_start_and_decay(cfg, global_step)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(100):
            print(sess.run([global_step, lr, update_global_step]))
