import tensorflow as tf
import numpy as np
from .hooks import LabelDistHook, GroupAucHook, RegressionHook


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    tf.add_to_collection("label", labels)
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    cls_summ = tf.summary.histogram("class_dist", tf.argmax(input=logits, axis=1))
    print(tf.argmax(input=logits, axis=1))
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            tf.estimator.export.PredictOutput(outputs=predictions)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        print("build Predict graph")
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                          export_outputs=export_outputs)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    global_step = tf.train.get_or_create_global_step()
    is_replace = tf.equal(global_step % 3, 0)

    if mode == tf.estimator.ModeKeys.TRAIN:
        print("build train graph")
        label_dist_hook = LabelDistHook(name="train_label", label_tensor=labels,
                                        mask_tensor=tf.ones_like(labels))
        group_auc_hook = GroupAucHook(name="train_classification", group_tensor=tf.ones_like(labels),
                                      label_tensor=tf.one_hot(labels, depth=10)[:, 0],
                                      pred_tensor=tf.sigmoid(logits[:, 0]))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op,
                                          training_hooks=[label_dist_hook, group_auc_hook])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    summary_hook = tf.train.SummarySaverHook(save_steps=1, summary_op=[cls_summ], output_dir="model_dir/eval")

    print("build eval graph")
    label_dist_hook = LabelDistHook(name="eval_label", label_tensor=labels,
                                    mask_tensor=tf.ones_like(labels), reset_step=50)
    group_auc_hook = GroupAucHook(name="eval_classification", group_tensor=tf.ones_like(labels),
                                  label_tensor=tf.one_hot(labels, depth=10)[:, 0],
                                  pred_tensor=tf.sigmoid(logits[:, 0]), reset_step=50)
    regression_info_hook = RegressionHook(name="wtf",
                                          group_tensor=tf.one_hot(labels, depth=10)[:, 0],
                                          label_tensor=tf.one_hot(labels, depth=10)[:, 0],
                                          pred_tensor=tf.sigmoid(logits[:, 0]),
                                          mask_tensor=tf.ones_like(logits[:, 0]), reset_step=50)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=tf.cond(tf.random_uniform(shape=[], maxval=1) > 0.7,
                     lambda: tf.constant(100, dtype=tf.float32),
                     lambda: tf.constant(200, dtype=tf.float32)),
        eval_metric_ops=eval_metric_ops, export_outputs=None,
        evaluation_hooks=[summary_hook, label_dist_hook, group_auc_hook, regression_info_hook])


def input_fn():
    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    train_data = train_data / np.float32(255)
    train_labels = train_labels.astype(np.int32)  # not required

    eval_data = eval_data / np.float32(255)
    eval_labels = eval_labels.astype(np.int32)  # not required

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=64,
        num_epochs=500,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    return train_input_fn, eval_input_fn


def serving_input_fn():
    example_proto = tf.placeholder(dtype=tf.string, shape=[None])
    receiver_tensor = {"data": example_proto}
    feature = tf.parse_example(example_proto, features={"x": tf.FixedLenFeature([], dtype=tf.string)})
    img = tf.io.decode_raw(feature['x'], out_type=tf.float32)
    feature['x'] = img
    return tf.estimator.export.ServingInputReceiver(features=feature, receiver_tensors=receiver_tensor)


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    estimator_config = tf.estimator.RunConfig(model_dir="model_dir",
                                              save_checkpoints_steps=100)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, config=estimator_config)

    train_input_fn, eval_input_fn = input_fn()

    exporter = tf.estimator.BestExporter(serving_input_receiver_fn=serving_input_fn)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None,
                                      start_delay_secs=20,
                                      exporters=exporter, throttle_secs=5)

    for i in range(2):
        # mnist_classifier.train(train_input_fn, steps=300)
        # logging_global_step = tf.train.LoggingTensorHook(
        #     tensors={"show_me_global_step": "global_step"}, every_n_iter=1)
        # print(mnist_classifier.evaluate(eval_input_fn))
        for i, v in enumerate(mnist_classifier.predict(eval_input_fn, yield_single_examples=True)):
            print(v)
            break
    #
    # mnist_classifier.export_saved_model()

    # tf.estimator.train_and_evaluate(estimator=mnist_classifier, train_spec=train_spec, eval_spec=eval_spec)
