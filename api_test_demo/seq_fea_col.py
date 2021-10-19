from __future__ import print_function
import tensorflow as tf
from tensorflow import feature_column

tokens = feature_column.sequence_categorical_column_with_hash_bucket("tokens", hash_bucket_size=1000,
                                                                     dtype=tf.int64)

token_embedding = feature_column.embedding_column(tokens, dimension=10)
columns = [token_embedding]

features = {
    "tokens": tf.constant([[1, 2, 3], [2, 3, 0]], dtype=tf.int64),
    "heights": tf.constant([[0.1, 0.2, 0.3], [0.1, 1.2, 2.]], dtype=tf.float32)
}

input_layer, sequence_length = tf.contrib.feature_column.sequence_input_layer(features, columns)

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        inp, seq_len = sess.run([input_layer, sequence_length])
        print(inp.shape, seq_len.shape)
