import tensorflow as tf


def t_softmax(logits, t, axis=-1, name=None):
    weights = tf.nn.relu(logits + t - tf.reduce_max(logits, axis=axis, keepdims=True))
    return weights * tf.exp(logits) / tf.reduce_sum(weights * tf.exp(logits), axis, keepdims=True)