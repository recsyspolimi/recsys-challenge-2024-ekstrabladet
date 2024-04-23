import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from .entmax15 import entmax15


#@tf.keras.saving.register_keras_serializable(package="MyLayers", name="AbstractLayer")
class AbstractLayer(tfkl.Layer):
    def __init__(self, k, d, kernel_regularizer=None, use_bias=True, **kwargs):
        super(AbstractLayer, self).__init__(**kwargs)
        self.k = k
        self.d = d
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias

    def build(self, input_shape):
        self.mask_weights = self.add_weight(shape=(self.k, input_shape[-1]),
                                            initializer='he_normal',
                                            trainable=True,
                                            name='feature_weights',
                                            regularizer=self.kernel_regularizer)
        self.fc = tfkl.Conv1D(filters=2 * self.k * self.d,
                              kernel_size=1,
                              groups=self.k, # one group means one vector of shape input_shape[-1]
                              use_bias=self.use_bias)
        self.bn = tfkl.BatchNormalization()
        super(AbstractLayer, self).build(input_shape)

    def call(self, inputs):
        input_shape = tf.shape(inputs) # [Bs, N]

        # f' = M .* f
        mask = entmax15(self.mask_weights)
        x = tf.einsum('kn,bn->bkn', mask, inputs) # [Bs, k, N] - N = #input_features

        x = self.fc(tf.reshape(x, shape=[input_shape[0], -1, self.k * input_shape[1]]))
        x = self.bn(x)
        # output shape is [Bs, 1, k * d * 2]
        # each group of d * 2 elements is a different feature selection & abstraction group
        x = sum(
            # f* = ReLU(q .* BN(W2 f')), where q = sigmoid(BN(W1 f'))
            tf.nn.relu(tf.nn.sigmoid(chunk[:, :, :self.d]) * chunk[:, :, self.d:])
            for chunk in tf.split(x, num_or_size_splits=self.k, axis=-1)
        )
        x = tf.reshape(x, (input_shape[0], self.d))
        return x
        # return tf.squeeze(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.d)

    def get_config(self):
        config = super(AbstractLayer, self).get_config()
        config.update({
            'k': self.k,
            'd': self.d,
            'kernel_regularizer': tfk.saving.serialize_keras_object(self.kernel_regularizer),
            'use_bias': self.use_bias
        })
        return config

    @classmethod
    def from_config(cls, config):
        kernel_regularizer_config = config.pop("kernel_regularizer")
        kernel_regularizer = tfk.saving.deserialize_keras_object(kernel_regularizer_config)
        return cls(**config, kernel_regularizer=kernel_regularizer)