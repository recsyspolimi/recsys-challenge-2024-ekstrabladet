import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from .activations import t_softmax
from tensorflow.keras import saving


# @saving.register_keras_serializable(package="MyLayers", name="GatedFeatureLearningUnit")
class GatedFeatureLearningUnit(tfkl.Layer):
    def __init__(self, n_stages=4, dropout_rate=0.1, l1=1e-4, l2=1e-4, init_t=0.5, **kwargs):
        super(GatedFeatureLearningUnit, self).__init__(**kwargs)
        self.n_stages = n_stages
        self.dropout_rate = dropout_rate
        self.l1 = l1
        self.l2 = l2
        self.init_t = init_t

    def build(self, input_shape):
        self.W_i = [
            tfkl.Dense(2 * input_shape[-1],
                       kernel_regularizer=tfk.regularizers.L1L2(l1=self.l1, l2=self.l2),
                       name=f'i_dense_{i}')
            for i in range(self.n_stages)
        ]
        self.W_o = [
            tfkl.Dense(input_shape[-1],
                       kernel_regularizer=tfk.regularizers.L1L2(l1=self.l1, l2=self.l2),
                       name=f'o_dense_{i}')
            for i in range(self.n_stages)
        ]
        for i in range(self.n_stages):
            self.W_i[i].build((2 * input_shape[-1],))
            self.W_o[i].build((2 * input_shape[-1],))
        self.mask_weights = self.add_weight(shape=(self.n_stages, input_shape[-1]),
                                            initializer='he_normal',
                                            trainable=True,
                                            name='feature_weights',
                                            regularizer=tfk.regularizers.L1L2(l1=self.l1, l2=self.l2))
        self.t = self.add_weight(shape=(),
                                 initializer=tfk.initializers.Constant(self.init_t),
                                 trainable=True,
                                 name='t')

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        feature_mask = t_softmax(self.mask_weights, self.t)
        H_n = inputs
        for n in range(self.n_stages):
            X_n = feature_mask[n, :] * inputs
            H_in_n = self.W_i[n](tf.concat([H_n, X_n], axis=-1))
            z_n = tf.nn.sigmoid(H_in_n[:, input_shape[-1]:])
            r_n = tf.nn.sigmoid(H_in_n[:, :input_shape[-1]])
            H_out_n = tf.nn.tanh(self.W_o[n](tf.concat([tfkl.multiply([r_n, H_n]), X_n], axis=-1)))
            H_n = tfkl.add([tfkl.multiply([(1 - z_n), H_n]), tfkl.multiply([z_n, H_out_n])])

        return H_n

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(GatedFeatureLearningUnit, self).get_config()
        config.update({
            'n_stages': self.n_stages,
            'dropout_rate': self.dropout_rate,
            'l1': self.l1,
            'l2': self.l2,
            'init_t': self.init_t
        })
        return config