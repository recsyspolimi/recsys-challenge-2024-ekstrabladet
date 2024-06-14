import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from tensorflow.keras import saving


@saving.register_keras_serializable(package="MyLayers", name="SequenceEmbeddingLayer")
class SequenceMultiHotEmbeddingLayer(tfkl.Layer):
    
    def __init__(
        self,
        cardinality: int,
        embedding_dim: int,
        pool_mode: str = 'max', # sum or max
        embedding_regularizer: tfk.regularizers = None
    ):
        if pool_mode not in ['sum', 'max']:
            raise ValueError('Pooling mode not supported')
        
        self.cardinality = cardinality
        self.embedding_dim = embedding_dim
        self.pool_mode = pool_mode
        self.embedding_regularizer = embedding_regularizer
        
    def build(self, input_shape):
        self.embedding_matrix = self.add_weight(shape=(self.cardinality, self.embedding_dim), # [N, M]
                                                initializer='he_normal',
                                                trainable=True,
                                                name='embedding_matrix',
                                                regularizer=self.embedding_regularizer)
        
    def call(self, inputs):
        # inputs_shape = [Bs, T, N]
        x = tf.einsum('nm,btn->btnm', self.embedding_matrix, inputs) # [Bs, T, N, M]
        if self.pool_mode == 'max':
            x = tfk.ops.max(x, axis=-2) # [Bs, T, M]
        elif self.pool_mode == 'sum':
            x = tfk.ops.sum(x, axis=-2)
        return x