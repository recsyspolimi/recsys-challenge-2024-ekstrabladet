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
        embeddings_regularizer: tfk.regularizers = None,
        **kwargs
    ):
        super(SequenceMultiHotEmbeddingLayer, self).__init__(**kwargs)
        if pool_mode not in ['sum', 'max']:
            raise ValueError('Pooling mode not supported')
        
        self.cardinality = cardinality
        self.embedding_dim = embedding_dim
        self.pool_mode = pool_mode
        self.embeddings_regularizer = embeddings_regularizer
        
    def build(self, input_shape):
        self.embedding_matrix = self.add_weight(shape=(self.cardinality, self.embedding_dim), # [N, M]
                                                initializer='he_normal',
                                                trainable=True,
                                                name='embedding_matrix',
                                                regularizer=self.embeddings_regularizer)
        
    def call(self, inputs):
        # inputs_shape = [Bs, T, N]
        x = tf.einsum('nm,btn->btnm', self.embedding_matrix, inputs) # [Bs, T, N, M]
        if self.pool_mode == 'max':
            x = tfk.ops.max(x, axis=-2) # [Bs, T, M]
        elif self.pool_mode == 'sum':
            x = tfk.ops.sum(x, axis=-2)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'cardinality': self.cardinality,
            'embedding_dim': self.embedding_dim,
            'pool_mode': self.pool_mode,
            'embeddings_regularizer': tf.keras.regularizers.serialize(self.embeddings_regularizer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['embeddings_regularizer'] = tf.keras.regularizers.deserialize(config['embeddings_regularizer'])
        return cls(**config)
    
    