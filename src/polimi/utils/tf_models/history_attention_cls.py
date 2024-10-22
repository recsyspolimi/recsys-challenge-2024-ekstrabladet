import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from typing_extensions import Dict, List, Tuple, Union
import logging
import pandas as pd
import numpy as np
import gc
import os
import joblib
import json

from .base_model import TabularNNModel
from .layers import SequenceMultiHotEmbeddingLayer


class AttentionHistoryClassificationModel(TabularNNModel):
    
    def __init__(
        self,
        categorical_features: list[str] = None,
        numerical_features: list[str] = None,
        use_gaussian_noise: bool = False,
        gaussian_noise_std: float = 0.01,
        max_categorical_embedding_dim: int = 50,
        vocabulary_sizes: Dict[str, int] = None,
        verbose: bool = False, 
        model_name: str = 'AttentionHistoryClassificationModel',
        random_seed: int = 42,
        seq_embedding_dims: Dict[str, Tuple[int, int, bool]] = None, # feature: (cardinality, embedding_dim, is_multi_hot_vector)
        seq_numerical_features: List[str] = None,
        window_size: int = 20,
        embedding_dim: int = 128,
        l1_lambda: float = 1e-4,
        l2_lambda: float = 1e-4,
        dropout_rate: float = 0.1,
        n_encoder_layers: int = 0,
        query_n_layers: int = 1,
        query_start_units: int = 128,
        query_units_decay: int = 2,
        query_activation: str = 'relu',
        dense_n_layers: int = 1,
        dense_start_units: int = 128,
        dense_units_decay: int = 2,
        dense_activation: str = 'relu',
    ):
        
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.use_gaussian_noise = use_gaussian_noise
        self.gaussian_noise_std = gaussian_noise_std
        self.max_categorical_embedding_dim = max_categorical_embedding_dim
        self.vocabulary_sizes = vocabulary_sizes
        self.verbose = verbose
        self.model_name = model_name
        self.random_seed = random_seed
        self.seq_embedding_dims = seq_embedding_dims
        self.seq_numerical_features = seq_numerical_features
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.n_encoder_layers = n_encoder_layers
        self.dropout_rate = dropout_rate
        self.query_n_layers = query_n_layers
        self.query_start_units = query_start_units
        self.query_units_decay = query_units_decay
        self.query_activation = query_activation
        self.dense_n_layers = dense_n_layers
        self.dense_start_units = dense_start_units
        self.dense_units_decay = dense_units_decay
        self.dense_activation = dense_activation
        self.model = None
        
    def _build_dense_input_layers(self):
        inputs = {}
        concat_inputs = []
        if len(self.numerical_features) > 0:
            numeric_input_layer = tfkl.Input(shape=(len(self.numerical_features),), name='numerical_columns')
            inputs['numerical_columns'] = numeric_input_layer
            concat_inputs.append(numeric_input_layer)
        
        if len(self.categorical_features) > 0:
            
            embedding_layers = []
            for feature in self.categorical_features:
                vocabulary_size = self.vocabulary_sizes[feature]
                embedding_dim = min(self.max_categorical_embedding_dim, (vocabulary_size + 1) // 2)
                cat_input = tfkl.Input(shape=(1,), name=feature)
                inputs[feature] = cat_input
                embedding_layer = tfkl.Embedding(input_dim=vocabulary_size + 1, output_dim=embedding_dim)(cat_input)
                flatten_layer = tfkl.Flatten()(embedding_layer)
                embedding_layers.append(flatten_layer)
            concat_inputs += embedding_layers
                
        if len(concat_inputs) > 1:
            encoded_inputs = tfkl.Concatenate()(concat_inputs)
        else:
            encoded_inputs = concat_inputs[0]
            
        if self.use_gaussian_noise:
            encoded_inputs = tfkl.GaussianNoise(self.gaussian_noise_std, name='GaussianNoise')(encoded_inputs)
            
        return inputs, encoded_inputs
        
    def _build(self):
        inputs, x0 = self._build_dense_input_layers()

        rnn_embeddings = []
        self.recurrent_features = []
        for feature_name, (cardinality, embedding_dim, is_multi_hot_vector) in self.seq_embedding_dims.items():
            self.recurrent_features.append(feature_name)
            if is_multi_hot_vector:
                input_layer = tfkl.Input(shape=(self.window_size, cardinality), name=feature_name)
                embedding_layer = SequenceMultiHotEmbeddingLayer(
                    cardinality, 
                    embedding_dim,
                    embeddings_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
                    pool_mode='mean',
                )(input_layer)
            else:
                input_layer = tfkl.Input(shape=(self.window_size,), name=feature_name)
                embedding_layer = tfkl.Embedding(
                    cardinality, 
                    embedding_dim,
                    embeddings_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda)
                )(input_layer)
            inputs[feature_name] = input_layer
            rnn_embeddings.append(embedding_layer)
            
        numerical_sequence_input = tfkl.Input(shape=(self.window_size, len(self.seq_numerical_features)), name='input_numerical')
        inputs['input_numerical'] = numerical_sequence_input
        rnn_embeddings.append(numerical_sequence_input)
        embeddings = tfkl.Concatenate(axis=-1)(rnn_embeddings)        
        
        values = tfkl.Conv1D(self.embedding_dim, 
                             kernel_size=1,
                             kernel_initializer=tfk.initializers.HeNormal(seed=self.random_seed),
                             kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
                             bias_regularizer=tfk.regularizers.l2(l2=self.l2_lambda),
                             name='EmbeddingsLinear')(embeddings)
        values = tfkl.BatchNormalization(name='EmbeddingsBatchNormalization')(values)
        
        for i in range(self.n_encoder_layers):
            values = tfkl.MultiHeadAttention(
                num_heads=4,
                key_dim=self.embedding_dim,
                dropout=self.dropout_rate,
                kernel_initializer=tfk.initializers.HeNormal(seed=self.random_seed),
                kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
                bias_regularizer=tfk.regularizers.l2(l2=self.l2_lambda),
                name=f'AttentionEncoder{i}'
            )(query=values, value=values)
            values_ff = tfkl.Conv1D(self.embedding_dim, 
                                    kernel_size=1,
                                    kernel_initializer=tfk.initializers.HeNormal(seed=self.random_seed),
                                    kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
                                    bias_regularizer=tfk.regularizers.l2(l2=self.l2_lambda),
                                    activation=self.dense_activation,
                                    name=f'FeedForward0Encoder{i}')(values)
            values_ff = tfkl.Conv1D(self.embedding_dim, 
                                    kernel_size=1,
                                    kernel_initializer=tfk.initializers.HeNormal(seed=self.random_seed),
                                    kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
                                    bias_regularizer=tfk.regularizers.l2(l2=self.l2_lambda),
                                    name=f'FeedForward1Encoder{i}')(values_ff)
            values_ff = tfkl.Dropout(self.dropout_rate, name=f'DropoutEncoder{i}')(values_ff)
            values = tfkl.Add(name=f'AddEncoder{i}')([values, values_ff])
            values = tfkl.LayerNormalization(name=f'LayerNormalizationEncoder{i}')(values)

        units = self.query_start_units
        query = x0
        for i in range(self.query_n_layers):
            query = tfkl.Dense(
                units=units,
                kernel_initializer=tfk.initializers.HeNormal(seed=self.random_seed),
                kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
                bias_regularizer=tfk.regularizers.l2(l2=self.l2_lambda),
                name=f'QueryDense{i}'
            )(query)
            query = tfkl.BatchNormalization(name=f'QueryBatchNormalization{i}')(query)
            query = tfkl.Activation(self.query_activation, name=f'QueryActivation{i}')(query)
            query = tfkl.Dropout(self.dropout_rate, name=f'QueryDropout{i}')(query)
            units = int(units / self.query_units_decay)
            
        query = tfkl.Dense(
            units=self.embedding_dim,
            kernel_initializer=tfk.initializers.HeNormal(seed=self.random_seed),
            kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
            bias_regularizer=tfk.regularizers.l2(l2=self.l2_lambda),
            name='FinalQueryDense'
        )(query)
        query = tfkl.Lambda(lambda x: tf.expand_dims(x, axis=1))(query)
        
        history_embedding = tfkl.MultiHeadAttention(
            num_heads=4,
            key_dim=self.embedding_dim,
            dropout=self.dropout_rate,
            kernel_initializer=tfk.initializers.HeNormal(seed=self.random_seed),
            kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
            bias_regularizer=tfk.regularizers.l2(l2=self.l2_lambda),
        )(query=query, value=values)
            
        history_embedding = tfkl.Flatten()(history_embedding)
        x = tfkl.Concatenate(axis=-1)([x0, history_embedding])
        
        units = self.dense_start_units
        for i in range(self.dense_n_layers):
            x = tfkl.Dense(
                units=units,
                kernel_initializer=tfk.initializers.HeNormal(seed=self.random_seed),
                kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
                bias_regularizer=tfk.regularizers.l2(l2=self.l2_lambda),
                name=f'Dense{i}'
            )(x)
            x = tfkl.BatchNormalization(name=f'BatchNormalization{i}')(x)
            x = tfkl.Activation(self.dense_activation, name=f'Activation{i}')(x)
            x = tfkl.Dropout(self.dropout_rate, name=f'Dropout{i}')(x)
            units = int(units / self.dense_units_decay)

        outputs = tfkl.Dense(
            units=1,
            kernel_initializer=tfk.initializers.GlorotUniform(seed=self.random_seed),
            kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
            bias_regularizer=tfk.regularizers.l2(l2=self.l2_lambda),
            name='OutputDense',
            activation='sigmoid'
        )(x)
        self.model = tfk.Model(inputs=inputs, outputs=outputs)
        
    def fit(
        self, 
        train_dataset: tf.data.Dataset,
        validation_data: tf.data.Dataset = None,
        early_stopping_rounds: int = 1,
        batch_size: int = 256,
        epochs: int = 1,
        optimizer: tfk.optimizers.Optimizer = None, 
        loss: tfk.losses.Loss = None, 
        metrics: List[Union[str, tfk.metrics.Metric]] = None,
        early_stopping_monitor: str = 'val_loss',
        early_stopping_mode: str = 'auto',
        lr_scheduler: Union[callable, tfk.callbacks.Callback] = None,
        save_checkpoints: bool = True,
        checkpoint_dir: str = None
    ):      
        if self.model is None:
            self._build() 
            
        if validation_data is not None:
                
            logging.info(f'Training with early stopping patience {early_stopping_rounds}')
            early_stopping = tfk.callbacks.EarlyStopping(
                monitor=early_stopping_monitor, 
                patience=early_stopping_rounds, 
                mode=early_stopping_mode, 
                restore_best_weights=True
            )
            callbacks = [early_stopping]
        else:
            callbacks = []
        
        if lr_scheduler is not None:
            if type(lr_scheduler) == callable:
                scheduler = tfk.callbacks.LearningRateScheduler(lr_scheduler)
                callbacks.append(scheduler)
            else:
                callbacks.append(lr_scheduler)
                
        callbacks.append(tfk.callbacks.TerminateOnNaN())
                
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
        )
        self.model.summary(print_fn=logging.info)
        
        if save_checkpoints and checkpoint_dir is not None:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            self.save(checkpoint_dir, with_model=False)
            logging.info(f'Checkpoints will be saved at {checkpoint_dir}')
            callbacks.append(tfk.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'checkpoint.weights.h5'),
                save_weights_only=True,
                monitor=early_stopping_monitor if validation_data else 'loss',
                mode=early_stopping_mode if validation_data else 'auto',
                save_best_only=True))
            
        fit_history = self.model.fit(
            train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            validation_batch_size=batch_size if validation_data is not None else None,
            callbacks=callbacks
        ).history
        
        logging.info(f'Fit complete after {len(fit_history["loss"])}')
        
        if save_checkpoints and checkpoint_dir:
            self.model.load_weights(os.path.join(checkpoint_dir, 'checkpoint.weights.h5'))
            pd.DataFrame.from_dict(fit_history).to_csv(os.path.join(checkpoint_dir, 'history.csv'), index=False)
        
    def predict(self, test_dataset, batch_size=256, **kwargs):
        return self.model.predict(test_dataset, batch_size=batch_size, **kwargs)
    
    def save(self, directory, with_model=True):
        '''Pass with model=False only if saving before fit'''
        features_info = {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'recurrent_features': self.recurrent_features,
        }
        with open(os.path.join(directory, 'features_info.json'), 'w') as features_file:
            json.dump(features_info, features_file)
        if with_model:
            self.model.save(os.path.join(directory, f'{self.model_name}.keras'))
            
    def load(self, directory):
        self.model = tfk.models.load_model(os.path.join(directory, f'{self.model_name}.keras'))
        with open(os.path.join(directory, 'features_info.json'), 'r') as features_file:
            features_info = json.load(features_file)
        self.numerical_features = features_info['numerical_features']
        self.categorical_features = features_info['categorical_features']
        self.recurrent_features = features_info['recurrent_features']
        
        