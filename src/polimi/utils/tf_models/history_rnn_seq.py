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


class TemporalHistorySequenceModel(TabularNNModel):
    
    def __init__(
        self,
        seq_embedding_dims: Dict[str, Tuple[int, int, bool]] = None, # feature: (cardinality, embedding_dim, is_multi_hot_vector)
        seq_numerical_features: List[str] = None,
        n_recurrent_layers: int = 1,
        recurrent_embedding_dim: int = 64,
        l1_lambda: float = 1e-4,
        l2_lambda: float = 1e-4,
        random_seed: int = 42
    ):
        
        self.seq_embedding_dims = seq_embedding_dims
        self.seq_numerical_features = seq_numerical_features
        self.n_recurrent_layers = n_recurrent_layers
        self.recurrent_embedding_dim = recurrent_embedding_dim
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.random_seed = random_seed
        self.model = None
        self.model_name = 'TemporalHistorySequenceModel'
        
    def _build(self):

        inputs = {}
        rnn_embeddings = []
        self.recurrent_features = []
        for feature_name, (cardinality, embedding_dim, is_multi_hot_vector) in self.seq_embedding_dims.items():
            self.recurrent_features.append(feature_name)
            if is_multi_hot_vector:
                input_layer = tfkl.Input(shape=(None, cardinality), name=feature_name)
                embedding_layer = SequenceMultiHotEmbeddingLayer(
                    cardinality, 
                    embedding_dim,
                    embeddings_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda)
                )(input_layer)
            else:
                input_layer = tfkl.Input(shape=(None,), name=feature_name)
                embedding_layer = tfkl.Embedding(
                    cardinality, 
                    embedding_dim,
                    embeddings_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda)
                )(input_layer)
            inputs[feature_name] = input_layer
            rnn_embeddings.append(embedding_layer)
            
        numerical_sequence_input = tfkl.Input(shape=(None, len(self.seq_numerical_features)), name='input_numerical')
        inputs['input_numerical'] = numerical_sequence_input
        rnn_embeddings.append(numerical_sequence_input)
        embeddings_rnn = tfkl.Concatenate(axis=-1)(rnn_embeddings)
        
        x_recurrent = embeddings_rnn
        for _ in range(self.n_recurrent_layers - 1):
            x_recurrent = tfkl.GRU(self.recurrent_embedding_dim, return_sequences=True, use_bias=False)(x_recurrent)
            
        x_recurrent = tfkl.GRU(self.recurrent_embedding_dim, return_sequences=False, use_bias=False)(x_recurrent)
        
        outputs = []
        self.outputs_feature_names = [feature_name for feature_name in ['topics', 'subcategory', 'category', 'sentiment_label'] if f'input_{feature_name}' in self.recurrent_features]
        for feature_name in self.outputs_feature_names:
            features_output = tfkl.Dense(
                units=self.seq_embedding_dims[f'input_{feature_name}'][0],
                kernel_initializer=tfk.initializers.HeNormal(seed=self.random_seed),
                kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
                name=f'output_{feature_name}',
                use_bias=False,
                activation='sigmoid' if self.seq_embedding_dims[f'input_{feature_name}'][2] else 'softmax'
            )(x_recurrent)
            outputs.append(features_output)
            
        self.model = tfk.Model(inputs=inputs, outputs=outputs, name='TemporalHistorySequenceModel')     
        
    def fit(
        self, 
        train_dataset: tf.data.Dataset,
        validation_data: tf.data.Dataset = None,
        early_stopping_rounds: int = 1,
        batch_size: int = 256,
        epochs: int = 5,
        optimizer: tfk.optimizers.Optimizer = None, 
        loss: Union[tfk.losses.Loss, List[tfk.losses.Loss]] = None, 
        loss_weights: List[float] = None,
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
            loss_weights=loss_weights,
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
        
    def predict(self, X, batch_size=256, **kwargs):
        inputs = [X[feature] for feature in self.recurrent_features]
        return self.model.predict(inputs, batch_size=batch_size, **kwargs)
    
    def save(self, directory, with_model=True):
        '''Pass with model=False only if saving before fit'''
        features_info = {
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
        self.recurrent_features = features_info['recurrent_features']
        
        