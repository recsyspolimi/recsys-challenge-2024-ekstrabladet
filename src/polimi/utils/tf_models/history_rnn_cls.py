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


class TemporalHistoryClassificationModel(TabularNNModel):
    
    def __init__(
        self,
        categorical_features: list[str] = None,
        numerical_features: list[str] = None,
        categorical_transform: str = 'embeddings',
        numerical_transform: str = 'quantile-normal',
        use_gaussian_noise: bool = False,
        gaussian_noise_std: float = 0.01,
        max_categorical_embedding_dim: int = 50,
        verbose: bool = False, 
        model_name: str = 'model',
        random_seed: int = 42,
        seq_embedding_dims: Dict[str, Tuple[int, int, bool]] = None, # feature: (cardinality, embedding_dim, is_multi_hot_vector)
        seq_numerical_features: List[str] = None,
        n_recurrent_layers: int = 1,
        recurrent_embedding_dim: int = 64,
        l1_lambda: float = 1e-4,
        l2_lambda: float = 1e-4,
        dense_n_layers: int = 1,
        dense_start_units: int = 128,
        dense_units_decay: int = 2,
        dense_dropout_rate: float = 0.1,
        dense_activation: str = 'relu',
        **kwargs
    ):
        
        super(TemporalHistoryModel, self).__init__(categorical_features=categorical_features, numerical_features=numerical_features,
                                                   categorical_transform=categorical_transform, numerical_transform=numerical_transform,
                                                   use_gaussian_noise=use_gaussian_noise, gaussian_noise_std=gaussian_noise_std, 
                                                   max_categorical_embedding_dim=max_categorical_embedding_dim,
                                                   verbose=verbose, model_name=model_name, random_seed=random_seed, **kwargs)
        
        self.seq_embedding_dims = seq_embedding_dims
        self.seq_numerical_features = seq_numerical_features
        self.n_recurrent_layers = n_recurrent_layers
        self.recurrent_embedding_dim = recurrent_embedding_dim
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.dense_n_layers = dense_n_layers
        self.dense_start_units = dense_start_units
        self.dense_units_decay = dense_units_decay
        self.dense_dropout_rate = dense_dropout_rate
        self.dense_activation = dense_activation
        
    def _build(self):
        inputs_dense, x0 = self._build_input_layers()
        
        rnn_inputs = []
        rnn_embeddings = []
        self.recurrent_features = []
        for feature_name, (cardinality, embedding_dim, is_multi_hot_vector) in self.seq_embedding_dims.items():
            self.recurrent_features.append(feature_name)
            if is_multi_hot_vector:
                input_layer = tfkl.Input(shape=(None, cardinality), name=f'{feature_name}_Input')
                embedding_layer = SequenceMultiHotEmbeddingLayer(
                    cardinality, 
                    embedding_dim,
                    embedding_regularizer=tfk.regularizers.L1L2(l1=self.l1_lambda, l2=self.l2_lambda)
                )(input_layer)
            else:
                input_layer = tfkl.Input(shape=(None,), name=f'{feature_name}_Input')
                embedding_layer = tfkl.Embedding(
                    cardinality, 
                    embedding_dim,
                    embedding_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda)
                )(input_layer)
            rnn_inputs.append(input_layer)
            rnn_embeddings.append(embedding_layer)
            
        numerical_sequence_input = tfkl.Input(shape=(None, len(self.seq_numerical_features)), name=f'{feature_name}_Input')
        rnn_inputs.append(numerical_sequence_input)
        rnn_embeddings.append(numerical_sequence_input)
        embeddings_rnn = tfkl.Concatenate(axis=-1)(rnn_embeddings)
        
        x_recurrent = embeddings_rnn
        for _ in range(self.n_recurrent_layers - 1):
            x_recurrent = tfkl.GRU(self.recurrent_embedding_dim, return_sequences=True)(x_recurrent)
            
        x_recurrent = tfkl.GRU(self.recurrent_embedding_dim, return_sequences=False)(x_recurrent)
        x = tfkl.Concatenate(axis=-1)([x0, x_recurrent])
        
        units = self.start_units
        for i in range(self.n_layers):
            x = tfkl.Dense(
                units=units,
                kernel_initializer=tfk.initializers.HeNormal(seed=self.random_seed),
                kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
                name=f'Dense{i}'
            )(x)
            x = tfkl.BatchNormalization(name=f'BatchNormalization{i}')(x)
            x = tfkl.Activation(self.activation, name=f'Activation{i}')(x)
            x = tfkl.Dropout(self.dropout_rate, name=f'Dropout{i}')(x)
            units = int(units / self.units_decay)

        outputs = tfkl.Dense(
            units=1,
            kernel_initializer=tfk.initializers.GlorotUniform(seed=self.random_seed),
            kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
            name='OutputDense',
            activation='sigmoid'
        )(x)
        self.model = tfk.Model(inputs=rnn_inputs + inputs_dense, outputs=outputs)
        
    def fit(
        self, 
        X: Tuple[pd.DataFrame, Dict[str, np.array]], 
        y: Union[np.array, pd.Series],
        validation_data: Tuple = None,
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
        checkpoint_dir: str = None,
        free_raw_data: bool = True
    ):       
        inputs = []
        if len(self.numerical_features) > 0:
            self._build_numerical_transformer()
            X[0][self.numerical_features] = X[0][self.numerical_features].replace([-np.inf, np.inf], np.nan).fillna(0)
            if self.xformer is not None:
                X_train_numerical = self.xformer.fit_transform(
                    X[0][self.numerical_features],
                    y
                ).astype(np.float32)
            else:
                X_train_numerical = X[self.numerical_features].values.astype(np.float32)
            inputs.append(X_train_numerical)
            
        if len(self.categorical_features) > 0:
            self.vocabulary_sizes = {}
            self.categories = []
            for f in self.categorical_features:
                X[0][f] = X[0][f].astype(str).fillna('NA')
                categories_train = list(X[f].unique())
                if 'NA' not in categories_train:
                    categories_train.append('NA')
                self.categories.append(categories_train)
                self.vocabulary_sizes[f] = len(categories_train)
            self._build_categorical_encoder()
            X_train_categorical = self.encoder.fit_transform(X[self.categorical_features], y).astype(np.int16)
            if self.categorical_transform == 'embeddings':
                inputs += [X_train_categorical[:, i].reshape(-1, 1) for i in range(len(self.categorical_features))]
            else:
                inputs.append(X_train_categorical)
                
        inputs.append([X[1][feature] for feature in self.recurrent_features])
                
        logging.info('Fitted data preprocessors')
        if free_raw_data:
            del X
            gc.collect()
        
        if len(inputs) == 1:
            inputs = inputs[0]
            
        if validation_data is not None:
            logging.info('Transforming validation data')
            X_val, y_val = validation_data
            validation_data = (self._transform_test_data(X_val), y_val)
            if free_raw_data:
                del X_val
                gc.collect()
                
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
                
        if self.model is None:
            self._build()
                
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
            inputs,
            y,
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
        inputs = self._transform_test_data(X[0])
        inputs.append([X[1][feature] for feature in self.recurrent_features])
        return self.model.predict(inputs, batch_size=batch_size, **kwargs)
    
    def save(self, directory, with_model=True):
        '''Pass with model=False only if saving before fit'''
        if with_model:
            self.model.save(os.path.join(directory, f'{self.model_name}.keras'))
        if self.encoder is not None:
            joblib.dump(self.encoder, os.path.join(directory, 'categorical_encoder.joblib'))
        if self.xformer is not None:
            joblib.dump(self.xformer, os.path.join(directory, 'numerical_transformer.joblib'))
        features_info = {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'recurrent_features': self.recurrent_features,
            'with_xformer': self.xformer is not None,
            'with_encoder': self.encoder is not None,
        }
        with open(os.path.join(directory, 'features_info.json'), 'w') as features_file:
            json.dump(features_info, features_file)
            
    def load(self, directory):
        self.model = tfk.models.load_model(os.path.join(directory, f'{self.model_name}.keras'))
        with open(os.path.join(directory, 'features_info.json'), 'r') as features_file:
            features_info = json.load(features_file)
        self.numerical_features = features_info['numerical_features']
        self.categorical_features = features_info['categorical_features']
        self.recurrent_features = features_info['recurrent_features']
        if len(self.numerical_features) > 0 and features_info['with_xformer']:
            try:
                self.xformer = joblib.load(os.path.join(directory, 'numerical_transformer.joblib'))
            except Exception:
                raise ValueError(f'Numerical transformer not found in {directory}')
        if len(self.categorical_features) > 0 and features_info['with_encoder']:
            try:
                self.encoder = joblib.load(os.path.join(directory, 'categorical_encoder.joblib'))
            except Exception:
                raise ValueError(f'Categorical encoder not found in {directory}')
            self.categories = self.encoder.categories_
        
        