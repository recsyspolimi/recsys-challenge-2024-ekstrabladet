import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import os
from typing_extensions import Union, Tuple, List
import pandas as pd
from sklearn.preprocessing import (
    QuantileTransformer,
    OneHotEncoder, 
    TargetEncoder,
    OrdinalEncoder, 
    MaxAbsScaler, 
    MinMaxScaler,
    StandardScaler,
    PowerTransformer
)
from abc import ABC, abstractmethod
import joblib
import optuna
import numpy as np
import logging
import json
import gc


class TabularNNModel(ABC):
    '''
    The base class for each tabular neural network model.
    
    This class handle the null values by just filling them with zero. To better handle them
    treat them separately outside the class.
    '''
    
    CATEGORICAL_TRANSFORMS = ['embeddings', 'one-hot-encoding', 'target-encoding']
    NUMERICAL_TRANSOFORMS = [None, 'min-max', 'standard', 'quantile-normal', 'quantile-uniform', 'max-abs', 'yeo-johnson']
    
    def __init__(
        self, 
        categorical_features: List[str] = None,
        numerical_features: List[str] = None,
        categorical_transform: str = 'embeddings',
        numerical_transform: str = None,
        use_gaussian_noise: bool = False,
        gaussian_noise_std: float = 0.01,
        max_categorical_embedding_dim: int = 50,
        verbose: bool = False, 
        model_name: str = 'model',
        random_seed: int = 42, 
        **kwargs
    ):
        '''
        Args:
            categorical_features (List[str]): the list of categorical features
            numerical_features (List[str]): the list of numerical features
            categorical_transform (str): the type of categorical encoder, can be one-hot-encoding, target-encoding
                or embeddings (in this case the categorical variables will go through a learnable embedding layer)
            numerical_transform (str): the type of numerical preprocessing. Can be any between: "yeo-johnson", "standard", 
                "quantile-normal", "quantile-normal", "max-abs". If None, no preprocessing is done
            use_gaussian_noise (bool): if True, applies a gaussian noise to the input
            gaussian_noise_std (float): the standard deviation of the gaussian noise
            max_categorical_embedding_dim (int): the maximum size of a categorical embedding. The actual size will be
                computed as min(max_categorical_embedding_dim, (vocabulary_size + 1) // 2) for each category separately
            verbose (bool)
            model_name (str)
            random_seed (int)
        '''
        super(TabularNNModel, self).__init__()
        
        if numerical_transform not in self.NUMERICAL_TRANSOFORMS:
            raise ValueError(f'Numerical transformation {numerical_transform} not available, choose one between {self.NUMERICAL_TRANSOFORMS}')
        if categorical_transform not in self.CATEGORICAL_TRANSFORMS:
            raise ValueError(f'Categorical encoding {categorical_transform} not available, choose one between {self.CATEGORICAL_TRANSFORMS}')
        if len(categorical_features) + len(numerical_features) == 0:
            raise ValueError('Provide at least one numerical feature or one categorical feature')
        
        self.verbose = verbose
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.numerical_features = numerical_features if numerical_features is not None else []
        self.categorical_transform = categorical_transform
        self.numerical_transform = numerical_transform
        self.max_categorical_embedding_dim = max_categorical_embedding_dim
        self.use_gaussian_noise = use_gaussian_noise
        self.gaussian_noise_std = gaussian_noise_std
        self.random_seed = random_seed
        self.model_name = model_name
        self.xformer = None
        self.encoder = None
        self.model: tfk.Model = None

    def __call__(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def fit(
        self, 
        X: pd.DataFrame, 
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
            X[self.numerical_features] = X[self.numerical_features].replace([-np.inf, np.inf], np.nan).fillna(0)
            if self.xformer is not None:
                X_train_numerical = self.xformer.fit_transform(
                    X[self.numerical_features],
                    y
                ).astype(np.float32)
            else:
                X_train_numerical = X[self.numerical_features].values.astype(np.float32)
            inputs.append(X_train_numerical)
            
        if len(self.categorical_features) > 0:
            self.vocabulary_sizes = {}
            self.categories = []
            for f in self.categorical_features:
                X[f] = X[f].astype(str).fillna('NA')
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
        return self.model.predict(self._transform_test_data(X), batch_size=batch_size, **kwargs)        
        
    def summary(self, expand_nested=True, **kwargs):
        self.model.summary(expand_nested=expand_nested, **kwargs)
    
    def plot(self, dpi:int=50, expand_nested:bool=True, show_shapes:bool=True):
        assert self.model is not None, 'Model not compiled yet'
        return tfk.utils.plot_model(self.model, expand_nested=expand_nested, show_shapes=show_shapes, dpi=dpi)

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
        
    @abstractmethod
    def _build(self):
        raise NotImplementedError('Method build not implemented')
    
    @classmethod
    def get_optuna_trial(cls, trial: optuna.Trial):
        params = {
            'use_gaussian_noise': trial.suggest_categorical('use_gaussian_noise', [True, False]),
            'numerical_transform': trial.suggest_categorical('numerical_transform', ['min-max', 'quantile-normal', 'yeo-johnson']),
        }
        if params['use_gaussian_noise']:
            params['gaussian_noise_std'] = trial.suggest_float('gaussian_noise_std', 1e-3, 1)
        return params
        
    def _transform_test_data(self, X: pd.DataFrame):
        inputs = []
        if len(self.numerical_features) > 0:
            X[self.numerical_features] = X[self.numerical_features].replace([-np.inf, np.inf], np.nan).fillna(0)
            if self.xformer is not None:
                X_numerical = self.xformer.transform(X[self.numerical_features]).astype(np.float32)
            else:
                X_numerical = X[self.numerical_features].values.astype(np.float32)
            inputs.append(X_numerical)    
        if len(self.categorical_features) > 0:
            for i, f in enumerate(self.categorical_features):
                X[f] = X[f].astype(str).fillna('NA')
                categories_val = list(X[f].unique())
                unknown_categories = [x for x in categories_val if x not in self.categories[i]]
                X[f] = X[f].replace(list(unknown_categories), 'NA')
            X_categorical = self.encoder.transform(X[self.categorical_features]).astype(np.int16)
            if self.categorical_transform == 'embeddings':
                X_categorical = [X_categorical[:, i].reshape(-1, 1) for i in range(len(self.categorical_features))]
                inputs += X_categorical
            else:
                inputs.append(X_categorical)
        if len(inputs) == 1:
            inputs = inputs[0]
        return inputs
                
    def _build_input_layers(self):
        inputs = []
        concat_inputs = []
        if len(self.numerical_features) > 0:
            numeric_input_layer = tfkl.Input(shape=(len(self.numerical_features),), name='NumericInput')
            inputs.append(numeric_input_layer)
            concat_inputs.append(numeric_input_layer)
        
        if self.categorical_features is not None:
            
            if self.categorical_transform == 'embeddings':
                embedding_layers = []
                categorical_inputs = []
                for feature in self.categorical_features:
                    vocabulary_size = self.vocabulary_sizes[feature]
                    embedding_dim = min(self.max_categorical_embedding_dim, (vocabulary_size + 1) // 2)
                    cat_input = tfkl.Input(shape=(1,), name=f'{feature}Input')
                    categorical_inputs.append(cat_input)
                    embedding_layer = tfkl.Embedding(input_dim=vocabulary_size + 1, output_dim=embedding_dim)(cat_input)
                    flatten_layer = tfkl.Flatten()(embedding_layer)
                    embedding_layers.append(flatten_layer)
                inputs += categorical_inputs
                concat_inputs += embedding_layers
                
            else:
                categorical_input = tfkl.Input(
                    shape=(self.categorical_encoding_shape,), 
                    name='CategoricalInput',
                )
                inputs.append(categorical_input)
                concat_inputs.append(categorical_input)
                
        if len(inputs) > 1:
            encoded_inputs = tfkl.Concatenate()(concat_inputs)
        else:
            encoded_inputs = concat_inputs[0]
            
        if self.use_gaussian_noise:
            encoded_inputs = tfkl.GaussianNoise(self.gaussian_noise_std, name='GaussianNoise')(encoded_inputs)
            
        return inputs, encoded_inputs
    
    def _build_numerical_transformer(self):
        self.xformer = None
        if self.numerical_transform == 'quantile-normal':
            self.xformer = QuantileTransformer(output_distribution='normal')
        elif self.numerical_transform == 'quantile-uniform':
            self.xformer = QuantileTransformer(output_distribution='uniform')
        elif self.numerical_transform == 'min-max':
            self.xformer = MinMaxScaler()
        elif self.numerical_transform == 'max-abs':
            self.xformer = MaxAbsScaler()
        elif self.numerical_transform == 'standard':
            self.xformer = StandardScaler()
        elif self.numerical_transform == 'yeo-johnson':
            self.xformer = PowerTransformer(method='yeo-johnson')
        
    def _build_categorical_encoder(self):
        if self.categorical_transform == 'embeddings':
            self.encoder = OrdinalEncoder(categories=self.categories)
        elif self.categorical_transform == 'target-encoding':
            self.encoder = TargetEncoder(target_type='binary', categories=self.categories)
            self.categorical_encoding_shape = len(self.categorical_features)
        elif self.categorical_transform == 'one-hot-encoding':
            self.encoder = OneHotEncoder(drop='first', sparse_output=False, categories=self.categories)
            self.categorical_encoding_shape = sum([len(cat) for cat in self.categories])