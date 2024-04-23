import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import os
from typing_extensions import Union, Tuple
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


class TabularNNModel(ABC):
    
    CATEGORICAL_TRANSFORMS = ['embeddings', 'one-hot-encoding', 'target-encoding']
    NUMERICAL_TRANSOFORMS = [None, 'min-max', 'standard', 'quantile-normal', 'quantile-normal', 'max-abs', 'box-cox', 'yeo-johnson']
    
    def __init__(
        self, 
        categorical_features: list[str] = None,
        numerical_features: list[str] = None,
        categorical_transform: str = 'embeddings',
        categories: list[list[str]] = None,
        numerical_transform: str = 'quantile-normal',
        use_gaussian_noise: bool = False,
        gaussian_noise_std: float = 0.01,
        max_categorical_embedding_dim: int = 50,
        verbose: bool = False, 
        model_name: str = 'model',
        random_seed: int = 42, 
        **kwargs
    ):
        super(TabularNNModel, self).__init__(verbose=verbose, **kwargs)
        
        if numerical_transform not in self.NUMERICAL_TRANSOFORMS:
            raise ValueError(f'Numerical transformation {numerical_transform} not available, choose one between {self.NUMERICAL_TRANSOFORMS}')
        if categorical_transform not in self.CATEGORICAL_TRANSFORMS:
            raise ValueError(f'Categorical encoding {categorical_transform} not available, choose one between {self.CATEGORICAL_TRANSFORMS}')
        if len(categorical_features) + len(numerical_features) == 0:
            raise ValueError('Provide at least one numerical feature or one categorical feature')
        if len(categories) < len(categorical_features):
            raise ValueError('Need to specify the categories of all the categorical features')
        
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.numerical_features = numerical_features if numerical_features is not None else []
        self.categorical_transform = categorical_transform
        self.categories = categories
        self.numerical_transform = numerical_transform
        self.max_categorical_embedding_dim = max_categorical_embedding_dim
        self.use_gaussian_noise = use_gaussian_noise
        self.gaussian_noise_std = gaussian_noise_std
        self.random_seed = random_seed
        self.model_name = model_name
        self.model: tfk.Model = None
        if len(self.categorical_features) > 0:
            self._build_numerical_transformer()
        if len(self.numerical_features) > 0:
            self._build_categorical_encoder()

    def __call__(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def fit(
        self, 
        X: pd.DataFrame, 
        y,
        validation_data: Tuple = None,
        early_stopping_rounds: int = 1,
        batch_size: int = 256,
        epochs: int = 1,
        early_stopping_monitor: str = 'val_loss',
        early_stopping_mode: str = 'auto',
        lr_scheduler: Union[callable, tfk.callbacks.Callback] = None,
    ):
        if self.model is None:
            raise ValueError('Model is not built, call the build method (or load the model from file)')
        
        inputs = []
        if len(self.numerical_features) > 0:
            if self.xformer is not None:
                X_train_numerical = self.xformer.fit_transform(X[self.numerical_features], y)
            else:
                X_train_numerical = X[self.numerical_features].values
            inputs.append(X_train_numerical)
            
        if len(self.categorical_features) > 0:
            X_train_categorical = self.encoder.fit_transform(X[self.categorical_features], y)
            inputs.append(X_train_categorical)
        
        if len(inputs) == 1:
            inputs = inputs[0]
            
        if validation_data is not None:
            X_val, y_val = validation_data
            validation_data = (self._transform_test_data(X_val), y_val)
                
        early_stopping = tfk.callbacks.EarlyStopping(
            monitor=early_stopping_monitor, 
            patience=early_stopping_rounds, 
            mode=early_stopping_mode, 
            restore_best_weights=True
        )
        
        if lr_scheduler is not None:
            if type(lr_scheduler) == tfk.callbacks.Callback:
                callbacks = [early_stopping, lr_scheduler]
            elif type(lr_scheduler) == callable:
                scheduler = tfk.callbacks.LearningRateScheduler(lr_scheduler)
                callbacks = [early_stopping, scheduler]   
        else:
            callbacks = [early_stopping]
            
        self.model.fit(
            X=X,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks
        )
        
    def predict(self, X, **kwargs):
        return self.model.predict(self._transform_test_data(X), **kwargs)
    
    def compile(self, optimizer, loss, metrics, **kwargs):
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            **kwargs
        )
        
    def summary(self, expand_nested=True, **kwargs):
        self.model.summary(expand_nested=expand_nested, **kwargs)

    def save(self, directory):
        self.model.save(os.path.join(directory, self.model_name))
        if self.categorical_transform is not None:
            joblib.dump(self.categorical_transform, os.path.join(directory, 'categorical_encoder.joblib'))
        if self.xformer is not None:
            joblib.dump(self.xformer, os.path.join(directory, 'numerical_transformer.joblib'))

    def load(self, directory):
        self.model = tfk.models.load_model(os.path.join(directory, self.model_name))
        if len(self.numerical_features) > 0:
            try:
                self.xformer = joblib.load(os.path.join(directory, 'numerical_transformer.joblib'))
            except Exception:
                raise ValueError(f'Numerical transformer not found in {directory}')
        if len(self.categorical_features) > 0:
            try:
                self.categorical_transform = joblib.load(os.path.join(directory, 'categorical_encoder.joblib'))
            except Exception:
                raise ValueError(f'Categorical encoder not found in {directory}')
        
    @abstractmethod
    def build(self):
        raise NotImplementedError('Method build not implemented')
    
    @abstractmethod
    @classmethod
    def get_optuna_trial(cls, trial: optuna.Trial):
        raise NotImplementedError('Method get_optuna_trial not implemented')   
        
    def _transform_test_data(self, X: pd.DataFrame):
        inputs = []
        if len(self.numerical_features) == 0:
            if self.xformer is not None:
                X_numerical = self.xformer.transform(X[self.numerical_features])
            else:
                X_numerical = X[self.numerical_features]
            inputs.append(X_numerical)    
        if len(self.categorical_features) == 0:
            X_categorical = self.encoder.transform(X[self.categorical_features])
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
        if len(self.categorical_features) > 0:
            numeric_input_layer = tfkl.Input(shape=(len(self.numerical_features)), name='NumericInput')
            inputs.append(numeric_input_layer)
        
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
                inputs += embedding_layers
                
            else:
                categorical_input = tfkl.Input(
                    shape=(self.categorical_encoding_shape,), 
                    name='CategoricalInput',
                )
                inputs.append(categorical_input)
                
        if len(inputs) > 1:
            encoded_inputs = tfkl.Concatenate()(inputs)
        else:
            encoded_inputs = inputs[0]
            
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
        elif self.numerical_transform == 'box-cox':
            self.xformer = PowerTransformer(method='box-cox')
        elif self.numerical_transform == 'yeo-johnson':
            self.xformer = PowerTransformer(method='yeo-johnson')
        
    def _build_categorical_encoder(self):
        if self.categorical_transform == 'embeddings':
            self.encoder = OrdinalEncoder(categories=self.categories)
            self.vocabulary_sizes = {
                self.categorical_features[i]: len(self.categories[i]) for i in range(len(self.categorical_features))
            }
        elif self.categorical_transform == 'target-encoding':
            self.encoder = TargetEncoder(target_type='binary', categories=self.categories)
            self.categorical_encoding_shape = len(self.categorical_features)
        elif self.categorical_transform == 'one-hot-encoding':
            self.encoder = OneHotEncoder(drop='first', sparse_output=False, categories=self.categories)
            self.categorical_encoding_shape = sum([len(cat) for cat in self.categories])