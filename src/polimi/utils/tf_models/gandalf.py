import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from .base_model import TabularNNModel
from .layers import GatedFeatureLearningUnit
from optuna import Trial
import joblib
import os
import json
import logging


class GANDALF(TabularNNModel):
    '''
    https://arxiv.org/abs/2207.08548
    '''
    
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
        n_stages: int = 6,
        init_t: float = 0.5,
        n_head_layers: int = 2,
        start_units_head: int = 128,
        head_units_decay: int = 2,
        dropout_rate: float = 0.1,
        l1_lambda: float = 1e-4,
        l2_lambda: float = 1e-4,
        activation: str = 'relu',
        **kwargs
    ):
        super(GANDALF, self).__init__(categorical_features=categorical_features, numerical_features=numerical_features,
                                      categorical_transform=categorical_transform, numerical_transform=numerical_transform,
                                      use_gaussian_noise=use_gaussian_noise, gaussian_noise_std=gaussian_noise_std, 
                                      max_categorical_embedding_dim=max_categorical_embedding_dim,
                                      verbose=verbose, model_name=model_name, random_seed=random_seed, **kwargs)
        self.n_stages = n_stages
        self.init_t = init_t
        self.n_head_layers = n_head_layers
        self.start_units_head = start_units_head
        self.head_units_decay = head_units_decay
        self.dropout_rate = dropout_rate
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.activation = activation
    
    def _build(self):
        inputs, x0 = self._build_input_layers()

        x = GatedFeatureLearningUnit(
            n_stages=self.n_stages,
            dropout_rate=self.dropout_rate,
            l1=self.l1_lambda,
            l2=self.l2_lambda,
            init_t=self.init_t
        )(x0)
        x = tfkl.BatchNormalization()(x)

        units = self.start_units_head
        for i in range(self.n_head_layers):
            x = tfkl.Dense(units, kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda))(x)
            x = tfkl.BatchNormalization()(x)
            x = tfkl.Dropout(self.dropout_rate)(x)
            x = tfkl.Activation(self.activation)(x)
            units = int(units / self.head_units_decay)
            
        outputs = tfkl.Dense(
            units=1,
            kernel_initializer=tfk.initializers.GlorotUniform(seed=self.random_seed),
            kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
            name='OutputDense',
            activation='sigmoid'
        )(x)
        self.model = tfk.Model(inputs=inputs, outputs=outputs)
        
    def load(self, directory):
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
            self.vocabulary_sizes = {}
            for i, f in enumerate(self.categorical_features):
                self.vocabulary_sizes[f] = len(self.categories[i])
        
        checkpoint_dir = os.path.join(directory, 'checkpoints')
        if os.path.exists(checkpoint_dir):
            logging.info('Attempting to load the checkpoint weights')
            self._build()
            if self.categorical_transform == 'embeddings':
                input_shape = [(len(self.numerical_features),)] + [(1,)] * len(self.categorical_features)
            else:
                input_shape = [(len(self.numerical_features),)] + [(len(self.encoder.get_feature_names_out()),)]
            self.model.build(input_shape)
            self.model.load_weights(os.path.join(checkpoint_dir, 'checkpoint.weights.h5'))
        else:
            raise ValueError('GANDALF checkpoint not found')
        
    @classmethod
    def get_optuna_trial(cls, trial: Trial):
        params = TabularNNModel.get_optuna_trial(trial)
        model_params = {
            'n_stages': trial.suggest_int('n_stages', 1, 10),
            'init_t': trial.suggest_float('init_t', 0.1, 1),
            'n_head_layers': trial.suggest_int('n_head_layers', 0, 2),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.01, 0.3),
            'l1_lambda': trial.suggest_float('l1_lambda', 1e-5, 1e-2, log=True),
            'l2_lambda': trial.suggest_float('l2_lambda', 1e-5, 1e-2, log=True),
            'activation': trial.suggest_categorical('activation', ['relu', 'swish'])
        }
        params.update(model_params)
        if params['n_head_layers'] > 0:
            params['start_units_head'] = trial.suggest_int('start_units_head', 16, 192)
            params['head_units_decay'] = trial.suggest_categorical('head_units_decay', [1, 1.5, 2, 2.5, 3, 3.5, 4])
        return params