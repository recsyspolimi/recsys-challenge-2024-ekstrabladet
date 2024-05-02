import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from .base_model import TabularNNModel
from .layers import AbstractLayer
from optuna import Trial


class DeepAbstractNetwork(TabularNNModel):
    '''
    https://arxiv.org/abs/2112.02962
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
        d0: int = 32, 
        d1: int = 64, 
        k0: int = 4, 
        use_bias: bool = True,
        num_basic_blocks: int = 8,
        dropout_rate: float = 0.1,
        l1_lambda: float = 1e-4,
        l2_lambda: float = 1e-4,
        activation: str = 'relu',
        **kwargs
    ):
        super(DeepAbstractNetwork, self).__init__(categorical_features=categorical_features, numerical_features=numerical_features,
                                                  categorical_transform=categorical_transform, numerical_transform=numerical_transform,
                                                  use_gaussian_noise=use_gaussian_noise, gaussian_noise_std=gaussian_noise_std, 
                                                  max_categorical_embedding_dim=max_categorical_embedding_dim,
                                                  verbose=verbose, model_name=model_name, random_seed=random_seed, **kwargs)
        self.d0 = d0
        self.d1 = d1
        self.k0 = k0
        self.use_bias = use_bias
        self.num_basic_blocks = num_basic_blocks
        self.dropout_rate = dropout_rate
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.activation = activation
    
    def _build(self):
        inputs, x0 = self._build_input_layers()

        f = x0
        for i in range(self.num_basic_blocks):
            f = AbstractLayer(
                d=self.d0,
                k=self.k0,
                kernel_regularizer=tfk.regularizers.L1L2(l1=self.l1_lambda, l2=self.l2_lambda),
                use_bias=self.use_bias
            )(f)
            f = AbstractLayer(
                d=self.d1,
                k=self.k0,
                kernel_regularizer=tfk.regularizers.L1L2(l1=self.l1_lambda, l2=self.l2_lambda),
                use_bias=self.use_bias
            )(f)

            x = tfkl.Dropout(self.dropout_rate)(x0)
            x = AbstractLayer(
                d=self.d1,
                k=self.k0,
                kernel_regularizer=tfk.regularizers.L1L2(l1=self.l1_lambda, l2=self.l2_lambda),
                use_bias=self.use_bias
            )(x)

            f = tfkl.Add()([f, x])
            
        outputs = tfkl.Dense(
            units=1,
            kernel_initializer=tfk.initializers.GlorotUniform(seed=self.random_seed),
            kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
            name='OutputDense',
            activation='sigmoid'
        )(f)
        self.model = tfk.Model(inputs=inputs, outputs=outputs)
        
    @classmethod
    def get_optuna_trial(cls, trial: Trial):
        params = TabularNNModel.get_optuna_trial(trial)
        model_params = {
            'd0': trial.suggest_int('d0', 16, 64),
            'd1': trial.suggest_int('d1', 64, 256),
            'k0': trial.suggest_int('k0', 1, 8),
            'use_bias': trial.suggest_categorical('use_bias', [True, False]),
            'num_basic_blocks': trial.suggest_int('num_basic_blocks', 2, 16),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.01, 0.4),
            'l1_lambda': trial.suggest_float('l1_lambda', 1e-5, 1e-2, log=True),
            'l2_lambda': trial.suggest_float('l2_lambda', 1e-5, 1e-2, log=True),
            'activation': trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh', 'swish'])
        }
        params.update(model_params)
        return params