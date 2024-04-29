from optuna import Trial
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from base_model import TabularNNModel


class DeepCrossNetwork(TabularNNModel):
    '''
    Deep and Cross (DCN) combines a MLP (the Deep part) and a feature interaction branch (the Cross part). The Cross brach
    models high order interactions between the features, similarly to what a factorization machine does.
    Original paper: https://arxiv.org/abs/1708.05123
    Here, a variant of the Cross branch is implemented (DCNv2): https://arxiv.org/abs/2008.13535
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
        n_layers: int = 1,
        start_units: int = 128,
        units_decay: int = 2,
        dropout_rate: float = 0.1,
        l1_lambda: float = 1e-4,
        l2_lambda: float = 1e-4,
        activation: str = 'relu',
        **kwargs
    ):
        '''
        Args:
            n_layers (int): the number of both deep and cross layers
            start_units (int): the number of hidden units in the first deep layer
            units_decay (int): the decay to decrease the number of hidden units at each deep layer
            dropout_rate (float): the dropout rate
            l1_lambda (float): l1 regularization coefficient
            l2_lambda (float): l2 regularization coefficient
            activation (str): the activation function of the hidden layers in the deep part
        '''
        super(DeepCrossNetwork, self).__init__(categorical_features=categorical_features, numerical_features=numerical_features,
                                               categorical_transform=categorical_transform, numerical_transform=numerical_transform,
                                               use_gaussian_noise=use_gaussian_noise, gaussian_noise_std=gaussian_noise_std, 
                                               max_categorical_embedding_dim=max_categorical_embedding_dim,
                                               verbose=verbose, model_name=model_name, random_seed=random_seed, **kwargs)
        self.n_layers = n_layers
        self.start_units = start_units
        self.units_decay = units_decay
        self.dropout_rate = dropout_rate
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.activation = activation
    
    def build(self):
        inputs, x0 = self._build_input_layers()
        
        cross = x0
        for i in range(self.n_layers):
            units = cross.shape[-1]
            x = tfkl.Dense(
                units=units,
                kernel_initializer=tfk.initializers.HeNormal(seed=self.random_seed),
                kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
                name=f'CrossDense{i}'
            )(cross)
            cross = x0 * x + cross
        cross = tfkl.BatchNormalization(name='CrossBatchNormalization')(cross)

        deep = x0
        units = self.start_units
        for i in range(self.n_layers):
            deep = tfkl.Dense(
                units=units,
                kernel_initializer=tfk.initializers.HeNormal(seed=self.random_seed),
                kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
                name=f'DeepDense{i}'
            )(deep)
            deep = tfkl.BatchNormalization(name=f'DeepBatchNormalization{i}')(deep)
            deep = tfkl.Activation(self.activation, name=f'DeepActivation{i}')(deep)
            deep = tfkl.Dropout(self.dropout_rate, name=f'DeepDropout{i}')(deep)
            units = int(units / self.units_decay)

        merged = tfkl.Concatenate(name='DeepCrossConcat')([cross, deep])
        outputs = tfkl.Dense(
            units=1,
            kernel_initializer=tfk.initializers.GlorotUniform(seed=self.random_seed),
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
            name='OutputDense',
            activation='sigmoid'
        )(merged)
        self.model = tfk.Model(inputs=inputs, outputs=outputs)
        
    @classmethod
    def get_optuna_trial(cls, trial: Trial):
        params = TabularNNModel.get_optuna_trial(trial)
        model_params = {
            'n_layers': trial.suggest_int('n_layers', 1, 6),
            'start_units': trial.suggest_int('start_units', 64, 1024, log=True),
            'units_decay': trial.suggest_categorical('units_decay', [1, 1.5, 2, 2.5, 3, 3.5, 4]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.01, 0.4),
            'l1_lambda': trial.suggest_float('l1_lambda', 1e-5, 1e-2, log=True),
            'l2_lambda': trial.suggest_float('l2_lambda', 1e-5, 1e-2, log=True),
            'activation': trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh', 'swish'])
        }
        params.update(model_params)
        return params