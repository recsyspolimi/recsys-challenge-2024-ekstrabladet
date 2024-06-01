from polimi.test.level_2_ensemble.training_predict_scripts.tree_model_training import train_predict_tree_model
from polimi.test.level_2_ensemble.training_predict_scripts.nn_model_training import train_predict_nn_model
model_list = ['catboost_ranker', 'catboost_classifier', 'light_gbm_classifier']

catboost_ranker_params  = {
    'iterations': 2421,
    'learning_rate': 0.061372161824290145,
    'rsm': 0.681769606695633,
    'reg_lambda': 0.4953354255208565,
    'grow_policy': 'SymmetricTree',
    'bootstrap_type': 'MVS',
    'subsample': 0.5108219602277233,
    'random_strength': 14.089062269780399,
    'fold_permutation_block': 39,
    'border_count': 34,
    'sampling_frequency': 'PerTreeLevel',
    'score_function': 'Cosine',
    'depth': 8,
    'mvs_reg': 0.0015341832942953422
}

catboost_classifier_params = {
    "iterations": 4648, 
    "learning_rate": 0.00940692561666741,
    "rsm": 0.23234648308448685, 
    "reg_lambda": 10.122555521780928, 
    "grow_policy": "Depthwise", 
    "bootstrap_type": "Bernoulli", 
    "subsample": 0.6994875944281512, 
    "random_strength": 0.0005368413451443827, 
    "fold_permutation_block": 38, 
    "border_count": 21, 
    "sampling_frequency": "PerTree", 
    "score_function": "Cosine", 
    "depth": 10, 
    "min_data_in_leaf": 415.138664786357
}

light_gbm_classifier_params = {
    "n_estimators": 3385,
    "max_depth": 10, 
    "num_leaves": 512,
    "subsample_freq": 8, 
    "subsample": 0.6869126439955963, 
    "learning_rate": 0.01648854491388402, 
    "colsample_bytree": 0.7449015558619945, 
    "colsample_bynode": 0.39052115012765787, 
    "reg_lambda": 0.3522398194357306, 
    "reg_alpha": 2.5686967985159077, 
    "max_bin": 508, 
    "min_split_gain": 2.6638730172118534e-06, 
    "min_child_weight": 5.007894411946897e-05, 
    "min_child_samples": 1760, 
    "extra_trees": False
}

light_gbm_ranker_params = {
    "n_estimators": 4994,
    "max_depth": 11, 
    "num_leaves": 650, 
    "subsample_freq": 7, 
    "subsample": 0.6999539200776504, 
    "learning_rate": 0.010139056373959225,
    "colsample_bytree": 0.7336212288256696, 
    "colsample_bynode": 0.39667472227408895, 
    "reg_lambda": 8.372736294570737, 
    "reg_alpha": 0.0048213090396457626, 
    "max_bin": 36, 
    "min_split_gain": 0.01018486460511667, 
    "min_child_weight": 0.0005391141639526838, 
    "min_child_samples": 2607, 
    "extra_trees": False
}

gandalf_params = {
    "model_hyperparams" : {
        "use_gaussian_noise": False, 
        "numerical_transform": "yeo-johnson", 
        "n_stages": 8, 
        "init_t": 0.6017092415733495,
        "n_head_layers": 2, 
        "dropout_rate": 0.06854356318895957, 
        "l1_lambda": 2.4646537353178322e-05,
        "l2_lambda": 6.722760166896771e-05, 
        "activation": "swish",
        "start_units_head": 96,
        "head_units_decay": 3
    },
    'model_name': "GANDALF",
    "epochs": 29, 
    "learning_rate": 0.0001787433082708194, 
    "weight_decay": 0.0018302703085207681, 
    "use_scheduler": False
}
    
   
deep_cross_params = {
    "model_hyperparams" : {
        "use_gaussian_noise": False, 
        "numerical_transform": "yeo-johnson", 
        "n_layers": 3, 
        "start_units": 404, 
        "units_decay": 3, 
        "dropout_rate": 0.13008661411038497, 
        "l1_lambda": 3.808760427408383e-05, 
        "l2_lambda": 0.004500425434981053, 
        "activation": "relu", 
    },
    'model_name': "DeepCrossNetwork",
    "epochs": 24, 
    "learning_rate": 0.00011227893464714229, 
    "weight_decay": 0.008084158754106566, 
    "use_scheduler": True, 
    "scheduling_rate": 0.08396521698755141
}

mlp_params = {
    "model_hyperparams": {
        "use_gaussian_noise": False, 
        "numerical_transform": "yeo-johnson", 
        "n_layers": 5, 
        "start_units": 398, 
        "units_decay": 2, 
        "dropout_rate": 0.16629782368872437, 
        "l1_lambda": 3.160844423348517e-05, 
        "l2_lambda":  1.9150885985013116e-05, 
        "activation": "swish"
    },
    "model_name": "MLP",
    "epochs": 21, 
    "learning_rate": 0.0005126659910783247, 
    "weight_decay": 0.0041951618430015394, 
    "use_scheduler": True, 
    "scheduling_rate": 0.07932215771838395
}

wide_deep_params = {
    "model_hyperparams" : {
     'use_gaussian_noise': False,
     'numerical_transform': 'yeo-johnson',
     'dropout_rate': 0.21578272032382537, 
     'l1_lambda': 1.0525685132669135e-05, 
     'l2_lambda': 2.3488147642445815e-05, 
     'activation': 'relu',
     'n_layers' : 2,
     'start_units': 65, 
     'units_decay': 1, 
    },
    'model_name': "WideDeepNetwork",
    'epochs': 45, 
    'learning_rate': 0.008521401469080741,
    'weight_decay': 0.003750084600313782, 
    'use_scheduler': True,
    'scheduling_rate': 0.07856145738509372
}

MODEL_DICT = {
    'catboost_ranker' : {
                            'model_class' : 'catboost',
                            'type' : 'tree_model',
                            'ranker' : True ,
                            'subsample' : False,
                            'param' : catboost_ranker_params,
                        },
    'catboost_classifier' : {
                                'model_class' : 'catboost',
                                'type' : 'tree_model',
                                'ranker' : False ,
                                'subsample' : True,
                                'param' : catboost_classifier_params,
                            },
    'light_gbm_classifier' : {
                                'model_class' : 'lgbm',
                                'type' : 'tree_model',
                                'ranker' : False ,
                                'subsample' : True,
                                'param' : light_gbm_classifier_params,
                            },
    
    'light_gbm_ranker' : {
                                'model_class' : 'lgbm',
                                'type' : 'tree_model',
                                'ranker' : True ,
                                'subsample' : False,
                                'param' : light_gbm_ranker_params,
                            },
    'mlp' : {
                                'model_class' : 'MLP',
                                'type' : 'nn_model',
                                'subsample' : True,
                                'param' : mlp_params,
                            },
    'GANDALF' : {
                                'model_class' : 'GANDALF',
                                'type' : 'nn_model',
                                'subsample' : True,
                                'param' : gandalf_params,
                            },
    'dcn' : {
                                'model_class' : 'DeepCrossNetwork',
                                'type' : 'nn_model',
                                'subsample' : True,
                                'param' : deep_cross_params,
                            },
    'wd' : {
                                'model_class' : 'WideDeepNetwork',
                                'type' : 'nn_model',
                                'subsample' : True,
                                'param' : deep_cross_params,
                            },
    
}

def get_hyperparameters(model_name):
    if model_name not in MODEL_DICT.keys():
        raise 'Wrong name'
    else: 
        return MODEL_DICT[model_name]['param']


def require_subsampled_set(model_name):
    if model_name not in MODEL_DICT.keys():
        raise 'Wrong name'
    else : return MODEL_DICT[model_name]['subsample']
    

def train_predict_model(train_ds, val_ds, data_info, model_name):
    
    if model_name not in MODEL_DICT.keys():
        raise 'Wrong name'
    if  MODEL_DICT[model_name]['type'] == 'tree_model':
        predictions = train_predict_tree_model(train_ds, val_ds, data_info, 
                                               MODEL_DICT[model_name]['model_class'],  MODEL_DICT[model_name]['ranker'], 
                                               get_hyperparameters(model_name))
        
    elif MODEL_DICT[model_name]['type'] == 'nn_model':
        predictions = train_predict_nn_model(train_ds, val_ds, data_info, get_hyperparameters(model_name))
    else :
        raise 'Wrong name' 
    
    return predictions
   
    