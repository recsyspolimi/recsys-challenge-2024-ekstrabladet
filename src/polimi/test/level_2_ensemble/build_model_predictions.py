from polimi.test.level_2_ensemble.training_predict_scripts.tree_model_training import train_predict_tree_model
from polimi.test.level_2_ensemble.training_predict_scripts.nn_model_training import train_predict_nn_model
model_list = ['catboost_ranker', 'catboost_classifier', 'light_gbm_classifier']

catboost_ranker_params = catboost_params = {
    'iterations': 2000,
    'depth': 8,
    'colsample_bylevel': 0.5
}

catboost_classifier_params = {
    "iterations": 1000,
    "subsample": 0.5,
    "rsm": 0.7
}

light_gbm_classifier_params = {
    "n_estimators": 1000,
    "subsample": 0.1,
    "learning_rate": 0.01,
    "max_depth": 6,
    "colsample_bytree": 0.7
}

gandalf = {
    "model_hyperparams" : {
     'use_gaussian_noise': True,
     'numerical_transform': 'min-max',
     'gaussian_noise_std': 0.01, 
     'n_stages': 6, 
     'init_t': 0.5,
     'n_head_layers': 2, 
     'dropout_rate': 0.2, 
     'l1_lambda': 1e-4, 
     'l2_lambda': 1e-4, 
     'activation': 'swish', 
     'start_units_head': 128, 
     'head_units_decay': 2, 
    },
    'model_name': "GANDALF",
    'epochs': 20, 
    'learning_rate': 1e-3,
    'weight_decay': 5e-5, 
    'use_scheduler': False,
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
   
    