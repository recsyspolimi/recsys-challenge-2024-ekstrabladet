import optuna
from typing_extensions import List, Type
from lightgbm import LGBMClassifier, LGBMRanker
from catboost import CatBoostClassifier, CatBoostRanker
from xgboost import XGBClassifier, XGBRanker
from polimi.utils.model_wrappers import FastRGFClassifierWrapper


def get_models_params(trial: optuna.Trial, model: Type, categorical_columns: List[str] = None, random_seed: int = 42, use_gpu: bool = False):
    if model in [LGBMClassifier, LGBMRanker]:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 5000, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 8, 1024),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.1, 0.7),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.8),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1000, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1000, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-6, 1, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-7, 1e-1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 10000, log=True),
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
            "random_seed": random_seed,
            "device_type": "gpu" if use_gpu else "cpu",
            "verbose": -1
        }
        if use_gpu:
            params['max_bin'] = 63
            params['gpu_use_dp'] = False
            
        else:
            params["max_bin"] = trial.suggest_int("max_bin", 8, 512, log=True)

    elif model in [CatBoostClassifier, CatBoostRanker]:
        params = {
            'iterations': trial.suggest_int('iterations', 100, 5000, step=10),
            'learning_rate': trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
            'reg_lambda': trial.suggest_float("reg_lambda", 1e-5, 1e5, log=True),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bernoulli', 'MVS']),
            'subsample': trial.suggest_float("subsample", 0.05, 0.7),
            'fold_permutation_block': trial.suggest_int('fold_permutation_block', 1, 100),
            'border_count': trial.suggest_int('border_count', 8, 512, log=True),
            # 'has_time': trial.suggest_categorical('has_time', [True, False]),
            'cat_features': categorical_columns,
            'random_seed': random_seed,
            'task_type': 'GPU' if use_gpu else 'CPU',
        }
        
        if params['grow_policy'] == 'Lossguide':
            params['max_leaves'] = trial.suggest_int("max_leaves", 8, 64, log=True)
            params['depth'] = trial.suggest_int("depth", 2, 14)
            if not use_gpu:
                params['langevin'] = trial.suggest_categorical("langevin", [True, False])
                if params['langevin']:
                    params['diffusion_temperature'] = trial.suggest_float('diffusion_temperature', 1e2, 1e6, log=True)
        else: # for Lossguide, Cosine is not supported. Newton and NewtonL2 are only supported in GPU
            if not use_gpu:
                params['sampling_frequency'] = trial.suggest_categorical('sampling_frequency', ['PerTree', 'PerTreeLevel'])
                params['score_function'] = trial.suggest_categorical('score_function', ['Cosine', 'L2'])
            
            params['depth'] = trial.suggest_int("depth", 2, 10)

        if params['grow_policy'] != 'SymmetricTree':
            params['min_data_in_leaf'] = trial.suggest_float('min_data_in_leaf', 10, 1000)     
                
        if params['bootstrap_type'] == 'MVS' and not use_gpu:
            params['mvs_reg'] = trial.suggest_float('mvs_reg', 1e-4, 1e4, log=True)
        
        if not use_gpu or (use_gpu and model == CatBoostRanker):
            params['rsm'] = trial.suggest_float("rsm", 0.05, 0.8, log=True)

        if use_gpu:     
            params['border_count'] = trial.suggest_int('border_count', 8, 254, log=True) # suggested to be small on GPU
            if model == CatBoostRanker:
                params['depth'] = trial.suggest_int("depth", 2, 8)
        else:
            params['random_strength'] = trial.suggest_float('random_strength', 1e-6, 1e2, log=True)
     
    elif model in [XGBClassifier, XGBRanker]:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 5000, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1000, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1000, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'max_leaves': trial.suggest_int('max_leaves', 8, 1024),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'max_bin': trial.suggest_int('max_bin', 8, 512),
            'gamma': trial.suggest_float('gamma', 1e-7, 10, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-7, 10, log=True),
            'subsample': trial.suggest_float('subsample', 0.05, 0.5),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.8),
            "enable_categorical": True,
            "device": "gpu" if use_gpu else "cpu",
        }
            
    elif model == FastRGFClassifierWrapper:
        params = {
            "n_estimators": trial.suggest_int('n_estimators', 100, 10000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "max_leaf": trial.suggest_int("max_leaf", 8, 128),
            "tree_gain_ratio": trial.suggest_float("tree_gain_ratio", 0.1, 1),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 500),
            "l1": trial.suggest_float("l1", 1, 10000, log=True),
            "l2": trial.suggest_float("l2", 1, 100000, log=True),
            "max_bin": trial.suggest_int("max_bin", 8, 1024, log=True),
            "opt_algorithm": trial.suggest_categorical("opt_algorithm", ["rgf", "epsilon-greedy"]),
            "verbose": 0,
            "n_jobs": -1,
        }
        if params['opt_algorithm'] == 'epsilon-greedy':
            params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 0.1, log=True)
    
    else:
        raise ValueError(f'Model not recognized')
    
    return params