import os
import logging
from lightgbm import LGBMClassifier, LGBMRanker
from datetime import datetime
import argparse
import pandas as pd
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing_extensions import List, Tuple, Dict, Type
import optuna
import polars as pl
from catboost import CatBoostClassifier, CatBoostRanker
from xgboost import XGBClassifier, XGBRanker
from sklearn.model_selection import GroupKFold

import sys
sys.path.append('/home/ubuntu/RecSysChallenge2024/src')
sys.path.append('/home/ubuntu/fastauc')

from ebrec.evaluation.metrics_protocols import *
from polimi.utils._tuning_params import get_models_params
from polimi.utils.model_wrappers import FastRGFClassifierWrapper
from fastauc.fast_auc import CppAuc

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def get_model_class(name: str = 'catboost', ranker: bool = False):
    if name == 'catboost':
        return CatBoostClassifier if not ranker else CatBoostRanker
    if name == 'lgbm':
        return LGBMClassifier if not ranker else LGBMRanker
    elif name == 'xgb':
        return XGBClassifier if not ranker else XGBRanker
    elif name == 'fast_rgf':
        if ranker:
            logging.log('RGF do not support ranking problems, param is_rank will be ignored')
        return FastRGFClassifierWrapper   


def optimize_parameters(X: pd.DataFrame, y: pd.DataFrame, evaluation_ds: pd.DataFrame, group_ids: pd.DataFrame,
                        categorical_features: List[str], model_class: Type = CatBoostClassifier, n_folds: int = 5,
                        study_name: str = 'catboost_cls_tuning', n_trials: int = 100, storage: str = None) -> Tuple[Dict, pd.DataFrame]:
    
    folds = []
    for train_idx, test_idx in GroupKFold(n_splits=n_folds).split(X, y, groups=X['user_id']):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[test_idx].drop(columns=['impression_id', 'user_id'])
            y_val = y.iloc[test_idx]
            fold_evaluation_ds = pl.from_pandas(evaluation_ds.iloc[test_idx])
            groups = group_ids.iloc[train_idx]
            
            if model_class in [CatBoostRanker, XGBRanker, LGBMRanker]:
                sorted_indices = X_train.sort_values(by='impression_id').index.values
                X_train = X_train.loc[sorted_indices]
                y_train = y_train.loc[sorted_indices]
                groups = groups.loc[sorted_indices]
            
            X_train = X_train.drop(columns=['impression_id', 'user_id'])
            folds.append((X_train, X_val, y_train, groups, fold_evaluation_ds))
    
    def objective_function(trial: optuna.Trial):
        params = get_models_params(trial, model_class, categorical_features)
        
        auc = []
        for X_train, X_val, y_train, groups, evaluation_ds in folds:
            model = model_class(**params)
        
            if model_class == CatBoostRanker:
                model.fit(X_train, y_train, group_id=groups['impression_id'], verbose=100)
            elif model_class in [XGBRanker, LGBMRanker]:
                model.fit(X_train, y_train, group=groups.groupby('impression_id').size().values, verbose=100)
            else:
                model.fit(X_train, y_train, verbose=50)
            if model_class in [CatBoostRanker, XGBRanker, LGBMRanker]:
                prediction_ds = evaluation_ds.with_columns(pl.Series(model.predict(X_val)).alias('prediction')) \
                    .group_by('impression_id').agg(pl.col('target'), pl.col('prediction'))
            else:
                prediction_ds = evaluation_ds.with_columns(pl.Series(model.predict_proba(X_val)[:, 1]).alias('prediction')) \
                    .group_by('impression_id').agg(pl.col('target'), pl.col('prediction'))
                    
            cpp_auc = CppAuc()
            auc_fold = np.mean(
                [cpp_auc.roc_auc_score(np.array(y_t).astype(bool), np.array(y_s).astype(np.float32)) 
                 for y_t, y_s in zip(prediction_ds['target'].to_list(), 
                                     prediction_ds['prediction'].to_list())]
            )
            auc.append(auc_fold)
        return np.mean(auc)
        
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)
    study.optimize(objective_function, n_trials=n_trials, n_jobs=1)
    return study.best_params, study.trials_dataframe()
    

def load_datasets(train_dataset_path, validation_dataset_path):
    logging.info(f"Loading the training dataset from {train_dataset_path}")
    
    train_ds = pd.read_parquet(os.path.join(train_dataset_path, 'train_ds.parquet'))
    group_ids = train_ds['impression_id'].to_frame()
    with open(os.path.join(train_dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Data info: {data_info}')
    
    train_ds['impression_id'] = train_ds['impression_id'].astype(str).apply(lambda x: x + '_1')
    train_ds['user_id'] = train_ds['user_id'].astype(str).apply(lambda x: x + '_1')
    if 'impression_time' in train_ds.columns:
        train_ds = train_ds.drop(columns=['impression_time'])
        
    logging.info(f"Loading the validation dataset from {validation_dataset_path}")
    
    val_ds = pd.read_parquet(os.path.join(validation_dataset_path, 'validation_ds.parquet'))[train_ds.columns]
    val_ds['impression_id'] = val_ds['impression_id'].astype(str).apply(lambda x: x + '_2')
    val_ds['user_id'] = val_ds['user_id'].astype(str).apply(lambda x: x + '_2')
    val_ds[data_info['categorical_columns']] = val_ds[data_info['categorical_columns']].astype('category')
    
    complete_ds = pd.concat([train_ds, val_ds], ignore_index=True, axis=0)
    if 'postcode' in complete_ds.columns:
        complete_ds['postcode'] = complete_ds['postcode'].fillna(5).astype(int)
    if 'article_type' in complete_ds.columns:
        complete_ds['article_type'] = complete_ds['article_type'].fillna('article_default')
    complete_ds[data_info['categorical_columns']] = complete_ds[data_info['categorical_columns']].astype('category')
    # impression_id kept to sort values if it is a ranking problem, user_id will be useful for group kfold
    X = complete_ds.drop(columns=['article', 'target'])
    y = complete_ds['target']
    
    evaluation_ds = complete_ds[['impression_id', 'article', 'user_id', 'target']]
    group_ids = complete_ds['impression_id'].to_frame()
    
    logging.info(f'Features ({len(X.columns)}): {np.array(list(X.columns))}')
    logging.info(f'Categorical features: {np.array(data_info["categorical_columns"])}')
    return X, y, evaluation_ds, group_ids, data_info['categorical_columns']


def main(train_dataset_path: str, validation_dataset_path: str, output_dir: str, model_name: str, n_folds: int = 5,
         is_ranking: bool = False, study_name: str = 'lightgbm_tuning', n_trials: int = 100, storage: str = None):
    X, y, evaluation_ds, group_ids, cat_features = load_datasets(train_dataset_path, validation_dataset_path)
    model_class = get_model_class(model_name, is_ranking)
    
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr (prevents double logs)

    best_params, trials_df = optimize_parameters(X=X, y=y, evaluation_ds=evaluation_ds, n_folds=n_folds,
                                                 categorical_features=cat_features, group_ids=group_ids, model_class=model_class,
                                                 study_name=study_name, n_trials=n_trials, storage=storage)
    
    params_file_path = os.path.join(output_dir, 'lightgbm_best_params.json')
    logging.info(f'Best parameters: {best_params}')
    logging.info(f'Saving the best parameters at: {params_file_path}')
    with open(params_file_path, 'w') as params_file:
        json.dump(best_params, params_file)    
        
    trials_file_path = os.path.join(output_dir, 'trials_dataframe.csv')
    logging.info(f'Saving the trials dataframe at: {trials_file_path}')
    trials_df.to_csv(trials_file_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for catboost")
    parser.add_argument("-output_dir", default="/home/ubuntu/experiments/", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-training_dataset_path", default=None, type=str, required=True,
                        help="Parquet file where the training dataset is placed")
    parser.add_argument("-validation_dataset_path", default=None, type=str, required=True,
                        help="Parquet file where the validation dataset is placed")
    parser.add_argument("-n_folds", default=5, type=int, required=False,
                        help="Number of folds for cross validation")
    parser.add_argument("-n_trials", default=100, type=int, required=False,
                        help="Number of optuna trials to perform")
    parser.add_argument("-study_name", default=None, type=str, required=False,
                        help="Optional name of the study. Should be used if a storage is provided")
    parser.add_argument("-storage", default=None, type=str, required=False,
                        help="Optional storage url for saving the trials")
    parser.add_argument("-model_name", choices=['catboost', 'fast_rgf', 'xgb'], 
                        type=str, default='catboost_cls', help='The type of model to tune')
    parser.add_argument('--is_rank', action='store_true', default=False, 
                        help='Whether to treat the problem as a ranking problem')
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    TRAIN_DATASET_DIR = args.training_dataset_path
    VALIDATION_DATASET_DIR = args.validation_dataset_path
    N_TRIALS = args.n_trials
    STUDY_NAME = args.study_name
    STORAGE = args.storage
    MODEL_NAME = args.model_name
    IS_RANK = args.is_rank
    N_FOLDS = args.n_folds
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'{MODEL_NAME}_tuning_{timestamp}' if not IS_RANK else f'{MODEL_NAME}_ranker_tuning_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(TRAIN_DATASET_DIR, VALIDATION_DATASET_DIR, output_dir, MODEL_NAME, n_folds=N_FOLDS, is_ranking=IS_RANK, 
         study_name=STUDY_NAME, n_trials=N_TRIALS, storage=STORAGE)