import os
import logging
from lightgbm import LGBMClassifier
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

import sys
sys.path.append('/home/ubuntu/RecSysChallenge2024/src')

from ebrec.evaluation.metrics_protocols import *
from polimi.utils._custom import get_models_params
from polimi.utils.model_wrappers import FastRGFClassifierWrapper

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def get_model_class(name: str = 'catboost', ranker: bool = False):
    if name == 'catboost':
        return CatBoostClassifier if not ranker else CatBoostRanker
    elif name == 'xgb':
        return XGBClassifier if not ranker else XGBRanker
    elif name == 'fast_rgf':
        if ranker:
            logging.log('RGF do not support ranking problems, param is_rank will be ignored')
        return FastRGFClassifierWrapper


def optimize_parameters(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, evaluation_ds: pl.DataFrame, 
                        categorical_features: List[str], group_ids: pd.DataFrame, model_class: Type = CatBoostClassifier, 
                        study_name: str = 'catboost_cls_tuning', n_trials: int = 100, storage: str = None) -> Tuple[Dict, pd.DataFrame]:
    '''
    The X_train dataframe must be sorted by the impression_id for the ranking problems
    '''
    
    def objective_function(trial: optuna.Trial):
        params = get_models_params(trial, model_class, categorical_features)
        model = model_class(**params)
        if model_class == CatBoostRanker:
            model.fit(X_train, y_train, group_id=group_ids['impression_id'], verbose=False)
        elif model_class == XGBRanker:
            model.fit(X_train, y_train, group=group_ids.groupby('impression_id').count().values, verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)
        prediction_ds = evaluation_ds.with_columns(pl.Series(model.predict_proba(X_val)[:, 1]).alias('prediction')) \
            .group_by('impression_id').agg(pl.col('target'), pl.col('prediction'))
        met_eval = MetricEvaluator(
            labels=prediction_ds['target'].to_list(),
            predictions=prediction_ds['prediction'].to_list(),
            metric_functions=[AucScore()]
        )
        return met_eval.evaluate().evaluations['auc']
        
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)
    study.optimize(objective_function, n_trials=n_trials, n_jobs=-1)
    return study.best_params, study.trials_dataframe()
    

def load_datasets(train_dataset_path, validation_dataset_path):
    logging.info(f"Loading the training dataset from {train_dataset_path}")
    
    train_ds = pd.read_parquet(os.path.join(train_dataset_path, 'train_ds.parquet')).sort_values(by=['impression_id'])
    group_ids = train_ds['impression_id'].to_frame()
    with open(os.path.join(train_dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Data info: {data_info}')
    
    train_ds = train_ds.drop(columns=['impression_id', 'article', 'user_id', 'impression_time'])
    
    # Delete cateegories percentage columns
    train_ds[data_info['categorical_columns']] = train_ds[data_info['categorical_columns']].astype('category')
    
    X_train = train_ds.drop(columns=['target'])
    y_train = train_ds['target']
    
    logging.info(f'Features ({len(X_train.columns)}): {np.array(list(X_train.columns))}')
    logging.info(f'Categorical features: {np.array(data_info["categorical_columns"])}')
    
    logging.info(f"Loading the validation dataset from {validation_dataset_path}")
    
    val_ds = pd.read_parquet(os.path.join(validation_dataset_path, 'validation_ds.parquet'))
    val_ds[data_info['categorical_columns']] = val_ds[data_info['categorical_columns']].astype('category')

    X_val = val_ds[X_train.columns]
    evaluation_ds = pl.from_pandas(val_ds[['impression_id', 'article', 'target']])
    return X_train, y_train, X_val, evaluation_ds, group_ids, data_info['categorical_columns']


def main(train_dataset_path: str, validation_dataset_path: str, output_dir: str, model_name: str,
         is_ranking: bool = True, study_name: str = 'lightgbm_tuning', n_trials: int = 100, storage: str = None):
    X_train, y_train, X_val, evaluation_ds, group_ids, cat_features = load_datasets(train_dataset_path, validation_dataset_path)
    model_class = get_model_class(model_name, is_ranking)
    
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr (prevents double logs)

    best_params, trials_df = optimize_parameters(X_train=X_train, y_train=y_train, X_val=X_val, evaluation_ds=evaluation_ds,
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
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'Lightgbm_Tuning_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(TRAIN_DATASET_DIR, VALIDATION_DATASET_DIR, output_dir, study_name=STUDY_NAME, n_trials=N_TRIALS, storage=STORAGE)