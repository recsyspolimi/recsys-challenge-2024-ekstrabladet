import os
import logging
from pathlib import Path
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
from tqdm import tqdm
from polimi.utils._catboost import subsample_dataset


import sys
sys.path.append('/home/ubuntu/RecSysChallenge2024/src')
sys.path.append('/home/ubuntu/fastauc')

from ebrec.evaluation.metrics_protocols import *
from polimi.utils._tuning_params import get_models_params
from polimi.utils.model_wrappers import FastRGFClassifierWrapper
from fastauc.fastauc.fast_auc import CppAuc, fast_numba_auc

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


def optimize_parameters(folds_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pl.DataFrame, pd.DataFrame],
                        categorical_features: List[str], model_class: Type = CatBoostClassifier, use_gpu: bool = False,
                        study_name: str = 'catboost_cls_tuning', n_trials: int = 100, storage: str = None) -> Tuple[Dict, pd.DataFrame]:
    
    def objective_function(trial: optuna.Trial):
        logging.info('GPU enabled' if use_gpu else 'GPU disabled')
        params = get_models_params(trial, model_class, categorical_features, use_gpu=use_gpu)
        
        auc = []
        for X_train, X_val, y_train, evaluation_ds, groups in folds_data:
            X_val = X_val[X_train.columns] # In case of wrong order of features
            if model_class in [XGBClassifier, XGBRanker]: # XGBoost does not support inf values
                X_train = X_train.replace([-np.inf, np.inf], np.nan)
                X_val = X_val.replace([-np.inf, np.inf], np.nan)
            
            model = model_class(**params)
            
        
            if model_class == CatBoostRanker:
                model.fit(X_train, y_train, group_id=groups['impression_id'], verbose=100)
            elif model_class in [XGBRanker, LGBMRanker]:
                model.fit(X_train, y_train, group=groups.groupby('impression_id').size().values)
            elif model_class == LGBMClassifier:
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train, verbose=100)
                
                             
            if model_class in [CatBoostRanker, XGBRanker, LGBMRanker]:    
                prediction_ds = evaluation_ds.with_columns(pl.Series(model.predict(X_val)).alias('prediction')) \
                    .group_by('impression_id').agg(pl.col('target'), pl.col('prediction'))
            else:
                prediction_ds = evaluation_ds.with_columns(pl.Series(model.predict_proba(X_val)[:, 1]).alias('prediction')) \
                    .group_by('impression_id').agg(pl.col('target'), pl.col('prediction'))
                    
            auc_fold = np.mean(
                [fast_numba_auc(np.array(y_t).astype(bool), np.array(y_s).astype(np.float32)) 
                 for y_t, y_s in zip(prediction_ds['target'].to_list(), 
                                     prediction_ds['prediction'].to_list())]
            )
            auc.append(auc_fold)
            logging.info(f'Fold AUC: {auc_fold}')
        
        trial.set_user_attr('folds_auc', auc)
        return np.mean(auc)
        
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)
    study.optimize(objective_function, n_trials=n_trials, n_jobs=1)
    return study.best_params, study.trials_dataframe()


def load_folds(folds_path, dataset_path, is_ranking=False):
    logging.info(f"Loading the folded dataset from {folds_path}")
    folds_path = Path(folds_path)
    dataset_path = Path(dataset_path)
    fold_folders = sorted(
        [folder.name for folder in folds_path.glob('fold_*') if folder.is_dir()],
        key=lambda x: int(x.split('_')[-1])
    )
    train_ds_name = 'train_ds.parquet' if is_ranking else 'train_ds_subsample.parquet'
    
    with open(os.path.join(folds_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
    
    folds_data = []
    for fold in tqdm(fold_folders):
        fold_path = folds_path / fold
        parquet_files = [path.name for path in fold_path.glob('*.parquet')]
        assert 'train_ds.parquet' in parquet_files, 'The train_ds.parquet file is missing'
        
        train_ds_name = 'train_ds.parquet'
        if not is_ranking and not 'train_ds_subsample.parquet' in parquet_files:
            logging.info('Subsampling the train dataset...')
            subsample_dataset(dataset_path / 'train' / 'behaviors.parquet', fold_path / 'train_ds.parquet', fold_path / 'train_ds_subsample.parquet')
            train_ds_name = 'train_ds_subsample.parquet'
            logging.info('Subsampling complete')
            
            
        train_ds = pl.read_parquet(fold_path / train_ds_name)
        val_ds = pl.read_parquet(fold_path / 'validation_ds.parquet')
        
        if 'postcode' in train_ds.columns:
            train_ds = train_ds.with_columns(pl.col('postcode').fill_null(5))
            val_ds = val_ds.with_columns(pl.col('postcode').fill_null(5))
        if 'article_type' in train_ds.columns:
            train_ds = train_ds.with_columns(pl.col('article_type').fill_null('article_default'))
            val_ds = val_ds.with_columns(pl.col('article_type').fill_null('article_default'))
        if 'impression_time' in train_ds.columns:
            train_ds = train_ds.drop(['impression_time'])
            val_ds = val_ds.drop(['impression_time'])
            
        if is_ranking:
            train_ds = train_ds.sort(by='impression_id')
            
        group_ids = train_ds.select(['impression_id']).to_pandas()
        X_train = train_ds.drop(['impression_id', 'article', 'user_id', 'target']).to_pandas()
        X_train[data_info['categorical_columns']] = X_train[data_info['categorical_columns']].astype('category')
        y_train = train_ds.select(['target']).to_pandas()['target']
        
        X_val = val_ds.drop(['impression_id', 'article', 'user_id', 'target']).to_pandas()
        X_val[data_info['categorical_columns']] = X_val[data_info['categorical_columns']].astype('category')
        evaluation_ds = val_ds.select(['impression_id', 'article', 'user_id', 'target'])
        
        folds_data.append((X_train, X_val, y_train, evaluation_ds, group_ids))
        
    logging.info(f'Features ({len(X_train.columns)}): {np.array(list(X_train.columns))}')
    logging.info(f'Categorical features: {np.array(data_info["categorical_columns"])}')
    
    return folds_data, data_info["categorical_columns"]


def main(folds_path: str, dataset_path:str, output_dir: str, model_name: str, use_gpu:bool=False,
         is_ranking: bool = False, study_name: str = 'lightgbm_tuning', n_trials: int = 100, storage: str = None):
    # X, y, evaluation_ds, group_ids, cat_features = load_datasets(train_dataset_path, validation_dataset_path)
    folds_data, cat_features = load_folds(folds_path, dataset_path, is_ranking)
    model_class = get_model_class(model_name, is_ranking)
    
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr (prevents double logs)

    best_params, trials_df = optimize_parameters(folds_data, categorical_features=cat_features, model_class=model_class,
                                                 study_name=study_name, n_trials=n_trials, storage=storage, use_gpu=use_gpu)
    
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
                        help="The directory where the logs will be placed")
    parser.add_argument("-folds_dataset_path", default=None, type=str, required=True,
                        help="The root path of the preprocessed folds")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="The root path of the preprocessed folds")
    parser.add_argument("-n_trials", default=100, type=int, required=False,
                        help="Number of optuna trials to perform")
    parser.add_argument("-study_name", default=None, type=str, required=False,
                        help="Optional name of the study. Should be used if a storage is provided")
    parser.add_argument("-storage", default=None, type=str, required=False,
                        help="Optional storage url for saving the trials")
    parser.add_argument("-model_name", choices=['lgbm', 'catboost', 'fast_rgf', 'xgb'], 
                        type=str, default='catboost_cls', help='The type of model to tune')
    parser.add_argument('--is_rank', action='store_true', default=False, 
                        help='Whether to treat the problem as a ranking problem')
    parser.add_argument('--use_gpu', action='store_true', default=False, 
                        help='Whether to train the tree model using GPUs')
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    FOLDS_DATASET_DIR = args.folds_dataset_path
    DATASET_PATH = args.dataset_path
    N_TRIALS = args.n_trials
    STUDY_NAME = args.study_name
    STORAGE = args.storage
    MODEL_NAME = args.model_name
    IS_RANK = args.is_rank
    USE_GPU = args.use_gpu
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'{MODEL_NAME}_tuning_{timestamp}_cv' if not IS_RANK else f'{MODEL_NAME}_ranker_tuning_{timestamp}_cv'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(FOLDS_DATASET_DIR, DATASET_PATH, output_dir, MODEL_NAME, is_ranking=IS_RANK, 
         study_name=STUDY_NAME, n_trials=N_TRIALS, storage=STORAGE, use_gpu=USE_GPU)