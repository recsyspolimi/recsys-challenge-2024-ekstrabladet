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
from typing_extensions import List, Tuple, Dict
import optuna
import polars as pl

import sys
sys.path.append('/home/ubuntu/RecSysChallenge2024/src')

from ebrec.evaluation.metrics_protocols import *
from polimi.preprocessing_pipelines.pre_68f import strip_new_features

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def optimize_parameters(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, evaluation_ds: pl.DataFrame, 
                        study_name: str = 'lightgbm_tuning', n_trials: int = 100, storage: str = None) -> Tuple[Dict, pd.DataFrame]:
    
    def objective_function(trial: optuna.Trial):
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
            "max_bin": trial.suggest_int("max_bin", 8, 512, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-6, 1, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-7, 1e-1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 10000, log=True),
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        }
        
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
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
    
    train_ds = pd.read_parquet(os.path.join(train_dataset_path, 'train_ds.parquet'))
    with open(os.path.join(train_dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Data info: {data_info}')
    
    train_ds = train_ds.drop(columns=['impression_id', 'article', 'user_id', 'impression_time'])
    
    # Delete cateegories percentage columns
    train_ds = strip_new_features(train_ds)
    train_ds = train_ds.drop(columns = 'impression_time')
    train_ds[data_info['categorical_columns']] = train_ds[data_info['categorical_columns']].astype('category')
    
    X_train = train_ds.drop(columns=['target'])
    y_train = train_ds['target']
    
    logging.info(f'Features ({len(X_train.columns)}): {np.array(list(X_train.columns))}')
    logging.info(f'Categorical features: {np.array(data_info["categorical_columns"])}')
    
    logging.info(f"Loading the validation dataset from {validation_dataset_path}")
    
    val_ds = pd.read_parquet(os.path.join(validation_dataset_path, 'validation_ds.parquet'))
    
    val_ds = strip_new_features(val_ds)
    val_ds = val_ds.drop(columns = 'impression_time')
    val_ds[data_info['categorical_columns']] = val_ds[data_info['categorical_columns']].astype('category')

    X_val = val_ds[X_train.columns]
    evaluation_ds = pl.from_pandas(val_ds[['impression_id', 'article', 'target']])
    return X_train, y_train, X_val, evaluation_ds


def main(train_dataset_path: str, validation_dataset_path: str, output_dir: str,
         study_name: str = 'lightgbm_tuning', n_trials: int = 100, storage: str = None):
    X_train, y_train, X_val, evaluation_ds = load_datasets(train_dataset_path, validation_dataset_path)
    
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr (prevents double logs)

    best_params, trials_df = optimize_parameters(X_train, y_train, X_val, evaluation_ds, study_name, n_trials, storage)
    
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
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    TRAIN_DATASET_DIR = args.training_dataset_path
    VALIDATION_DATASET_DIR = args.validation_dataset_path
    N_TRIALS = args.n_trials
    STUDY_NAME = args.study_name
    STORAGE = args.storage
    
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