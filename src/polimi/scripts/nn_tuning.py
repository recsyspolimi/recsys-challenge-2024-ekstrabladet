import os
import logging
from datetime import datetime
import argparse
import pandas as pd
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing_extensions import List, Tuple, Dict, Type, TypeVar
import optuna
import polars as pl
import tensorflow as tf

import sys
sys.path.append('/home/ubuntu/RecSysChallenge2024/src')

from ebrec.evaluation.metrics_protocols import *
from polimi.utils.tf_models import *
from polimi.utils.tf_models.utils import *
from fastauc.fastauc.fast_auc import CppAuc

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
T = TypeVar('T', bound=TabularNNModel)


def get_model_class(name: str = 'mlp'):
    if name == 'mlp':
        return MLP
    elif name == 'dcn':
        return DeepCrossNetwork
    elif name == 'wd':
        return WideDeepNetwork
    elif name == 'danet':
        return DeepAbstractNetwork
    elif name == 'gandalf':
        return GANDALF


def optimize_parameters(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, evaluation_ds: pl.DataFrame, 
                        categorical_features: List[str], numerical_features: List[str],
                        study_name: str = 'mlp_tuning', n_trials: int = 100, storage: str = None, 
                        model_class: Type[T] = TabularNNModel) -> Tuple[Dict, pd.DataFrame]:
    
    def objective_function(trial: optuna.Trial):
        # TODO: early stopping instead of tunable epochs?
        params = model_class.get_optuna_trial(trial)
        model = model_class(categorical_features=categorical_features, numerical_features=numerical_features, **params)
        
        # training hyperparameters
        epochs = trial.suggest_int('epochs', 1, 50)
        if model_class == GANDALF:
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
            clipnorm = 1.0
        else:
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2)
            clipnorm = 5.0
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2)
        use_scheduler = trial.suggest_categorical('use_scheduler', [True, False])
        lr_scheduler = get_simple_decay_scheduler(trial.suggest_float('scheduling_rate', 1e-3, 0.1)) if use_scheduler else None
        
        model.fit(
            X_train, 
            y_train,
            epochs=epochs,
            batch_size=128,
            lr_scheduler=lr_scheduler,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(curve='ROC', name='auc')],
            optimizer=tf.keras.optimizers.AdamW(learning_rate, weight_decay=weight_decay, clipnorm=clipnorm),
        )
        
        prediction_ds = evaluation_ds.with_columns(
            pl.Series(model.predict(X_val)).alias('prediction')
        ).group_by('impression_id').agg(pl.col('target'), pl.col('prediction'))
        
        cpp_auc = CppAuc()
        return np.mean(
            [cpp_auc.roc_auc_score(np.array(y_t).astype(bool), np.array(y_s).astype(np.float32)) 
                for y_t, y_s in zip(prediction_ds['target'].to_list(), 
                                    prediction_ds['prediction'].to_list())]
        )
        
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)
    study.optimize(objective_function, n_trials=n_trials, n_jobs=1)
    return study.best_params, study.trials_dataframe()

def load_datasets(train_dataset_path, validation_dataset_path):
    logging.info(f"Loading the training dataset from {train_dataset_path}")
    
    train_ds = pl.read_parquet(os.path.join(train_dataset_path, 'train_ds.parquet'))
    with open(os.path.join(train_dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Data info: {data_info}')
    
    if 'postcode' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in train_ds.columns:
        train_ds = train_ds.drop(['impression_time'])
    
    train_ds = train_ds.to_pandas()
    train_ds = train_ds.drop(columns=['impression_id', 'article', 'user_id'])
    train_ds[data_info['categorical_columns']] = train_ds[data_info['categorical_columns']].astype('category')
    
    X_train = train_ds.drop(columns=['target'])
    y_train = train_ds['target']
    
    categorical_columns = data_info['categorical_columns']
    numerical_columns = [c for c in X_train.columns if c not in categorical_columns]
    
    logging.info(f'Features ({len(X_train.columns)}): {np.array(list(X_train.columns))}')
    logging.info(f'Categorical features: {np.array(data_info["categorical_columns"])}')
    
    logging.info(f"Loading the validation dataset from {validation_dataset_path}")
    
    val_ds = pl.read_parquet(os.path.join(validation_dataset_path, 'validation_ds.parquet'))
    
    if 'postcode' in val_ds.columns:
        val_ds = val_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in val_ds.columns:
        val_ds = val_ds.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in val_ds.columns:
        val_ds = val_ds.drop(['impression_time'])
    
    val_ds = val_ds.to_pandas()
    val_ds[data_info['categorical_columns']] = val_ds[data_info['categorical_columns']].astype('category')

    X_val = val_ds[X_train.columns]
    evaluation_ds = pl.from_pandas(val_ds[['impression_id', 'article', 'target']])
    return X_train, y_train, X_val, evaluation_ds, categorical_columns, numerical_columns


def main(train_dataset_path: str, validation_dataset_path: str, output_dir: str, model_class: Type[T] = TabularNNModel,
         study_name: str = 'nn_tuning', n_trials: int = 100, storage: str = None):
    X_train, y_train, X_val, evaluation_ds, cat_feat, num_feat = load_datasets(train_dataset_path, validation_dataset_path)
    
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr (prevents double logs)

    best_params, trials_df = optimize_parameters(X_train=X_train, y_train=y_train, X_val=X_val, evaluation_ds=evaluation_ds, 
                                                 categorical_features=cat_feat, numerical_features=num_feat, study_name=study_name, 
                                                 n_trials=n_trials, storage=storage, model_class=model_class)
    
    params_file_path = os.path.join(output_dir, 'best_params.json')
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
    parser.add_argument("-nn_type", choices=['mlp', 'dcn', 'wd', 'danet', 'gandalf'], type=str, default='mlp',
                        help='The type of neural network model to tune')
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
    # TODO: add cross validation in optuna
    parser.add_argument('--cv', action='store_true', default=False, help='Whether to use cross valiation or not')
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    TRAIN_DATASET_DIR = args.training_dataset_path
    VALIDATION_DATASET_DIR = args.validation_dataset_path
    N_TRIALS = args.n_trials
    STUDY_NAME = args.study_name
    STORAGE = args.storage
    NN_CLASS = get_model_class(args.nn_type)
    USE_CV = args.cv
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'{args.nn_type}_tuning_{timestamp}' if not USE_CV else f'{args.nn_type}_cv_tuning_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(TRAIN_DATASET_DIR, VALIDATION_DATASET_DIR, output_dir, model_class=NN_CLASS, 
         study_name=STUDY_NAME, n_trials=N_TRIALS, storage=STORAGE)