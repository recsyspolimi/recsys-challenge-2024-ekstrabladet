import os
import logging
from datetime import datetime
import argparse
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing_extensions import List, Tuple, Dict, Type, TypeVar
import polars as pl
import gc
import tensorflow as tf

from polimi.utils.tf_models import *
from polimi.utils.tf_models.utils import *

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
T = TypeVar('T', bound=TabularNNModel)


def get_model_class(name: str = 'mlp') -> T:
    if name == 'MLP':
        return MLP
    elif name == 'DeepCrossNetwork':
        return DeepCrossNetwork
    elif name == 'WideDeepNetwork':
        return WideDeepNetwork
    elif name == 'DeepAbstractNetwork':
        return DeepAbstractNetwork
    elif name == 'GANDALF':
        return GANDALF


def main(dataset_path, params_path, output_dir, early_stopping_path, es_patience):
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")
    
    train_ds = pl.read_parquet(os.path.join(dataset_path, 'train_ds.parquet'))
    val_ds = pl.read_parquet(os.path.join(dataset_path, 'train_ds.parquet')) if early_stopping_path else None
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Data info: {data_info}')
        
    if 'postcode' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('postcode').fill_null(5))
        if early_stopping_path:
            val_ds = val_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('article_type').fill_null('article_default'))
        if early_stopping_path:
            val_ds = val_ds.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in train_ds.columns:
        train_ds = train_ds.drop(['impression_time'])
    
    train_ds = train_ds.drop(['impression_id', 'article', 'user_id']).to_pandas()
    train_ds[data_info['categorical_columns']] = train_ds[data_info['categorical_columns']].astype('category')

    X = train_ds.drop(columns=['target'])
    y = train_ds['target']
    
    del train_ds
    gc.collect()
    
    if early_stopping_path:
        val_ds = val_ds.drop(['impression_id', 'article', 'user_id']).to_pandas()
        val_ds[data_info['categorical_columns']] = val_ds[data_info['categorical_columns']].astype('category')
        validation_data = (val_ds[X.columns], val_ds['target'])
        del val_ds
        gc.collect()
    else:
        validation_data = None
    
    categorical_columns = data_info['categorical_columns']
    numerical_columns = [c for c in X.columns if c not in categorical_columns]
    
    del train_ds
    gc.collect()
    
    logging.info(f'Features ({len(X.columns)}): {np.array(list(X.columns))}')
    logging.info(f'Categorical features: {np.array(data_info["categorical_columns"])}')
    logging.info(f'Reading neural network parameters from path: {params_path}')
    
    with open(params_path, 'r') as params_file:
        params = json.load(params_file)
        
    logging.info(f'Params: {params}')
    logging.info(f'Starting to train the {params["model_name"]} model')
    
    model: TabularNNModel = get_model_class(params['model_name'])(categorical_features=categorical_columns, 
                                                                  numerical_features=numerical_columns, 
                                                                  **params['model_hyperparams'])
    lr_scheduler = get_simple_decay_scheduler(params['scheduling_rate']) if params['use_scheduler'] else None
    model.fit(
        X,
        y,
        epochs=params['epochs'],
        batch_size=512,
        lr_scheduler=lr_scheduler,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(curve='ROC', name='auc')],
        optimizer=tf.keras.optimizers.AdamW(params['learning_rate'], weight_decay=params['weight_decay'], clipnorm=5.0),
        validation_data=validation_data,
        early_stopping_monitor='val_auc',
        early_stopping_mode='max',
        early_stopping_rounds=es_patience,
    )
        
    logging.info(f'Model fitted. Saving the model and the feature importances at: {output_dir}')
    model.save(output_dir)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for neural networks")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the preprocessed dataset is placed")
    parser.add_argument("-early_stopping_path", default=None, type=str,
                        help="Directory where the early stopping dataset is placed")
    parser.add_argument("-params_file", default=None, type=str, required=True,
                        help="File path where the catboost hyperparameters are placed")
    parser.add_argument("-model_name", default=None, type=str,
                        help="The name of the model")
    parser.add_argument("-early_stopping_patience", default=2, type=int,
                        help="The patience for early stopping")
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    HYPERPARAMS_PATH = args.params_file
    EARLY_STOPPING_PATH = args.early_stopping_path
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    model_name = args.model_name
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if model_name is None:
        model_name = f'NN_Training_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_DIR, HYPERPARAMS_PATH, output_dir, EARLY_STOPPING_PATH, EARLY_STOPPING_PATIENCE)
