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
import joblib
import torch
import tensorflow as tf
from torch.optim import AdamW
from deepctr_torch.callbacks import EarlyStopping

from polimi.utils.tf_models import *
from polimi.utils.tf_models.utils import *
import sys
sys.path.append('/home/ubuntu/RecSysChallenge2024/src')

from ebrec.evaluation.metrics_protocols import *
from polimi.utils.tf_models import *
from polimi.utils.tf_models.utils import *
from fastauc.fastauc.fast_auc import CppAuc

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
T = TypeVar('T', bound=TabularNNModel)

from sklearn.preprocessing import PowerTransformer, OrdinalEncoder
from deepctr_torch.models import NFM
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names

def create_layer_tuple(num_layers,start):
    start_value = start
    layer_values = [start_value]
    for _ in range(num_layers - 1):
        start_value //= 2  # Update start_value by dividing by 2
        layer_values.append(start_value)
    return tuple(layer_values)

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
T = TypeVar('T', bound=TabularNNModel)


def main(dataset_path, output_dir, early_stopping_path, transform_path):
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")
    
    train_ds = pl.read_parquet(os.path.join(dataset_path, 'train_ds.parquet'))
    val_ds = pl.read_parquet(os.path.join(early_stopping_path, 'validation_ds.parquet')) if early_stopping_path else None
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
        if transform_path:
            xformer = joblib.load(transform_path)
            val_ds[xformer.feature_names_in_] = xformer.transform(val_ds[xformer.feature_names_in_].replace(
                [-np.inf, np.inf], np.nan).fillna(0)).astype(np.float32)
            for i, cat_col in enumerate(categorical_columns):
                categories_val = list(val_ds[cat_col].unique())
                unknown_categories = [x for x in categories_val if x not in categories[i]]
                val_ds[cat_col] = val_ds[cat_col].replace(list(unknown_categories), 'Unknown')
            fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=val_ds[feat].nunique(),embedding_dim=64)
                       for i,feat in enumerate(val_ds[categorical_columns].columns)] + [DenseFeat(feat, 1,)
                      for feat in numerical_columns]
            val_feature_names = get_feature_names(fixlen_feature_columns)
            test = {name:val_ds[name] for name in val_feature_names}
        validation_data = (test, val_ds['target'].values)
        del val_ds
        gc.collect()
    else:
        validation_data = None
    
    categorical_columns = data_info['categorical_columns']
    numerical_columns = [c for c in X.columns if c not in categorical_columns]
    
    logging.info(f'Features ({len(X.columns)}): {np.array(list(X.columns))}')
    logging.info(f'Categorical features: {np.array(data_info["categorical_columns"])}')

    logging.info(f'Starting to train the DeepFM model')
    params = {
        "dnn_dropout": 0.3444209355032415,
        "l2_reg_embedding": 0.004617505794106905,
        "l2_reg_linear": 0.00004561003389761619,
        "l2_reg_dnn": 0.005616631061636492,
        "trials": 14,
        "num_layers": 2,
        "start": 256,
        "lr": 0.009448207882844778
  }
    dnn_hidden_units = create_layer_tuple(params['num_layers'],params['start'])
    categories = []
    vocabulary_sizes = {}
    for cat_col in categorical_columns:
        categories_train = list(X[cat_col].unique())
        categories_train.append('Unknown')
        vocabulary_sizes[cat_col] = len(categories_train)
        categories.append(categories_train)

    train_fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=vocabulary_sizes[feat],embedding_dim=64)
                for i,feat in enumerate(categorical_columns)] + [DenseFeat(feat, 1,)
                for feat in numerical_columns]
    train_feature_names = get_feature_names(train_fixlen_feature_columns)
    model = NFM(train_fixlen_feature_columns,train_fixlen_feature_columns,dnn_dropout=params['dnn_dropout'],l2_reg_embedding=params['l2_reg_embedding'],l2_reg_linear=params['l2_reg_linear'],l2_reg_dnn=params['l2_reg_dnn'],dnn_hidden_units=dnn_hidden_units,dnn_activation='relu',task='binary')
    train_model_input = {name:X[name] for name in train_feature_names}
    model.compile(AdamW(model.parameters(),params['lr']),"binary_crossentropy",metrics=['auc'], )
    es = EarlyStopping(monitor='val_auc', min_delta=0, verbose=2, patience=5, mode='max')
    model.fit(train_model_input,y.values,batch_size=1024,epochs=10,validation_split=validation_data,callbacks=[es])
    
        
    logging.info(f'Model fitted. Saving the model and the feature importances at: {output_dir}')
    model.save(output_dir)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for DeepFM")
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
    parser.add_argument("-early_stopping_patience", default=5, type=int,
                        help="The patience for early stopping")
    parser.add_argument("-numerical_transformer_es", default=None, type=str,
                        help="The path for numerical transformer to transform the early stopping data if needed")
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    HYPERPARAMS_PATH = args.params_file
    EARLY_STOPPING_PATH = args.early_stopping_path
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    TRANSFORM_PATH = args.numerical_transformer_es
    model_name = args.model_name
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if model_name is None:
        model_name = f'DeepFM_Training_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_DIR, output_dir, EARLY_STOPPING_PATH, TRANSFORM_PATH)

