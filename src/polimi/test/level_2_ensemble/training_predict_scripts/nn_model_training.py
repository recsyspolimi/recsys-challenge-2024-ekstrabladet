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


def train_predict_nn_model(train_ds, val_ds, data_info, params):
 
    if 'postcode' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in train_ds.columns:
        train_ds = train_ds.drop(['impression_time'])
    
    train_ds = train_ds.drop(['impression_id', 'article', 'user_id']).to_pandas()
    train_ds[data_info['categorical_columns']] = train_ds[data_info['categorical_columns']].astype('category')

    X = train_ds.drop(columns=['target'])
    y = train_ds['target']
    
    categorical_columns = data_info['categorical_columns']
    numerical_columns = [c for c in X.columns if c not in categorical_columns]
    
    del train_ds
    gc.collect()
    
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
    )
    
    if 'postcode' in val_ds.columns:
        val_ds = val_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in val_ds.columns:
        val_ds = val_ds.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in val_ds.columns:
        val_ds = val_ds.drop(['impression_time'])
    
    val_ds = val_ds.to_pandas()
    val_ds[data_info['categorical_columns']] = val_ds[data_info['categorical_columns']].astype('category')

    X_val = val_ds[X.columns]
    evaluation_ds = pl.from_pandas(val_ds[['impression_id', 'article', 'target']])
        
    prediction_ds = evaluation_ds.with_columns(
            pl.Series(model.predict(X_val)).alias('prediction')
        )
    return prediction_ds