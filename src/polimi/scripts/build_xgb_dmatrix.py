import os
import logging
from pathlib import Path
from xgboost import XGBClassifier, XGBRanker
import xgboost as xgb
from datetime import datetime
import argparse
import pandas as pd
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing_extensions import List
import polars as pl
import polars.selectors as cs
import gc
from polimi.utils._polars import reduce_polars_df_memory_size, check_for_inf
from polimi.utils._custom import save_json
import time

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"

def build_X_y(data_dir: Path, output_dir: Path):
    
    with open(data_dir / 'data_info.json') as data_info_file:
        data_info = json.load(data_info_file)
        
    print(f'Data info: {data_info}')
    
    file_path = data_dir / 'train_ds.parquet'
    
    print(f"Reading the dataset from {file_path}")
    train_ds = pl.read_parquet(file_path)
    
    train_ds = train_ds.sort(by='impression_id')
        
    if 'postcode' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in train_ds.columns:
        train_ds = train_ds.drop(['impression_time'])
        
    print('Checking for inf in train_ds...')
    rows_with_inf, cols_with_inf = check_for_inf(train_ds)
    print(f'Rows with inf: {rows_with_inf}')
    print(f'Columns with inf: {cols_with_inf}')
    
    print('Replacing inf from train_ds...')
    train_ds = train_ds.with_columns(pl.when(~(cs.numeric().is_infinite())).then(cs.numeric()))
    print('Finish')
    
        
    train_ds = reduce_polars_df_memory_size(train_ds)
    train_ds = train_ds.drop(['impression_id', 'article', 'user_id']).to_pandas()
    train_ds[data_info['categorical_columns']] = train_ds[data_info['categorical_columns']].astype('category')
    save_json(dict(zip(train_ds.columns, [str(dtype) for dtype in train_ds.dtypes])), output_dir / 'train_ds_dtypes.json')
    
    
    print('Building X dataset...')
    X = train_ds.drop(columns=['target'])
    print('Saving X to parquet...')
    X.to_parquet(output_dir / 'X.parquet')
    print('Building y dataset...')
    y = train_ds['target'].to_frame()
    print('Saving y to parquet...')
    y.to_parquet(output_dir / 'y.parquet')
    
    
def buildDMatrix(data_dir: Path, output_dir: Path, params: dict):
    print('Loading the preprocessed X dataset...')
    X = pd.read_parquet(data_dir / 'X.parquet')
    print('Loading the preprocessed y dataset...')
    y = pd.read_parquet(data_dir / 'y.parquet')
    
    print('Start building XGB DMatrix...')
    start_time = time.time()
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
    print(f'Finished in {((time.time() - start_time)/60):.2f}min.. Starting saving the DMatrix binary...')
    dtrain.save_binary(output_dir / 'dmatrix.buffer')
    
if __name__ == '__main__':

    DATASET_DIR = Path('/Users/lorecampa/Desktop/Projects/RecSysChallenge2024/dataset/preprocessing/subsample_new_with_recsys_small')
    OUTPUT_DIR = Path('/Users/lorecampa/Desktop/Projects/RecSysChallenge2024/experiments')
    PARAMS_PATH = Path('/Users/lorecampa/Desktop/Projects/RecSysChallenge2024/configuration_files/xgb_cls_new_with_recsys_noK.json')
        
    with open(PARAMS_PATH, 'r') as params_file:
        params = json.load(params_file)
        
    OUTPUT_DIR = OUTPUT_DIR / f'xgb_dmatrix'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    build_X_y(DATASET_DIR, OUTPUT_DIR)

    DATASET_DIR = Path('/Users/lorecampa/Desktop/Projects/RecSysChallenge2024/experiments/xgb_dmatrix')

    # buildDMatrix(DATASET_DIR, OUTPUT_DIR, params)
