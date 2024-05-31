import os
import logging
from lightgbm import Dataset
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
import gc

from polimi.preprocessing_pipelines.pre_68f import strip_new_features

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def main(dataset_path, output_dir):
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")
    
    train_ds = pl.read_parquet(os.path.join(dataset_path, 'train_ds.parquet'))
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Data info: {data_info}')
    
    train_ds = train_ds.sort(by='impression_id')
    groups = train_ds.select(['impression_id']).to_pandas().groupby('impression_id').size()
        
    if 'postcode' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in train_ds.columns:
        train_ds = train_ds.drop(['impression_time'])
        
    features = [col for col in train_ds.columns if col not in ['impression_id', 'article', 'user_id', 'impression_time']]
    categorical_features = data_info['categorical_columns']
    
    logging.info(f'Features ({len(features)}): {np.array(features)}')
    logging.info(f'Categorical features: {np.array(categorical_features)}')
    
    features_per_batch = 30
    start_batch = 0
    lgbm_dataset = None
    while start_batch < len(features):
        logging.info(f'Adding features ({start_batch}, {end_batch}) to the dataset')
        end_batch = min(len(features), start_batch + features_per_batch)
        
        batch_features = features[start_batch:end_batch]
        categorical_batch_features = [f for f in batch_features if f in categorical_features]
        
        batch_dataset = Dataset(
            train_ds.select(batch_features + ['target']).to_pandas(),
            label='target',
            feature_name=batch_features,
            categorical_feature=categorical_batch_features,
            group=groups,
        )
        
        if lgbm_dataset is None:
            lgbm_dataset = batch_dataset
        else:
            lgbm_dataset = lgbm_dataset.add_features_from(batch_dataset)
            
        start_batch = end_batch
        
    logging.info(f'Saving converted dataset at: {os.path.join(output_dir, "lgbm_dataset.bin")}')
    lgbm_dataset.save_binary(os.path.join(output_dir, 'lgbm_dataset.bin'))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for catboost")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the preprocessed dataset is placed")
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_PATH = args.dataset_path
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(OUTPUT_DIR, f'LGBM_Ranker_Dataset_{timestamp}')
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_PATH, output_dir)
