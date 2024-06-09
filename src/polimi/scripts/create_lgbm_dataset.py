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


def main(train_dataset_path, val_dataset_path, output_dir, is_ranking_dataset, num_bins):
    logging.info(f"Loading the preprocessed dataset from {train_dataset_path}")
    
    train_ds = pl.scan_parquet(os.path.join(train_dataset_path, 'train_ds.parquet'))
    val_ds = pl.scan_parquet(os.path.join(val_dataset_path, 'validation_ds.parquet')) if val_dataset_path else None
    with open(os.path.join(train_dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Data info: {data_info}')
    
    if is_ranking_dataset:
        train_ds = train_ds.sort(by='impression_id')
        if val_dataset_path:
            val_ds = val_ds.sort(by='impression_id')
            groups = pl.concat([
                train_ds.select(['impression_id']).collect(),
                val_ds.select(['impression_id']).collect()
            ]).to_pandas().groupby('impression_id').size()
        else:
            groups = train_ds.select(['impression_id']).collect().to_pandas().groupby('impression_id').size()
    else:
        groups = None
        
    if 'postcode' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('postcode').fill_null(5))
        if val_dataset_path:
            val_ds = val_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('article_type').fill_null('article_default'))
        if val_dataset_path:
            val_ds = val_ds.with_columns(pl.col('article_type').fill_null('article_default'))
        
    features = [col for col in train_ds.columns if col not in ['impression_id', 'article', 'user_id', 'impression_time', 'target']]
    categorical_features = data_info['categorical_columns']
    
    logging.info(f'Features ({len(features)}): {np.array(features)}')
    logging.info(f'Categorical features: {np.array(categorical_features)}')
    logging.info(f'Creating lightgbm dataset with {num_bins} bins')
    
    if val_dataset_path is None:
        lgbm_dataset = Dataset(
            train_ds.select(features).collect().to_pandas().astype({c: 'category' for c in categorical_features}),
            label=train_ds.select(['target']).collect().to_numpy().flatten(),
            feature_name=features,
            categorical_feature=categorical_features,
            params={'max_bin': num_bins},
            group=groups
        ).construct()
    else:
        lgbm_dataset = Dataset(
            pl.concat([train_ds.select(features), val_ds.select(features)], how='diagonal_relaxed') \
                .collect().to_pandas().astype({c: 'category' for c in categorical_features}),
            label=pl.concat([train_ds.select(['target']), val_ds.select(['target'])]).collect().to_numpy().flatten(),
            feature_name=features,
            categorical_feature=categorical_features,
            params={'max_bin': num_bins},
            group=groups
        ).construct()
    
    logging.info(f'Saving converted dataset at: {os.path.join(output_dir, "lgbm_dataset.bin")}')
    lgbm_dataset.save_binary(os.path.join(output_dir, 'lgbm_dataset.bin'))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for catboost")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-train_dataset_path", default=None, type=str, required=True,
                        help="Directory where the preprocessed dataset is placed")
    parser.add_argument("-val_dataset_path", default=None, type=str, required=False,
                        help="Directory where the preprocessed dataset is placed")
    parser.add_argument("-num_bins", default=64, type=int, required=True,
                        help="Directory where the preprocessed dataset is placed")
    parser.add_argument('--ranker_dataset', action='store_true', default=False, 
                        help='Whether to create the dataset for LGBMRanker or not')
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    TRAIN_DATASET_PATH = args.train_dataset_path
    VAL_DATASET_PATH = args.val_dataset_path
    RANKER_DATASET = args.ranker_dataset
    NUM_BINS = args.num_bins

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    log_path = os.path.join(OUTPUT_DIR, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(TRAIN_DATASET_PATH, VAL_DATASET_PATH, OUTPUT_DIR, RANKER_DATASET, NUM_BINS)
