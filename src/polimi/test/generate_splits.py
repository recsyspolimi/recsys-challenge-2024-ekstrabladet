import polars as pl
import pandas as pd
import os
import json
import numpy as np
from catboost import CatBoostClassifier, CatBoostRanker, Pool, sum_models
from sklearn.utils import resample
from polimi.utils._inference import _inference
import gc
from ebrec.evaluation.metrics_protocols import *
import catboost

dataset_path = '/home/ubuntu/experiments/preprocessing_train_small_new'
output_path = '/home/ubuntu/experiments/test_batch_training/batches/'
N_BATCH = 10


if __name__ == '__main__':

    train_ds = pl.read_parquet(os.path.join(dataset_path, 'train_ds.parquet'))
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)

    print(f'Data info: {data_info}')

    if 'postcode' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in train_ds.columns:
        train_ds = train_ds.with_columns(
            pl.col('article_type').fill_null('article_default'))

    impression_time_ds = train_ds.select(['impression_id', 'impression_time']).with_columns(
        pl.col("impression_time").cast(pl.Date)).unique('impression_id')
  
    train_ds[data_info['categorical_columns']
             ] = train_ds[data_info['categorical_columns']].to_pandas().astype('category')
    
    per_batch_elements = int(impression_time_ds.shape[0] / N_BATCH)

    for batch in range(N_BATCH):
        print(f'-------------BATCH {batch}-----------')
        
        if (batch - N_BATCH + 1) < 0:
            sampled_impressions = resample(
                impression_time_ds, replace=False, n_samples=per_batch_elements, stratify=impression_time_ds['impression_time'])
            impression_time_ds = impression_time_ds.filter(
                ~pl.col('impression_id').is_in(sampled_impressions['impression_id']))
        else:
            sampled_impressions = impression_time_ds
        
        sampled_impressions.write_parquet(output_path + f'/batch_{batch}.parquet')

   