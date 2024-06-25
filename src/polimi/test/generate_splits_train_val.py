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

train_ds = '/home/ubuntu/experiments/preprocessing_train_new_with_recsys'
val_ds = '/home/ubuntu/experiments/preprocessing_val_new_with_recsys'

output_path = '/home/ubuntu/experiments/catboost_rnk_recsys_train_val/batches'
N_BATCH = 12


if __name__ == '__main__':

    train_ds = pl.scan_parquet(os.path.join(train_ds, 'train_ds.parquet')).select(['impression_id','impression_time'])
    val_ds = pl.scan_parquet(os.path.join(val_ds, 'validation_ds.parquet')).select(['impression_id','impression_time'])
    
    train_ds = train_ds.collect()
    val_ds = val_ds.collect()
    
    df = train_ds.vstack(val_ds)
    print(df.shape)

    impression_time_ds = df.with_columns(
        pl.col("impression_time").cast(pl.Date)).unique('impression_id')
  
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

   