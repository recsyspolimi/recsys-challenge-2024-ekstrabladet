import polars as pl
import pandas as pd
from tqdm import tqdm
import os
import json
import numpy as np
from catboost import CatBoostClassifier, CatBoostRanker, Pool, sum_models
from sklearn.utils import resample
from polimi.utils._inference import _inference
import gc
from ebrec.evaluation.metrics_protocols import *
import catboost
from ebrec.utils._behaviors import sampling_strategy_wu2019
import matplotlib.pyplot as plt
from fastauc.fastauc.fast_auc import CppAuc
from polimi.utils._polars import reduce_polars_df_memory_size

# TARGET 0.8037703435093432

# TARGET DEMO 0.7916280544043559

dataset_path = '/home/ubuntu/experiments_1/preprocessing_train_small_new'
original_datset_path = '/home/ubuntu/dataset/ebnerd_small/train/behaviors.parquet'
validation_path = '/home/ubuntu/experiments_1/preprocessing_validation_small_new'

catboost_params = {
    "iterations": 1000,
    "subsample": 0.5,
    "rsm": 0.7
}

EVAL = True
SAVE_FEATURES_ORDER = False
SAVE_PREDICTIONS = True
N_BATCH = 10
NPRATIO = 2


def build_two_splits():
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
if __name__ == '__main__':
    
    train_ds = reduce_polars_df_memory_size(pl.read_parquet(os.path.join(dataset_path, 'train_ds.parquet')), verbose=False)
    
    
    starting_dataset =  pl.read_parquet(original_datset_path).select(['impression_id','user_id','article_ids_inview','article_ids_clicked'])
    
    behaviors = pl.concat(
        rows.pipe(
            sampling_strategy_wu2019, npratio=NPRATIO, shuffle=False, with_replacement=True, seed=123
        ).explode('article_ids_inview').drop(columns =['article_ids_clicked']).rename({'article_ids_inview' : 'article'})\
        .with_columns(pl.col('user_id').cast(pl.UInt32),
                      pl.col('article').cast(pl.Int32))\
        
         for rows in tqdm(starting_dataset.iter_slices(1000), total=starting_dataset.shape[0] // 1000)
    )
        
    train_ds = behaviors.join(train_ds, on = ['impression_id','user_id','article'], how = 'left')
    column_oder = train_ds.columns
    print(f'N features {len(column_oder)}')
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)

    print(f'Data info: {data_info}')

    print(f'Starting to train the catboost model')
    
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
    
    categorical_columns = data_info['categorical_columns']
    categorical_columns = [cat for cat in categorical_columns if cat in column_oder]  
         
    if 'postcode' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in train_ds.columns:
        train_ds = train_ds.drop(['impression_time'])
        
    train_ds = train_ds.drop(['impression_id', 'article', 'user_id']).to_pandas()
    train_ds[categorical_columns] = train_ds[categorical_columns].astype('category')

    X = train_ds.drop(columns = ['target'])
    print(X.columns[0])
    y = train_ds['target']
    
    model = CatBoostClassifier(**catboost_params, cat_features=categorical_columns)
    model.fit(X, y, verbose=25)
    

            
    if EVAL:
        with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
            data_info = json.load(data_info_file)

        categorical_columns = data_info['categorical_columns']
        categorical_columns = [cat for cat in categorical_columns if cat in column_oder] 
        
        
        val_ds = pl.read_parquet(validation_path + '/validation_ds.parquet').select(column_oder)
        # val_ds = build_topic_endorsement(val_ds, 'validation')
        
        if 'postcode' in val_ds.columns:
            val_ds = val_ds.with_columns(pl.col('postcode').fill_null(5))
        if 'article_type' in val_ds.columns:
            val_ds = val_ds.with_columns(
                pl.col('article_type').fill_null('article_default'))

        if 'impression_time' in val_ds.columns:
            val_ds = val_ds.drop('impression_time')

        val_ds_pandas = val_ds.drop(
            ['impression_id', 'article', 'user_id']).to_pandas()

        val_ds_pandas[categorical_columns
                      ] = val_ds_pandas[categorical_columns].astype('category')

        X_val = val_ds_pandas.drop(columns = ['target'])
        # X_val = X_val.drop(columns = drop_features)
        # X_val = X_val.drop(columns = to_drop)
        # X_val = X_val.drop(columns = click_feat)
        y_val = val_ds_pandas['target']
        
        evaluation_ds = val_ds[['impression_id', 'article', 'target']]        
        prediction_ds = evaluation_ds.with_columns(pl.Series(model.predict_proba(X_val)[:, 1]).alias('prediction'))
        if SAVE_PREDICTIONS:
            prediction_ds.write_parquet('/home/ubuntu/experiments/test_batch_training/classifier_predictions.parquet')
        prediction_ds = prediction_ds.group_by('impression_id').agg(pl.col('target'), pl.col('prediction'))
            

        cpp_auc = CppAuc()
        result = np.mean(
            [cpp_auc.roc_auc_score(np.array(y_t).astype(bool), np.array(y_s).astype(np.float32)) 
                for y_t, y_s in zip(prediction_ds['target'].to_list(), 
                                    prediction_ds['prediction'].to_list())]
        )
        print('AUC : ')
        print(result)