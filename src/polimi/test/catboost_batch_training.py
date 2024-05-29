import polars as pl
import pandas as pd
import os
import json
import numpy as np
from catboost import CatBoostClassifier, CatBoostRanker, Pool, sum_models
import gc
from ebrec.evaluation.metrics_protocols import *

RANKER = False
dataset_path = '/home/ubuntu/experiments/preprocessing_train_small_new'
validation_path = '/home/ubuntu/experiments/preprocessing_validation_small_new'
batch_split_directory = '/home/ubuntu/experiments/test_batch_training/batches/'

# dataset_path = '/home/ubuntu/experiments/preprocessing_train_2024-05-18_09-34-07'
# validation_path = '/home/ubuntu/experiments/preprocessing_validation_2024-05-18_09-43-19'

model_path = '/home/ubuntu/experiments/test_batch_training'
# catboost_params = {
#     'iterations': 2000,
#     'depth': 8,
#     'colsample_bylevel': 0.5
# }

catboost_params = {
    "iterations": 1000,
    "subsample": 0.5,
    "rsm": 0.7
}
EVAL = True
SAVE_PREDICTIONS = True
N_BATCH = 10

def load_batch(dataset_path, batch_split_directory, batch_index):
    
    train_ds = pl.scan_parquet(dataset_path + '/train_ds.parquet')
    batch = pl.scan_parquet(batch_split_directory + f'/batch_{batch_index}.parquet').collect()
    
    subsampled_train = train_ds.filter(pl.col('impression_id').is_in(
            batch.select('impression_id'))).collect()
    
    if 'postcode' in subsampled_train.columns:
            subsampled_train = subsampled_train.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in subsampled_train.columns:
            subsampled_train = subsampled_train.with_columns(pl.col('article_type').fill_null('article_default'))
            
    subsampled_train = subsampled_train.sort(by='impression_id')
    groups = subsampled_train.select('impression_id').to_numpy().flatten()
    subsampled_train = subsampled_train.drop(
            ['impression_id', 'article', 'user_id', 'impression_time']).to_pandas()

    X = subsampled_train.drop(columns=['target'])
    y = subsampled_train['target']
    print(X.shape)

    if 'impression_time' in X:
        X = X.drop(['impression_time'])
    
    del train_ds,batch,subsampled_train
    gc.collect()
        
    return X, y, groups

    

def batch_training(model_path, catboost_params, data_info, dataset_path, batch_split_directory, RANKER):

    for batch in range(N_BATCH):
        print(f'-------------BATCH {batch}-----------')
        output_dir = model_path + f'/model_{batch}.cbm'
        if RANKER:
            model = CatBoostRanker(
                **catboost_params, cat_features=data_info['categorical_columns'])
        else:
            model = CatBoostClassifier(**catboost_params, cat_features=data_info['categorical_columns'])
        
        print(f'Collecting batch...')
        X, y, groups = load_batch(dataset_path, batch_split_directory, batch)
        
        print('Fitting Model...')
        
        if RANKER:
            model.fit(X, y, group_id=groups, verbose=20)
        else : 
            model.fit(X, y, verbose=20)

        model.save_model(output_dir)
        del model, X, y, groups
        gc.collect()

    models = []
    for batch in range(N_BATCH):
        if RANKER:
            model = CatBoostRanker(
                **catboost_params, cat_features=data_info['categorical_columns'])
        else:
            model = CatBoostClassifier(**catboost_params, cat_features=data_info['categorical_columns'])
            
        model.load_model(model_path + f'/model_{batch}.cbm', format='cbm')
        models.append(model)
    weights = [1/N_BATCH] * N_BATCH

    model = sum_models(models, weights=weights,
                       ctr_merge_policy='IntersectingCountersAverage')

    return model

if __name__ == '__main__':

    train_ds = pl.read_parquet(os.path.join(dataset_path, 'train_ds.parquet'))
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)

    print(f'Data info: {data_info}')

    print(f'Starting to train the catboost model')

    model = batch_training(model_path, catboost_params, data_info, dataset_path, batch_split_directory, RANKER)
    # model = incremental_training(model_path, train_ds, impression_time_ds, catboost_params, data_info)
            
    if EVAL:
        with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
            data_info = json.load(data_info_file)

        val_ds = pl.read_parquet(validation_path + '/validation_ds.parquet')

        if 'postcode' in train_ds.columns:
            val_ds = val_ds.with_columns(pl.col('postcode').fill_null(5))
        if 'article_type' in train_ds.columns:
            val_ds = val_ds.with_columns(
                pl.col('article_type').fill_null('article_default'))

        if 'impression_time' in val_ds.columns:
            val_ds = val_ds.drop('impression_time')

        val_ds_pandas = val_ds.drop(
            ['impression_id', 'article', 'user_id']).to_pandas()

        val_ds_pandas[data_info['categorical_columns']
                      ] = val_ds_pandas[data_info['categorical_columns']].astype('category')

        X_val = val_ds_pandas.drop(columns=['target'])
        y_val = val_ds_pandas['target']

        if RANKER:
            pred = val_ds.with_columns(
                pl.Series(model.predict(X_val)).alias('prediction'))
        else:
            pred = val_ds.with_columns(
                pl.Series(model.predict_proba(X_val)[:, 1]).alias('prediction'))
        if SAVE_PREDICTIONS:
            pred.write_parquet('/home/ubuntu/experiments/test_batch_training/ranker_predictions.parquet')
        gc.collect()
        evaluation_ds = pred.group_by('impression_id').agg(
            pl.col('target'), pl.col('prediction'))
        met_eval = MetricEvaluator(
            labels=evaluation_ds['target'].to_list(),
            predictions=evaluation_ds['prediction'].to_list(),
            metric_functions=[
                AucScore(),
                MrrScore(),
                NdcgScore(k=5),
                NdcgScore(k=10),
            ],
        )
        print(met_eval.evaluate())
