import polars as pl
import pandas as pd
import os
import json
import numpy as np
from sklearn.utils import resample
from polimi.utils._inference import _inference
import gc
from ebrec.evaluation.metrics_protocols import *
from lightgbm import LGBMRanker
import joblib

dataset_path = '/home/ubuntu/experiments/preprocessing_train_small_new'
validation_path = '/home/ubuntu/experiments/preprocessing_validation_small_new'
batch_split_directory = '/home/ubuntu/experiments/test_batch_training/batches/'

# dataset_path = '/home/ubuntu/experiments/preprocessing_train_2024-05-18_09-34-07'
# validation_path = '/home/ubuntu/experiments/preprocessing_validation_2024-05-18_09-43-19'

model_path = '/home/ubuntu/experiments/test_batch_training'
params = {
    'n_estimators': 4161,
    'max_depth': 11, 
    'num_leaves': 610,
    'subsample_freq': 1, 
    'subsample': 0.6552618946933639, 
    'learning_rate': 0.008575570828554459,
    'colsample_bytree': 0.7355889468729234,
    'colsample_bynode': 0.31281682604526095, 
    'reg_lambda': 11.407621501254667,
    'reg_alpha': 0.6125928057452898,
    'max_bin': 22, 
    'min_split_gain': 0.007803071843359968,
    'min_child_weight': 6.305938301704642e-07,
    'min_child_samples': 2533, 
    'extra_trees': False
}
EVAL = True
SAVE_PREDICTIONS = True
N_BATCH = 10

def load_batch(dataset_path, batch_split_directory, batch_index):
    
    train_ds = pl.scan_parquet(dataset_path + '/train_ds.parquet')
    batch = pl.scan_parquet(batch_split_directory + f'/batch_{batch_index}.parquet').collect()
    
    subsampled_train = train_ds.filter(pl.col('impression_id').is_in(
            batch.select('impression_id'))).collect()
    
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    if 'postcode' in subsampled_train.columns:
        subsampled_train = subsampled_train.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in subsampled_train.columns:
        subsampled_train = subsampled_train.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in subsampled_train.columns:
        subsampled_train = subsampled_train.drop(['impression_time'])
            
    subsampled_train = subsampled_train.to_pandas()
    group_ids = subsampled_train['impression_id'].to_frame()
    subsampled_train = subsampled_train.drop(columns=['impression_id', 'article', 'user_id'])
    subsampled_train[data_info['categorical_columns']] = subsampled_train[data_info['categorical_columns']].astype('category')
    
    X = subsampled_train.drop(columns=['target'])
    y = subsampled_train['target']
    print(X.shape)

    if 'impression_time' in X:
        X = X.drop(['impression_time'])
    
    del train_ds,batch,subsampled_train
    gc.collect()
        
    return X, y, group_ids

# started at 12.58    

def batch_training(model_path, catboost_params, data_info, dataset_path, batch_split_directory):

    for batch in range(N_BATCH):
        print(f'-------------BATCH {batch}-----------')
        output_dir = model_path + f'/model_{batch}.cbm'
        model = LGBMRanker(**params,verbosity=-1)
        
        print(f'Collecting batch...')
        X, y, groups = load_batch(dataset_path, batch_split_directory, batch)
        
        print('Fitting Model...')
        model.fit(X, y, group=groups.groupby('impression_id')['impression_id'].count().values)
        joblib.dump(model, output_dir)
        del model, X, y, groups
        gc.collect()

    return model

if __name__ == '__main__':

    train_ds = pl.read_parquet(os.path.join(dataset_path, 'train_ds.parquet'))
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)

    print(f'Data info: {data_info}')

    print(f'Starting to train the catboost model')

    model = batch_training(model_path, params, data_info, dataset_path, batch_split_directory)
    # model = incremental_training(model_path, train_ds, impression_time_ds, catboost_params, data_info)
            
    if EVAL:
        models = []
        for batch in range(N_BATCH):
            model = joblib.load(model_path + f'/model_{batch}.cbm')
            models.append(model)
        weights = [1/N_BATCH] * N_BATCH

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

        for model_index in range(len(models)):
            pred = val_ds.with_columns(
                pl.Series(models[model_index].predict(X_val)).alias(f'prediction_{model_index}'))
        
        pred = pred.with_columns(
                    *[(weights[i] * pl.col(f'prediction_{i}')).alias(f'prediction_{i}') for i in range(len(models))]
                ).with_columns(
                    pl.sum_horizontal([f"prediction_{i}" for i in range(len(models))]).alias('prediction')
                ).drop([f"prediction_{i}" for i in range(len(models))]).rename({'final_pred' : 'prediction'})
        
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
