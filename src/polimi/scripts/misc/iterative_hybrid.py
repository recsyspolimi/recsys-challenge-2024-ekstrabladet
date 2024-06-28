import polars as pl
import pandas as pd
import os
import json
import numpy as np
from sklearn.utils import resample
from polimi.utils._inference import _inference
from ebrec.evaluation.metrics_protocols import *
from fastauc.fastauc.fast_auc import CppAuc
from tqdm import tqdm

predictions = [
    '/home/ubuntu/dataset/models_predictions/ranker_predictions.parquet',
    '/home/ubuntu/dataset/models_predictions/catboost_predictions.parquet',
    '/home/ubuntu/dataset/models_predictions/mlp_predictions.parquet',
    '/home/ubuntu/dataset/models_predictions/deep_cross_predictions.parquet',
    '/home/ubuntu/dataset/models_predictions/fast_rgf_predictions.parquet',
    '/home/ubuntu/dataset/models_predictions/gandalf_predictions.parquet',
    '/home/ubuntu/dataset/models_predictions/lgbm_rf_predictions.parquet',
    '/home/ubuntu/dataset/models_predictions/logistic_regression_predictions.parquet',
    '/home/ubuntu/dataset/models_predictions/wide_deep_predictions.parquet'
]

names = ['ranker', 'catboost', 'mlp', 'deep', 'fast', 'gandalf', 'lgbm', 'logistic', 'wide_deep']

prediction_1 = ''
prediction_2 = ''
dataset_path = '/home/ubuntu/experiments/preprocessing_train_2024-05-20_14-40-50'

def get_two_splits():
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
    
    per_batch_elements = int(impression_time_ds.shape[0] / 2)
   
    first_split = resample(
                impression_time_ds, replace=False, n_samples=per_batch_elements, stratify=impression_time_ds['impression_time'])
    second_split = impression_time_ds.filter(
                ~pl.col('impression_id').is_in(first_split['impression_id']))
      
    return first_split, second_split, data_info['categorical_columns']
        

if __name__ == "__main__":
    # raise "Work in progress"

    result_1 = pl.read_parquet(predictions[0])
    result_1 = result_1.with_columns(
        (pl.col('prediction')-pl.col('prediction').min().over('impression_id')) / 
        (pl.col('prediction').max().over('impression_id')-pl.col('prediction').min().over('impression_id'))
    )
    if 'user_id' in result_1.columns:
        result_1 = result_1.drop('user_id')
        
    result_2 = pl.read_parquet(predictions[1])
    result_2 = result_2.with_columns(
        (pl.col('prediction')-pl.col('prediction').min().over('impression_id')) / 
        (pl.col('prediction').max().over('impression_id')-pl.col('prediction').min().over('impression_id'))
    )
    if 'user_id' in result_2.columns:
        result_2 = result_2.drop('user_id')

    prediction = result_1.join(result_2, on=['impression_id','article','target'])

    best = 0
    best_alpha = 0
    best_prediction = None
    print('Merging  :')
    print(names[0])
    print(names[1])
    for iter in tqdm(range(0, 101)):
        print('alpha')
        alpha = iter/100
        prediction_mid = prediction.with_columns(
            (alpha * pl.col('prediction')+ (1-alpha) * pl.col('prediction_right')).alias('prediction')
        ).drop('prediction_right')
        prediction_ds = prediction.group_by('impression_id').agg(pl.col('target'), pl.col('prediction'))
        
        cpp_auc = CppAuc()
        result = np.mean(
            [cpp_auc.roc_auc_score(np.array(y_t).astype(bool), np.array(y_s).astype(np.float32)) 
                for y_t, y_s in zip(prediction_ds['target'].to_list(), 
                                    prediction_ds['prediction'].to_list())]
        )
        if result > best:
            best = result
            best_alpha = alpha
            best_prediction = prediction_mid
    print(f'Best alpha value {best_alpha}')
    print(f'AUC : {best}')    

    for pred in range(2,9):
        result_2 = pl.read_parquet(predictions[pred])
        result_2 = result_2.with_columns(
            (pl.col('prediction')-pl.col('prediction').min().over('impression_id')) / 
            (pl.col('prediction').max().over('impression_id')-pl.col('prediction').min().over('impression_id'))
        )
        
        if 'user_id' in result_2.columns:
            result_2 = result_2.drop('user_id')
        print(result_2)
        prediction = best_prediction.join(result_2, on=['impression_id','article','target'])

        best = 0
        best_alpha = 0
        best_prediction = None
        print('Merging  :')
        print(names[pred])
        for iter in tqdm(range(0, 101)):
            print('alpha')
            alpha = iter/100
            prediction_mid = prediction.with_columns(
                (alpha * pl.col('prediction')+ (1-alpha) * pl.col('prediction_right')).alias('prediction')
            ).drop('prediction_right')
            prediction_ds = prediction_mid.group_by('impression_id').agg(pl.col('target'), pl.col('prediction'))
            
            cpp_auc = CppAuc()
            result = np.mean(
                [cpp_auc.roc_auc_score(np.array(y_t).astype(bool), np.array(y_s).astype(np.float32)) 
                    for y_t, y_s in zip(prediction_ds['target'].to_list(), 
                                        prediction_ds['prediction'].to_list())]
            )
            if result > best:
                best = result
                best_alpha = alpha
                best_prediction = prediction_mid
        print(f'Best alpha value {best_alpha}')
        print(f'AUC : {best}')    