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
validation_path = '/home/ubuntu/experiments/preprocessing_validation_small_new'
# dataset_path = '/home/ubuntu/experiments/preprocessing_train_2024-05-18_09-34-07'
# validation_path = '/home/ubuntu/experiments/preprocessing_validation_2024-05-18_09-43-19'
model_path = '/home/ubuntu/experiments/test_batch_training'
catboost_params = {
    'iterations': 2000,
    'depth': 8,
    'colsample_bylevel': 0.5
}
EVAL = True
N_BATCH = 5


def batch_training(model_path, train_ds, impression_time_ds, catboost_params, data_info):

    per_batch_elements = int(impression_time_ds.shape[0] / N_BATCH)

    for batch in range(N_BATCH):
        print(f'-------------BATCH {batch}-----------')
        output_dir = model_path + f'/model_{batch}.cbm'
        model = CatBoostRanker(
            **catboost_params, cat_features=data_info['categorical_columns'])

        if batch - 4 < 0:
            sampled_impressions = resample(
                impression_time_ds, replace=False, n_samples=per_batch_elements, stratify=impression_time_ds['impression_time'])
            impression_time_ds = impression_time_ds.filter(
                ~pl.col('impression_id').is_in(sampled_impressions['impression_id']))
        else:
            sampled_impressions = impression_time_ds
        print(f'Sampled {sampled_impressions.shape}')
        print(f'Remaining {impression_time_ds.shape}')
        print('Sampled DF')

        subsampled_train = train_ds.filter(pl.col('impression_id').is_in(
            sampled_impressions['impression_id']))
        subsampled_train = subsampled_train.sort(by='impression_id')
        groups = subsampled_train.select('impression_id').to_numpy().flatten()

        subsampled_train = subsampled_train.drop(
            ['impression_id', 'article', 'user_id', 'impression_time']).to_pandas()

        X = subsampled_train.drop(columns=['target'])
        y = subsampled_train['target']
        print(X.shape)

        if 'impression_time' in X:
            X = X.drop(['impression_time'])

        print('PreProcessed Data')
        model.fit(X, y, group_id=groups, verbose=20)

        model.save_model(output_dir)

    models = []
    for batch in range(N_BATCH):
        model = CatBoostRanker(
            **catboost_params, cat_features=data_info['categorical_columns'])
        model.load_model(model_path + f'/model_{batch}.cbm', format='cbm')
        models.append(model)
    weights = [1/N_BATCH] * N_BATCH

    model = sum_models(models, weights=weights,
                       ctr_merge_policy='IntersectingCountersAverage')

    return model


def incremental_training(model_path, train_ds, impression_time_ds, catboost_params, data_info):

    # must be testes
    output_dir = model_path + '/model.cbm'
    per_batch_elements = int(impression_time_ds.shape[0] / N_BATCH)

    model = CatBoostRanker(
        **catboost_params, cat_features=data_info['categorical_columns'])
    
    for batch in range(N_BATCH):
        print(f'-------------BATCH {batch}-----------')

        if batch - 4 < 0:
            sampled_impressions = resample(
                impression_time_ds, replace=False, n_samples=per_batch_elements, stratify=impression_time_ds['impression_time'])
            impression_time_ds = impression_time_ds.filter(
                ~pl.col('impression_id').is_in(sampled_impressions['impression_id']))
        else:
            sampled_impressions = impression_time_ds
        print('Sampled DF')
        subsampled_train = train_ds.filter(pl.col('impression_id').is_in(
            sampled_impressions['impression_id']))
        subsampled_train = subsampled_train.sort(by='impression_id')
        groups = subsampled_train.select('impression_id').to_numpy().flatten()

        subsampled_train = subsampled_train.drop(
            ['impression_id', 'article', 'user_id', 'impression_time']).to_pandas()

        X = subsampled_train.drop(columns=['target'])
        y = subsampled_train['target']

        if 'impression_time' in X:
            X = X.drop(['impression_time'])

        print('PreProcessed Data')
        if batch == 0:
            model.fit(X, y, group_id=groups, verbose=20)
        else:
            model.fit(X, y, group_id=groups, verbose=20, init_model=output_dir)

        model.save_model(output_dir)

    return model


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

    print(f'Starting to train the catboost model')

    model = batch_training(model_path, train_ds, impression_time_ds, catboost_params, data_info)
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

        pred = val_ds.with_columns(
            pl.Series(model.predict(X_val)).alias('prediction'))
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
