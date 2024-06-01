import polars as pl
import pandas as pd
import os
import json
import numpy as np
from catboost import CatBoostClassifier, CatBoostRanker, Pool, sum_models
import gc
from ebrec.evaluation.metrics_protocols import *
from fastauc.fastauc.fast_auc import CppAuc
import argparse

RANKER = True
TRAIN_VAL = True
dataset_path = '/home/ubuntu/experiments/preprocessing_train_new'
validation_path = '/home/ubuntu/experiments/preprocessing_validation_new'
batch_split_directory = '/home/ubuntu/experiments/batches_train_val_new/batches'

model_path = '/home/ubuntu/experiments/batches_train_val_new/models'
catboost_params = {
    'iterations': 2421,
    'learning_rate': 0.061372161824290145,
    'rsm': 0.681769606695633,
    'reg_lambda': 0.4953354255208565,
    'grow_policy': 'SymmetricTree',
    'bootstrap_type': 'MVS',
    'subsample': 0.5108219602277233,
    'random_strength': 14.089062269780399,
    'fold_permutation_block': 39,
    'border_count': 34,
    'sampling_frequency': 'PerTreeLevel',
    'score_function': 'Cosine',
    'depth': 8,
    'mvs_reg': 0.0015341832942953422
}


EVAL = False
SAVE_PREDICTIONS = False
N_BATCH = 10
BATCHES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def load_batch(dataset_path, batch_split_directory, batch_index):

    train_ds = pl.scan_parquet(dataset_path + '/train_ds.parquet')

    if TRAIN_VAL:
        val_ds = pl.scan_parquet(validation_path + '/validation_ds.parquet')
    batch = pl.scan_parquet(batch_split_directory +
                            f'/batch_{batch_index}.parquet').collect()

    subsampled_train = train_ds.filter(pl.col('impression_id').is_in(
        batch.select('impression_id'))).collect()
    columns = subsampled_train.columns

    if TRAIN_VAL:
        subsampled_val = val_ds.filter(pl.col('impression_id').is_in(
            batch.select('impression_id'))).select(columns).collect()
        subsampled_train = pl.concat(
            [subsampled_train, subsampled_val], how='vertical_relaxed')

    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
    
    if RANKER:
        subsampled_train = subsampled_train.sort(by='impression_id')
        groups = subsampled_train.select('impression_id').to_numpy().flatten()
        
    if 'postcode' in subsampled_train.columns:
        subsampled_train = subsampled_train.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in subsampled_train.columns:
        subsampled_train = subsampled_train.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in subsampled_train.columns:
        subsampled_train = subsampled_train.drop(['impression_time'])
    
    subsampled_train = subsampled_train.drop(['impression_id', 'article', 'user_id']).to_pandas()
    subsampled_train[data_info['categorical_columns']] = subsampled_train[data_info['categorical_columns']].astype('category')

    X = subsampled_train.drop(columns=['target'])
    y = subsampled_train['target']
    
    print(X.shape)

    if 'impression_time' in X:
        X = X.drop(['impression_time'])

    del train_ds, batch, subsampled_train
    gc.collect()

    if RANKER:
        return X, y, groups
    else:
        return X, y


def train_single_batch(batch, model_path, catboost_params, data_info, dataset_path, batch_split_directory, RANKER):
    print(f'-------------BATCH {batch}-----------')
    output_dir = model_path + f'/model_{batch}.cbm'
    if RANKER:
        model = CatBoostRanker(
            **catboost_params, cat_features=data_info['categorical_columns'])
    else:
        model = CatBoostClassifier(
            **catboost_params, cat_features=data_info['categorical_columns'])

    print(f'Collecting batch...')
    if RANKER:
        X, y, groups = load_batch(dataset_path, batch_split_directory, batch)
        print('Fitting Model...')
        model.fit(X, y, group_id=groups, verbose=20)
    else :
        X, y = load_batch(dataset_path, batch_split_directory, batch)
        print('Fitting Model...')
        model.fit(X, y, group_id=groups, verbose=20)
        
    model.save_model(output_dir)
    del model, X, y, groups
    gc.collect()


def batch_training(model_path, catboost_params, data_info, dataset_path, batch_split_directory, RANKER):

    for batch in BATCHES:
        train_single_batch(batch, model_path, catboost_params, data_info, dataset_path, batch_split_directory, RANKER)

    models = []
    for batch in range(N_BATCH):
        if RANKER:
            model = CatBoostRanker(
                **catboost_params, cat_features=data_info['categorical_columns'])
        else:
            model = CatBoostClassifier(
                **catboost_params, cat_features=data_info['categorical_columns'])

        model.load_model(model_path + f'/model_{batch}.cbm', format='cbm')
        models.append(model)
    weights = [1/N_BATCH] * N_BATCH

    model = sum_models(models, weights=weights,
                       ctr_merge_policy='IntersectingCountersAverage')

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for catboost")
    parser.add_argument("-batch", default="../../experiments/", type=int,
                        help="The number of the batch")
    args = parser.parse_args()
    batch = args.batch
    
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)

    print(f'Data info: {data_info}')

    print(f'Starting to train the catboost model')
    
    train_single_batch(batch, model_path, catboost_params, data_info, dataset_path, batch_split_directory, RANKER)
    # model = batch_training(model_path, catboost_params,
    #                        data_info, dataset_path, batch_split_directory, RANKER)
    # model = incremental_training(model_path, train_ds, impression_time_ds, catboost_params, data_info)

    if EVAL:
        print('Reading models...')
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
        
        print('Reading DF...')
        with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
            data_info = json.load(data_info_file)

        val_ds = pl.read_parquet(validation_path + '/validation_ds.parquet')

        if 'postcode' in val_ds.columns:
            val_ds = val_ds.with_columns(pl.col('postcode').fill_null(5))
        if 'article_type' in val_ds.columns:
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
        
        print('Running Model...')
        pred = val_ds.with_columns(
                pl.Series(model.predict(X_val)).alias('prediction'))
        # if RANKER:
        #     pred = val_ds.with_columns(
        #         pl.Series(model.predict(X_val)).alias('prediction'))
        # else:
        #     pred = val_ds.with_columns(
        #         pl.Series(model.predict_proba(X_val)[:, 1]).alias('prediction'))
        if SAVE_PREDICTIONS:
            pred.write_parquet(
                '/home/ubuntu/experiments/test_batch_training/ranker_predictions.parquet')
        gc.collect()
        
        evaluation_ds = pred.group_by('impression_id').agg(
            pl.col('target'), pl.col('prediction'))
        valuation_ds = pred.group_by('impression_id').agg(
            pl.col('target'), pl.col('prediction'))
        cpp_auc = CppAuc()
        
        print('Scoring...')
        result = np.mean(
                [cpp_auc.roc_auc_score(np.array(y_t).astype(bool), np.array(y_s).astype(np.float32)) 
                    for y_t, y_s in zip(evaluation_ds['target'].to_list(), 
                                        evaluation_ds['prediction'].to_list())]
            )
        print(result)
    

