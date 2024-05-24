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

# BEST FOR RANKER NPRTIO == 9, NP_RATIO == 8 SAME AS NP_RATIO == ALL

dataset_path = '/home/ubuntu/experiments/preprocessing_train_2024-05-18_09-34-07'
original_datset_path = '/home/ubuntu/dataset/ebnerd_demo/train/behaviors.parquet'
validation_path = '/home/ubuntu/experiments/preprocessing_validation_2024-05-18_09-43-19'

catboost_params = {
    "iterations": 1000,
    "subsample": 0.5,
    "rsm": 0.7
}

EVAL = True
SAVE_FEATURES_ORDER = False
N_BATCH = 10
NP_RATIOS = [-1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def train_model(np_ratio):
    train_ds = reduce_polars_df_memory_size(pl.read_parquet(
        os.path.join(dataset_path, 'train_ds.parquet')), verbose=False)

    starting_dataset = pl.read_parquet(original_datset_path).select(
        ['impression_id', 'user_id', 'article_ids_inview', 'article_ids_clicked'])
    if np_ratio != -1:
        behaviors = pl.concat(
            rows.pipe(
                sampling_strategy_wu2019, npratio=np_ratio, shuffle=False, with_replacement=True, seed=123
            ).explode('article_ids_inview').drop(columns=['article_ids_clicked']).rename({'article_ids_inview': 'article'})
            .with_columns(pl.col('user_id').cast(pl.UInt32),
                        pl.col('article').cast(pl.Int32))

            for rows in tqdm(starting_dataset.iter_slices(1000), total=starting_dataset.shape[0] // 1000)
        )
        train_ds = behaviors.join(
            train_ds, on=['impression_id', 'user_id', 'article'], how='left')
    
    print('Train df shape:')
    print(train_ds.shape)
    column_oder = train_ds.columns
    # train_ds = train_ds.vstack(self_ds.select(column_oder))
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
    print(f'Starting to train the catboost model')

    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)

    categorical_columns = data_info['categorical_columns']
    categorical_columns = [
        cat for cat in categorical_columns if cat in column_oder]

    if 'postcode' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in train_ds.columns:
        train_ds = train_ds.with_columns(
            pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in train_ds.columns:
        train_ds = train_ds.drop(['impression_time'])

    train_ds = train_ds.sort(by='impression_id')
    groups = train_ds.select('impression_id').to_numpy().flatten()
    train_ds = train_ds.drop(
        ['impression_id', 'article', 'user_id']).to_pandas()
    train_ds[categorical_columns] = train_ds[categorical_columns].astype(
        'category')
    
    X = train_ds.drop(columns=['target'])
    print(X.columns[0])
    y = train_ds['target']

    # model = CatBoostRanker(
    #         **catboost_params, cat_features=data_info['categorical_columns'])
    # model.fit(X, y, group_id=groups, verbose=100)
    model = CatBoostClassifier(
        **catboost_params, cat_features=categorical_columns)
    model.fit(X, y, verbose=100)

    return model, column_oder


def eval_model(model, columns):
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)

    categorical_columns = data_info['categorical_columns']
    categorical_columns = [
        cat for cat in categorical_columns if cat in columns]

    val_ds = pl.read_parquet(
          validation_path + '/validation_ds.parquet').select(columns)

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

    X_val = val_ds_pandas.drop(columns=['target'])
    y_val = val_ds_pandas['target']

    evaluation_ds = val_ds[['impression_id', 'article', 'target']]
    prediction_ds = evaluation_ds.with_columns(
                        pl.Series(model.predict_proba(X_val)[:, 1]).alias('prediction')
                    )
    # prediction_ds = evaluation_ds.with_columns(
    #                     pl.Series(model.predict(X_val)).alias('prediction')
    #                 )
    prediction_ds = prediction_ds.group_by('impression_id').agg(
            pl.col('target'), pl.col('prediction'))

    cpp_auc = CppAuc()
    result = np.mean(
            [cpp_auc.roc_auc_score(np.array(y_t).astype(bool), np.array(y_s).astype(np.float32))
                for y_t, y_s in zip(prediction_ds['target'].to_list(),
                                    prediction_ds['prediction'].to_list())]
        )

    return result


if __name__ == '__main__':
    results = []
    
    for np_ratio in NP_RATIOS:
        print(f'-----------NP RATIO : {np_ratio}-------')
        model, column_oder = train_model(np_ratio)
        res = eval_model(model, column_oder)
        print(f'AUC : {res}')
        print('-----------------------------------------')
        results.append(res)
    
    file = open('/home/ubuntu/experiments/test_batch_training/np_analysis.txt','w')
    for item in results:
        file.write(item+"\n")
    file.close()