
import polars as pl
import pandas as pd
import os
import json
import numpy as np
import numpy as np
import gc
from tqdm import tqdm
from pathlib import Path
import logging
from catboost import CatBoostRanker, sum_models
from polimi.utils._inference import _inference
from ebrec.utils._python import write_submission_file
import gc
from ebrec.evaluation.metrics_protocols import *

dataset_path = '/mnt/ebs_volume/experiments/preprocessing_test_new'
behaviors_path = '/home/ubuntu/dataset/ebnerd_testset/test/behaviors.parquet'
model_path = '/mnt/ebs_volume/models/catboost_ranker_baseline_batch_val'
dataset_name = 'test'
ranker = True
batch_size = 1000000
save_results = True
output_dir = '/home/ubuntu/experiments'

catboost_params = {
    'iterations': 2000,
    'depth': 8,
    'colsample_bylevel': 0.5
}

N_BATCH = 10


def _batch_predict(model, X, batch_size=None, ranker=False):
    if batch_size is None:
        if ranker:
            return model.predict(X)
        return model.predict_proba(X)[:, 1]
    start_idx = 0
    predictions = np.empty((0,), dtype=np.float32)
    with tqdm(total=X.shape[0] // batch_size) as pbar:
        while start_idx < X.shape[0]:
            end_idx = start_idx + batch_size
            if ranker:
                predictions = np.concatenate(
                    [predictions, model.predict(X.iloc[start_idx:end_idx])])
            else:
                predictions = np.concatenate(
                    [predictions, model.predict_proba(X.iloc[start_idx:end_idx])[:, 1]])
            start_idx = end_idx
            pbar.update(1)
            gc.collect()
    return predictions


def _inference(dataset_path, data_info, model, eval=False, batch_size=1000, ranker=False):
    logging.info(f'Reading dataset from {dataset_path}')
    inference_ds = pl.read_parquet(dataset_path)
    logging.info(f'Dataset read complete')

    if 'target' not in inference_ds.columns and eval:
        raise ValueError(
            'Target column not found in dataset. Cannot evaluate.')
        
    if 'postcode' in inference_ds.columns:
        inference_ds = inference_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in inference_ds.columns:
        inference_ds = inference_ds.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in inference_ds.columns:
        inference_ds = inference_ds.drop(['impression_time'])

    if 'target' in inference_ds.columns:
        evaluation_ds = inference_ds.select(['impression_id', 'user_id', 'article', 'target'])
        X = inference_ds.drop(['impression_id', 'target', 'article', 'user_id']).to_pandas()
    else:
        evaluation_ds = inference_ds.select(['impression_id', 'user_id', 'article'])
        X = inference_ds.drop(['impression_id', 'article', 'user_id']).to_pandas()
    
    X[data_info['categorical_columns']] = X[data_info['categorical_columns']].astype('category')
        
    logging.info('Starting to predict in batches')
    evaluation_ds = evaluation_ds.with_columns(
        pl.Series(_batch_predict(model, X, batch_size, ranker)).alias('prediction'))

    del X, inference_ds
    gc.collect()
    return evaluation_ds


if __name__ == '__main__':
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    print(data_info['categorical_columns'])
        
    models = []
    for batch in range(N_BATCH):
        model = CatBoostRanker(
            **catboost_params, cat_features=data_info['categorical_columns'])
        model.load_model(model_path + f'/model_{batch}.cbm', format='cbm')
        models.append(model)
    weights = [1/N_BATCH] * N_BATCH
    
    model = sum_models(models, weights=weights,
                       ctr_merge_policy='IntersectingCountersAverage')
    
    evaluation_ds = _inference(os.path.join(
            dataset_path, f'{dataset_name}_ds.parquet'), data_info, model, False , batch_size, ranker)
    
    max_impression = evaluation_ds.select(
        pl.col('impression_id').max()).item(0, 0)
    
    if save_results:
        evaluation_ds.write_parquet(os.path.join(
            output_dir, f'predictions.parquet'))
        path = Path(os.path.join(output_dir, 'predictions.txt'))

        # need to maintain the same order of the inview list
        behaviors = pl.read_parquet(behaviors_path, columns=[
                                    'impression_id', 'article_ids_inview', 'user_id'])
        ordered_predictions = behaviors.explode('article_ids_inview').with_row_index() \
            .join(evaluation_ds, left_on=['impression_id', 'article_ids_inview', 'user_id'],
                  right_on=['impression_id', 'article', 'user_id'], how='left') \
            .sort('index').group_by(['impression_id', 'user_id'], maintain_order=True).agg(pl.col('prediction'), pl.col('article_ids_inview')) \
            .with_columns(pl.col('prediction').list.eval(pl.element().rank(descending=True)).cast(pl.List(pl.Int16)))

        logging.info('Debugging predictions')
        logging.info(behaviors.filter(pl.col('impression_id') == max_impression).select(
            ['impression_id', 'article_ids_inview']).explode('article_ids_inview'))
        logging.info(evaluation_ds.filter(pl.col('impression_id') == max_impression).select(
            ['impression_id', 'article', 'prediction']))
        logging.info(ordered_predictions.filter(pl.col('impression_id') == max_impression)
                     .select(['impression_id', 'article_ids_inview', 'prediction'])
                     .explode(['article_ids_inview', 'prediction']))

        logging.info(f'Saving Results at: {path}')
        write_submission_file(ordered_predictions['impression_id'].to_list(),
                              ordered_predictions['prediction'].to_list(),
                              path)
    