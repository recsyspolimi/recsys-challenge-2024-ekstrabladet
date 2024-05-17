import pandas as pd
import polars as pl
import numpy as np
import gc
from tqdm import tqdm
import logging


def _batch_predict(model, X, batch_size=None):
    if batch_size is None:
        return model.predict_proba(X)[:, 1]
    start_idx = 0
    predictions = np.empty((0,), dtype=np.float32)
    with tqdm(total=X.shape[0] // batch_size) as pbar:
        while start_idx < X.shape[0]:
            end_idx = start_idx + batch_size
            predictions = np.concatenate(
                [predictions, model.predict_proba(X.iloc[start_idx:end_idx])[:, 1]])
            start_idx = end_idx
            pbar.update(1)
            gc.collect()
    return predictions


def _inference(dataset_path, data_info, model, eval=False, batch_size=1000):
    logging.info(f'Reading dataset from {dataset_path}')
    inference_ds = pl.read_parquet(dataset_path).to_pandas()
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
        pl.Series(_batch_predict(model, X, batch_size)).alias('prediction'))

    del X, inference_ds
    gc.collect()
    return evaluation_ds
