import pandas as pd
import polars as pl
import numpy as np
import gc
from tqdm import tqdm
import logging
from lightgbm import LGBMClassifier, LGBMRanker, Booster


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
            if ranker or isinstance(model, Booster):
                predictions = np.concatenate(
                    [predictions, model.predict(X.iloc[start_idx:end_idx])])
            else:
                predictions = np.concatenate(
                    [predictions, model.predict_proba(X.iloc[start_idx:end_idx])[:, 1]])
            start_idx = end_idx
            pbar.update(1)
            gc.collect()
    return predictions


def _inference(dataset_path, data_info, model, eval=False, batch_size=1000, ranker=False,drop_features=False,features_to_keep=None):
    logging.info(f'Reading dataset from {dataset_path}')
    inference_ds = pl.scan_parquet(dataset_path).collect()
    logging.info(f'Dataset read complete')

    if eval and drop_features:
        features_to_keep.append('target')

    if drop_features:
        inference_ds = inference_ds.select(features_to_keep)

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


def _batch_inference(dataset_path, data_info, model, eval=False, batch_size=1000, ranker=False,drop_features=False,features_to_keep=None, is_xgboost=False):
    logging.info(f'Reading dataset from {dataset_path}')
    inference_all_ds = pl.scan_parquet(dataset_path)
    n_batches = 10
    logging.info(f'Dataset read complete')
    if eval and drop_features:
        features_to_keep.append('target')

    per_batch_elements = int(inference_all_ds.select('impression_id').collect().shape[0] / n_batches)
    starting_index = 0
    evaluations = []
    
    for batch in range(n_batches):
        print(f'--------------Processing Batch {batch}---------------')
        if batch == n_batches - 1:
            inference_ds = inference_all_ds.slice(starting_index, None).collect()
        else :
            inference_ds = inference_all_ds.slice(starting_index, per_batch_elements).collect()
        
        starting_index = starting_index + per_batch_elements
        
        if drop_features:
            inference_ds = inference_ds.select(features_to_keep)

        if 'target' not in inference_ds.columns and eval:
            raise ValueError(
                'Target column not found in dataset. Cannot evaluate.')
            
        
        if is_xgboost:
            features = model.get_booster().feature_names
            if 'target' in inference_ds.columns:
                features = features + ['target']
            features = features + ['impression_id', 'user_id', 'article']
            inference_ds = inference_ds.select(features)
            assert all([features[i] == inference_ds.columns[i] for i in range(len(features))]), 'XGB requires features to be ordered in the same way as the model was trained on.'
            
        if isinstance(model, LGBMClassifier) or isinstance(model, LGBMRanker) or isinstance(model, Booster):
            logging.info('Reordering feature names for lightgbm')
            if isinstance(model, Booster):
                features = model.feature_name() + ['impression_id', 'user_id', 'article']
            else:
                features = model.feature_name_ + ['impression_id', 'user_id', 'article']
            if 'target' in inference_ds.columns:
                features = features + ['target']
            inference_ds = inference_ds.select(features)
            assert all([features[i] == inference_ds.columns[i] for i in range(len(features))]), 'LGBM requires features to be ordered in the same way as the model was trained on.'
            
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
        evaluations.append(evaluation_ds)
        
    return pl.concat(evaluations, how='vertical_relaxed')
