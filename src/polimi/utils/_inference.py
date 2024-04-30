import pandas as pd
import polars as pl
import numpy as np
import gc
from polimi.preprocessing_pipelines.pre_68f import strip_new_features


def _batch_predict(model, X, batch_size=None):
    if batch_size is None:
        return model.predict_proba(X)[:, 1]
    start_idx = 0
    predictions = np.empty((0,), dtype=np.float32)
    while start_idx < X.shape[0]:
        end_idx = start_idx + batch_size
        predictions = np.concatenate(
            [predictions, model.predict_proba(X.iloc[start_idx:end_idx])[:, 1]])
        start_idx = end_idx
    return predictions


def _inference(dataset_path, data_info, model, eval=False, batch_size=1000):

    inference_ds = pd.read_parquet(dataset_path)

    inference_ds[data_info['categorical_columns']
                 ] = inference_ds[data_info['categorical_columns']].astype('category')
    
    inference_ds = strip_new_features(inference_ds)

    if 'target' not in inference_ds.columns and eval:
        raise ValueError(
            'Target column not found in dataset. Cannot evaluate.')

    if 'target' in inference_ds.columns:
        evaluation_ds = pl.from_pandas(
            inference_ds[['impression_id', 'user_id', 'article', 'target']]) 
        X = inference_ds.drop(
            columns=['impression_id', 'target', 'article', 'user_id','impression_time'])
    else:
        evaluation_ds = pl.from_pandas(
            inference_ds[['impression_id', 'user_id', 'article']])
        X = inference_ds.drop(columns=['impression_id', 'article', 'user_id','impression_time'])

    evaluation_ds = evaluation_ds.with_columns(
        pl.Series(_batch_predict(model, X, batch_size)).alias('prediction'))

    del X, inference_ds
    gc.collect()
    return evaluation_ds