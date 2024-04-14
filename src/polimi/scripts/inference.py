import os
import logging
from datetime import datetime
import argparse
import pandas as pd
import joblib
import json
import numpy as np
from typing_extensions import List
import polars as pl
from pathlib import Path
import tqdm
import gc

import sys
sys.path.append('/home/ubuntu/RecSysChallenge2024/src')

from ebrec.evaluation.metrics_protocols import *
from ebrec.utils._python import write_submission_file


LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def batch_predict(model, X, batch_size=None):
    if batch_size is None:
        return model.predict_proba(X)[:, 1]
    start_idx = 0
    predictions = np.empty((0,), dtype=np.float32)
    while start_idx < X.shape[0]:
        end_idx = start_idx + batch_size
        predictions = np.concatenate([predictions, model.predict_proba(X.iloc[start_idx:end_idx])[:, 1]])
        start_idx = end_idx
    return predictions
    

def main(dataset_path, model_path, save_results, eval, behaviors_path, output_dir, batch_size=None):
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")
    
    dataset_name = 'validation' if eval else 'test'
    
    inference_ds = pd.read_parquet(os.path.join(dataset_path, f'{dataset_name}_ds.parquet'))
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Data info: {data_info}')
    
    inference_ds[data_info['categorical_columns']] = inference_ds[data_info['categorical_columns']].astype('category')

    if 'target' not in inference_ds.columns and eval:
        raise ValueError('Target column not found in dataset. Cannot evaluate.')
    
    if 'target' in inference_ds.columns:
        evaluation_ds = pl.from_pandas(inference_ds[['impression_id', 'user_id', 'article', 'target']])
        X = inference_ds.drop(columns=['impression_id', 'target', 'article', 'user_id'])
    else:
        evaluation_ds = pl.from_pandas(inference_ds[['impression_id', 'user_id', 'article']])
        X = inference_ds.drop(columns=['impression_id', 'article', 'user_id'])
        
    max_impression = evaluation_ds.select(pl.col('impression_id').max()).item(0,0)
    
    logging.info(f'Features ({len(X.columns)}): {np.array(list(X.columns))}')
    logging.info(f'Categorical features: {np.array(data_info["categorical_columns"])}')
    logging.info(f'Reading model parameters from path: {model_path}')
    logging.info('Starting inference.')
    
    model = joblib.load(model_path)
    evaluation_ds = evaluation_ds.with_columns(pl.Series(batch_predict(model, X, batch_size)).alias('prediction'))  
    
    del X, inference_ds 
    gc.collect()     
    
    logging.info('Inference completed.')
    
    if eval:
        evaluation_ds_grouped = evaluation_ds.group_by('impression_id').agg(pl.col('target'), pl.col('prediction'))
        met_eval = MetricEvaluator(
            labels=evaluation_ds_grouped['target'].to_list(),
            predictions=evaluation_ds_grouped['prediction'].to_list(),
            metric_functions=[
                AucScore(),
                MrrScore(),
                NdcgScore(k=5),
                NdcgScore(k=10),
            ],
        )
        logging.info(f'Evaluation results: {met_eval.evaluate()}')
        
    if save_results:
        path = Path(os.path.join(output_dir, 'predictions.txt'))
        
        # need to maintain the same order of the inview list
        behaviors = pl.read_parquet(behaviors_path , columns=['impression_id', 'article_ids_inview', 'user_id'])
        # ordered_predictions = pl.concat(
        #     rows.explode('article_ids_inview').with_row_index() \
        #         .join(evaluation_ds, left_on=['impression_id', 'article_ids_inview', 'user_id'], 
        #               right_on=['impression_id', 'article', 'user_id'], how='left') \
        #         .sort('index').group_by('impression_id').agg(pl.col('prediction'), pl.col('article_ids_inview')) \
        #         .with_columns(pl.col('prediction').list.eval(pl.element().rank(descending=True)))
        #     for rows in tqdm.tqdm(behaviors.iter_slices(10000), total=behaviors.shape[0] // 10000)
        # )
        ordered_predictions = behaviors.explode('article_ids_inview').with_row_index() \
            .join(evaluation_ds, left_on=['impression_id', 'article_ids_inview', 'user_id'], 
                  right_on=['impression_id', 'article', 'user_id'], how='left') \
            .sort('index').group_by('impression_id').agg(pl.col('prediction'), pl.col('article_ids_inview')) \
            .with_columns(pl.col('prediction').list.eval(pl.element().rank(descending=True)))
           
        logging.info('Debugging predictions') 
        logging.info(behaviors.filter(pl.col('impression_id') == max_impression).select(['impression_id', 'article_ids_inview']).explode('article_ids_inview'))
        logging.info(evaluation_ds.filter(pl.col('impression_id') == max_impression).select(['impression_id', 'article', 'prediction']))
        logging.info(ordered_predictions.filter(pl.col('impression_id') == max_impression) \
                .select(['impression_id', 'article_ids_inview', 'prediction']) \
                .explode(['article_ids_inview', 'prediction']))
        
        logging.info(f'Saving Results at: {path}')
        write_submission_file(ordered_predictions['impression_id'].to_list(), 
                              ordered_predictions['prediction'].to_list(),
                              path)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for catboost")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the preprocessed dataset is placed")
    parser.add_argument("-model_path", default=None, type=str, required=True,
                        help="File path where the model is placed")
    parser.add_argument("--submit", action='store_true', default=False, help='Whether to save the predictions or not')
    parser.add_argument('--eval', action='store_true', default=False, help='Whether to evaluate the predictions or not')
    parser.add_argument("-behaviors_path", default=None, 
                        help="The file path of the reference behaviors ordering. Mandatory to save predictions")
    parser.add_argument('-batch_size', default=None, type=int, required=False,
                        help='If passed, it will predict in batches to reduce the memory usage')
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    MODEL_PATH = args.model_path
    SAVE_INFERENCE = args.submit
    EVAL = args.eval
    BEHAVIORS_PATH = args.behaviors_path
    BATCH_SIZE = args.batch_size
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'Inference_Test_{timestamp}' if not EVAL else f'Inference_Validation_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_DIR, MODEL_PATH, SAVE_INFERENCE, EVAL, BEHAVIORS_PATH, output_dir, BATCH_SIZE)
