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

from ebrec.evaluation.metrics_protocols import *
from ebrec.utils._python import write_submission_file


LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def main(dataset_path, model_path, save_results, eval, behaviors_path, output_dir):
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")
    
    train_ds = pd.read_parquet(os.path.join(dataset_path, 'train_ds.parquet'))
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Data info: {data_info}')
    
    train_ds = train_ds.drop(columns=['article', 'user_id'])
    train_ds[data_info['categorical_columns']] = train_ds[data_info['categorical_columns']].astype('category')

    if 'target' not in train_ds.columns and eval:
        raise ValueError('Target column not found in dataset. Cannot evaluate.')
    
    if eval:
        evaluation_ds = pl.from_pandas(train_ds[['impression_id', 'article', 'target']])
        X = train_ds.drop(columns=['impression_id', 'target'])
        y = train_ds['target']
    else:
        evaluation_ds = pl.from_pandas(train_ds[['impression_id', 'article']])
        X = train_ds.drop(columns=['impression_id'])
    
    logging.info(f'Features ({len(X.columns)}): {np.array(list(X.columns))}')
    logging.info(f'Categorical features: {np.array(data_info["categorical_columns"])}')
    logging.info(f'Reading model parameters from path: {model_path}')
    logging.info('Starting inference.')
    
    model = joblib.load(model_path)
    evaluation_ds = evaluation_ds.with_columns(pl.Series(model.predict_proba(X)[:, 1]).alias('prediction'))
    
    if eval:
        evaluation_ds_grouped = evaluation_ds.group_by('impression_id').agg(pl.col('target'), pl.col('prediction'))
    
    logging.info('Inference completed.')
    
    if eval:
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
        behaviors = pl.read_parquet(behaviors_path , columns=['impression_id', 'article_ids_inview'])
        ordered_predictions = behaviors.explode('article_ids_inview').with_row_index() \
            .join(evaluation_ds, left_on='article_ids_inview', right_on='article', how='left') \
            .sort('index').group_by('impression_id').agg(pl.col('prediction'))
        
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
    parser.add_argument("-submit", action='store_false', help='Whether to save the predictions or not')
    parser.add_argument("-eval", action='store_false', help='Whether to evaluate the predictions or not')
    parser.add_argument("")
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    MODEL_PATH = args.model_path
    SAVE_INFERENCE = args.submit
    EVAL = args.eval
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'Inference_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_DIR, MODEL_PATH, SAVE_INFERENCE, EVAL, output_dir)
