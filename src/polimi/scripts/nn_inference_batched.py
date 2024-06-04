from polimi.utils._inference import _inference
from ebrec.utils._python import write_submission_file
from ebrec.evaluation.metrics_protocols import *
import os
import logging
from datetime import datetime
import argparse
import pandas as pd
import joblib
import json
import numpy as np
from typing_extensions import List, Tuple, Dict, Type, TypeVar
import polars as pl
from pathlib import Path
from tqdm import tqdm
import gc
from catboost import CatBoostClassifier

import sys
sys.path.append('/home/ubuntu/RecSysChallenge2024/src')

from polimi.utils.tf_models import *
from polimi.utils.tf_models.utils import *


LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
T = TypeVar('T', bound=TabularNNModel)


def get_model_class(name: str = 'mlp') -> T:
    if name == 'MLP':
        return MLP
    elif name == 'DeepCrossNetwork':
        return DeepCrossNetwork
    elif name == 'WideDeepNetwork':
        return WideDeepNetwork
    elif name == 'DeepAbstractNetwork':
        return DeepAbstractNetwork
    elif name == 'GANDALF':
        return GANDALF
    
    
def _batch_predict(model: TabularNNModel, X, batch_size=None, inner_batch_size=1024):
    start_idx = 0
    predictions = np.empty((0,), dtype=np.float32)
    with tqdm(total=X.shape[0] // batch_size) as pbar:
        while start_idx < X.shape[0]:
            end_idx = start_idx + batch_size
            predictions = np.concatenate(
                [predictions, model.predict(X.iloc[start_idx:end_idx].copy(), batch_size=inner_batch_size).flatten()])
            start_idx = end_idx
            pbar.update(1)
            gc.collect()
    return predictions


def main(dataset_path, model_path, save_results, eval, behaviors_path, output_dir, batch_size, params_file, n_blocks):
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")

    dataset_name = 'validation' if eval else 'test'
    logging.info(f'Reading dataset from {dataset_path}')
    inference_all_ds = pl.scan_parquet(os.path.join(dataset_path, f'{dataset_name}_ds.parquet'))
    logging.info(f'Dataset read complete')
    
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)

    if 'target' not in inference_all_ds.columns and eval:
        raise ValueError(
            'Target column not found in dataset. Cannot evaluate.')
        
    logging.info(f'Data info: {data_info}')
    logging.info(
        f'Categorical features: {np.array(data_info["categorical_columns"])}')
    logging.info(f'Reading neural network parameters from path: {params_file}')
    
    with open(params_file, 'r') as params_file:
        params = json.load(params_file)
        
    logging.info(f'Params: {params}')
    logging.info(f'Loading the {params["model_name"]} model')
    
    categorical_columns = data_info['categorical_columns']
    ignore_columns = ['impression_id', 'article', 'user_id', 'impression_time', 'target']
    numerical_columns = [c for c in inference_all_ds.columns if c not in categorical_columns + ignore_columns]
    
    model: TabularNNModel = get_model_class(params['model_name'])(categorical_features=categorical_columns, 
                                                                  numerical_features=numerical_columns, 
                                                                  **params['model_hyperparams'])
    model.load(model_path)
        
    per_batch_elements = int(inference_all_ds.select('impression_id').collect().shape[0] / n_blocks)
    starting_index = 0
    evaluations = []
        
    for block in range(n_blocks):
        logging.info(f'Processing Batch {block}')
        if block == n_blocks - 1:
            inference_ds = inference_all_ds.slice(starting_index, None).collect()
        else :
            inference_ds = inference_all_ds.slice(starting_index, per_batch_elements).collect()
        
        starting_index = starting_index + per_batch_elements
        
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

        logging.info('Starting inference.')
        evaluation_ds = evaluation_ds.with_columns(
            pl.Series(_batch_predict(model, X, batch_size=200000, inner_batch_size=batch_size).flatten()).alias('prediction'))
        evaluations.append(evaluation_ds)
        
        del X, inference_ds
        gc.collect()

    evaluation_ds = pl.concat(evaluations, how='vertical_relaxed')
    max_impression = evaluation_ds.select(
        pl.col('impression_id').max()).item(0, 0)

    logging.info('Inference completed.')

    if eval:
        evaluation_ds_grouped = evaluation_ds.group_by(
            'impression_id').agg(pl.col('target'), pl.col('prediction'))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Training script for catboost")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the preprocessed dataset is placed")
    parser.add_argument("-model_path", default=None, type=str, required=True,
                        help="Root directory where the model files are placed")
    parser.add_argument("-params_file", default=None, type=str, required=True,
                        help="File path where the hyperparameters are placed")
    parser.add_argument("--submit", action='store_true', default=False,
                        help='Whether to save the predictions or not')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Whether to evaluate the predictions or not')
    parser.add_argument("-behaviors_path", default=None,
                        help="The file path of the reference behaviors ordering. Mandatory to save predictions")
    parser.add_argument('-n_blocks', default=10, type=int, required=False,
                        help='The number of inference blocks. Only one block at a time will be loaded in memory')
    parser.add_argument('-batch_size', default=None, type=int, required=False,
                        help='If passed, each block will be predicted in batches to reduce the memory usage')
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    MODEL_PATH = args.model_path
    SAVE_INFERENCE = args.submit
    EVAL = args.eval
    BEHAVIORS_PATH = args.behaviors_path
    BATCH_SIZE = args.batch_size
    HYPERPARAMS_PATH = args.params_file
    N_BLOCKS = args.n_blocks

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'Inference_Test_NN_{timestamp}' if not EVAL else f'Inference_Validation_NN_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)

    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w",
                        format=LOGGING_FORMATTER, level=logging.INFO, force=True)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)

    main(DATASET_DIR, MODEL_PATH, SAVE_INFERENCE, EVAL, BEHAVIORS_PATH, 
         output_dir, BATCH_SIZE, HYPERPARAMS_PATH, N_BLOCKS)
