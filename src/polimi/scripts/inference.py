from polimi.utils._inference import _inference, _batch_inference
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
from typing_extensions import List
import polars as pl
from pathlib import Path
import tqdm
import gc
from catboost import CatBoostClassifier, CatBoostRanker
from fastauc.fastauc.fast_auc import CppAuc

import sys
sys.path.append('/home/ubuntu/RecSysChallenge2024/src')


LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def main(dataset_path, model_path, save_results, eval, behaviors_path, output_dir, batch_size=None, ranker = False, is_xgboost= False):
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")

    dataset_name = 'validation' if eval else 'test'
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)

    logging.info(f'Data info: {data_info}')
    logging.info(
        f'Categorical features: {np.array(data_info["categorical_columns"])}')
    logging.info(f'Reading model parameters from path: {model_path}')
    logging.info('Starting inference.')

    model = joblib.load(model_path)
    
    if isinstance(model, CatBoostClassifier) or isinstance(model, CatBoostRanker):
        cat_features = model.get_param('cat_features')
        print(f'CatBoost categorical columns: {cat_features}')
    
    evaluation_ds = _batch_inference(os.path.join(
            dataset_path, f'{dataset_name}_ds.parquet'), data_info, model, eval, batch_size, ranker, is_xgboost=is_xgboost)
    # evaluation_ds = _inference(os.path.join(
    #         dataset_path, f'{dataset_name}_ds.parquet'), data_info, model, eval, batch_size, ranker, is_xgboost)
    
    # if dataset_name == 'validation':
    #     evaluation_ds = _inference(os.path.join(
    #         dataset_path, f'{dataset_name}_ds.parquet'), data_info, model, eval, batch_size)
    # else:
    #     evaluation_ds = pl.concat(
    #         [_inference(os.path.join(dataset_path, f'Sliced_ds/test_slice_{i}.parquet'), data_info, model, eval, batch_size)
    #          for i in tqdm.tqdm(range(0, 101))], how='vertical_relaxed'
    #     )

    max_impression = evaluation_ds.select(
        pl.col('impression_id').max()).item(0, 0)

    gc.collect()

    logging.info('Inference completed.')

    if eval:
        evaluation_ds.write_parquet(os.path.join(
            output_dir, f'predictions.parquet'))
        evaluation_ds_grouped = evaluation_ds.group_by(
            'impression_id').agg(pl.col('target'), pl.col('prediction'))

        cpp_auc = CppAuc()
        auc= np.mean(
            [cpp_auc.roc_auc_score(np.array(y_t).astype(bool), np.array(y_s).astype(np.float32)) 
                for y_t, y_s in zip(evaluation_ds_grouped['target'].to_list(), 
                                    evaluation_ds_grouped['prediction'].to_list())]
        )
        logging.info(f'Evaluation results: {auc}')

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
                        help="File path where the model is placed")
    parser.add_argument("--submit", action='store_true', default=False,
                        help='Whether to save the predictions or not')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Whether to evaluate the predictions or not')
    parser.add_argument("-behaviors_path", default=None,
                        help="The file path of the reference behaviors ordering. Mandatory to save predictions")
    parser.add_argument('-batch_size', default=None, type=int, required=False,
                        help='If passed, it will predict in batches to reduce the memory usage')
    parser.add_argument('-ranker', default=False, type=bool, required=False,
                        help='Flag to specify if the model is a ranker')
    parser.add_argument('-XGBoost', default=False, type=bool, required=False,
                        help='Flag to specify if the model is a XGBoost Model')

    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    MODEL_PATH = args.model_path
    SAVE_INFERENCE = args.submit
    EVAL = args.eval
    BEHAVIORS_PATH = args.behaviors_path
    BATCH_SIZE = args.batch_size
    RANKER = args.ranker
    XGBOOST = args.XGBoost

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'Inference_Test_{timestamp}' if not EVAL else f'Inference_Validation_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)

    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w",
                        format=LOGGING_FORMATTER, level=logging.INFO, force=True)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)

    main(DATASET_DIR, MODEL_PATH, SAVE_INFERENCE,
         EVAL, BEHAVIORS_PATH, output_dir, BATCH_SIZE, RANKER, XGBOOST)
