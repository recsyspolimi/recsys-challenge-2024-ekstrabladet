
import os
import logging
from datetime import datetime
import argparse
from pathlib import Path
import polars as pl
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time

import sys
sys.path.append('/home/ubuntu/RecSysChallenge2024/src')

from polimi.preprocessing_pipelines.preprocessing_versions import BATCH_PREPROCESSING, PREPROCESSING
from polimi.preprocessing_pipelines.categorical_dict import get_categorical_columns
from polimi.utils._strategies import _behaviors_to_history, moving_window_split_iterator

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def main(input_path, output_dir, preprocessing_version='latest'):
    logging.info(f"Preprocessing version: ----{preprocessing_version}----")
    logging.info("Starting to build the dataset")
    logging.info(f"Dataset path: {input_path}")

    articles = pl.read_parquet(os.path.join(input_path, 'articles.parquet'))
    train_path = os.path.join(input_path, 'train')
    validation_path = os.path.join(input_path, 'validation')
    behaviors_train = pl.read_parquet(os.path.join(train_path, 'behaviors.parquet'))
    history_train = pl.read_parquet(os.path.join(train_path, 'history.parquet'))
    behaviors_val = pl.read_parquet(os.path.join(validation_path, 'behaviors.parquet'))
    history_val = pl.read_parquet(os.path.join(validation_path, 'history.parquet'))
    
    history_all = pl.concat([
    history_train.explode(pl.all().exclude('user_id')).join(
            history_val.explode(pl.all().exclude('user_id')), 
            on=['user_id', 'impression_time_fixed'], how='anti'),
        history_val.explode(pl.all().exclude('user_id')),
        _behaviors_to_history(behaviors_val).explode(pl.all().exclude('user_id')),
    ]).sort(['user_id', 'impression_time_fixed'])\
    .group_by('user_id').agg(pl.all())
    
    del history_train, history_val
    gc.collect()
    
    behaviors_all = pl.concat([
        behaviors_train,
        behaviors_val
    ]).sort('impression_time')
    del behaviors_train, behaviors_val
    gc.collect()
    
    urm_ner_scores_path=None
    emb_scores_path=None
    dataset_path=Path(input_path).parent
    output_dir = Path(output_dir)

    logging.info(
        'Finished to build parquet files. Starting feature engineering')
    start_time = time.time()
    for i, (history_k_train, behaviors_k_train, history_k_val, behaviors_k_val) in enumerate(
        moving_window_split_iterator(history_all, behaviors_all, window=4, window_val=2, stride=2, verbose=True)
    ):
        logging.info(f'Preprocessing fold {i}')
        if i != 1:
            continue
           
        fold_path = output_dir / f'fold_{i+1}'
        fold_path.mkdir(parents=True, exist_ok=True)
                
        logging.info(f'Starting training fold {i}...')
        features_k_train, _, unique_entities = PREPROCESSING[preprocessing_version](
            behaviors_k_train, history_k_train, articles, test=False, sample=False, previous_version=None,
            split_type='train', output_path=output_dir, emb_scores_path=emb_scores_path, 
            urm_ner_scores_path=urm_ner_scores_path, dataset_path=dataset_path)
        features_k_train.write_parquet(fold_path / 'train_ds.parquet')
        del features_k_train
        gc.collect()
        
        logging.info(f'Starting validation fold {i}...')
        features_k_val, _, unique_entities = PREPROCESSING[preprocessing_version](
            behaviors_k_val, history_k_val, articles, test=False, sample=False, previous_version=None,
            split_type='validation', output_path=output_dir, emb_scores_path=emb_scores_path, 
            urm_ner_scores_path=urm_ner_scores_path, dataset_path=dataset_path)
        
        features_k_val.write_parquet(fold_path / 'validation_ds.parquet')
        del features_k_val
        gc.collect()

        
    categorical_columns = get_categorical_columns(preprocessing_version)
    categorical_columns += [f'Entity_{entity}_Present' for entity in unique_entities]

    logging.info(f'Preprocessing moving window finished. {i + 1} folds completed in {((time.time() - start_time) / 60):.2f} minutes.')

    dataset_info = {
        'type': f'mw_w4_wval2_st2_kfold',
        'categorical_columns': categorical_columns,
        'unique_entities': unique_entities,
        'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }
    data_info_path = os.path.join(output_dir, 'data_info.json')
    with open(data_info_path, 'w') as data_info_file:
        json.dump(dataset_info, data_info_file)
    logging.info(f'Saved data info at: {data_info_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing moving window k fold")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the dataset is placed")
    parser.add_argument("-preprocessing_version", default='latest', type=str,
                        choices=['68f', '94f', '115f', '127f', '142f', '147f', 'new', 'new_emb_urm', 'latest'],
                        help="Specifiy the preprocessing version to use. Default is 'latest' valuses are ['68f', '94f', '115f','latest']")
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    PREPROCESSING_VERSION = args.preprocessing_version

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f'preprocessing_moving_window_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, experiment_name)
    os.makedirs(output_dir)

    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w",
                        format=LOGGING_FORMATTER, level=logging.INFO, force=True)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)

    main(DATASET_DIR, output_dir, PREPROCESSING_VERSION)
