import os
import logging
from datetime import datetime
import argparse
import polars as pl
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('/home/ubuntu/RecSysChallenge2024/src')

from polimi.utils._catboost import build_features


LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def main(input_path, output_dir, preprocessing_path):
    logging.info("Starting to build the dataset")
    logging.info(f"Dataset path: {input_path}")
    
    articles = pl.read_parquet(os.path.join(input_path, 'articles.parquet'))
    files_path = os.path.join(input_path, 'test')
    behaviors = pl.read_parquet(os.path.join(files_path, 'behaviors.parquet')).filter(pl.col('impression_id') == 0)
    history = pl.read_parquet(os.path.join(files_path, 'history.parquet'))
    
    logging.info('Finished to build parquet files. Starting feature engineering')
    
    previous_test_ds = pl.read_parquet(preprocessing_path).filter(pl.col('impression_id') != 0)
    
    dataset, tf_idf_vectorizer, unique_entities = build_features(behaviors, history, articles, test=True, sample=False)
    dataset = pl.concat([dataset.select(previous_test_ds.columns), previous_test_ds], how='vertical_relaxed')
    
    categorical_columns = ['device_type', 'is_sso_user', 'gender', 'is_subscriber', 'weekday',
                           'premium', 'category', 'sentiment_label', 'is_new_article', 'is_already_seen_article',
                           'MostFrequentCategory', 'MostFrequentWeekday', 'IsFavouriteCategory']
    categorical_columns += [f'Entity_{entity}_Present' for entity in unique_entities]
    
    dataset.write_parquet(os.path.join(output_dir, 'test_ds.parquet'))
    
    logging.info(f'Preprocessing complete. There are {len(dataset.columns)} columns: {np.array(dataset.columns)}')
    vectorizer_path = os.path.join(output_dir, 'tf_idf_vectorizer.joblib')
    logging.info(f'Saving the tf-idf vectorizer at: {vectorizer_path}')
    joblib.dump(tf_idf_vectorizer, vectorizer_path)
    
    dataset_info = {
        'type': 'test',
        'categorical_columns': categorical_columns,
        'unique_entities': unique_entities,
        'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }
    data_info_path = os.path.join(output_dir, 'data_info.json')
    with open(data_info_path, 'w') as data_info_file:
        json.dump(dataset_info, data_info_file)
    logging.info(f'Saved data info at: {data_info_path}')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for catboost")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the dataset is placed")
    parser.add_argument("-preprocessing_path", default=None, type=str, required=True,
                        help="Directory where the preprocessed non beyond accuracy samples are placed")
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    PREPROCESSING_DIR = args.preprocessing_path
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f'preprocessing_tests_beyond_acc_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, experiment_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_DIR, output_dir, PREPROCESSING_DIR)