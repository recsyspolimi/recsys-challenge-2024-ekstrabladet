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
sys.path.append('/media/disk1/recsys-challenge-2024/RecSysChallenge2024/src')

from polimi.utils._catboost import build_features_iterator


LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def main(input_path, output_dir, dataset_type='train'):
    logging.info("Starting to build the training dataset")
    logging.info(f"Dataset path: {input_path}")
    
    articles = pl.read_parquet(os.path.join(input_path, 'articles.parquet'))
    files_path = os.path.join(input_path, dataset_type)
    behaviors = pl.read_parquet(os.path.join(files_path, 'behaviors.parquet'))
    history = pl.read_parquet(os.path.join(files_path, 'history.parquet'))
    
    logging.info('Finished to build parquet files. Starting feature engineering')
    
    is_test_data = dataset_type == 'test'
    sample = dataset_type == 'train'
    
    dataset_complete = []
    i = 0
    for dataset, vectorizer, unique_entities in build_features_iterator(behaviors, history, articles, 
                                                                        test=is_test_data, sample=sample):
        dataset_complete.append(dataset)
        logging.info(f'Slice {i+1} preprocessed.')
        i += 1
    dataset_complete = pl.concat(dataset_complete)
    
    categorical_columns = ['device_type', 'is_sso_user', 'gender', 'is_subscriber', 'weekday',
                           'premium', 'category', 'sentiment_label', 'is_new_article', 'is_already_seen_article',
                           'MostFrequentCategory', 'MostFrequentWeekday', 'IsFavouriteCategory']
    categorical_columns += [f'Entity_{entity}_Present' for entity in unique_entities]
    
    dataset_complete.write_parquet(os.path.join(output_dir, f'{dataset_type}_ds.parquet'))
    
    logging.info(f'Preprocessing complete. There are {len(dataset.columns)} columns: {np.array(dataset.columns)}')
    vectorizer_path = os.path.join(output_dir, 'tf_idf_vectorizer.joblib')
    logging.info(f'Saving the tf-idf vectorizer at: {vectorizer_path}')
    joblib.dump(vectorizer, vectorizer_path)
    
    dataset_info = {
        'type': dataset_type,
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
    parser.add_argument("-dataset_type", choices=['train', 'validation', 'test'], default='train', type=str,
                        help="Specify the type of dataset: ['train', 'validation', 'test']")
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    DATASET_TYPE = args.dataset_type
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f'preprocessing_{DATASET_TYPE}_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, experiment_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_DIR, output_dir, DATASET_TYPE)