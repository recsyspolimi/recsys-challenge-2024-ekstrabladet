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

from polimi.preprocessing_pipelines.categorical_dict import get_categorical_columns
from polimi.preprocessing_pipelines.preprocessing_versions import BATCH_PREPROCESSING, TEST_BATCH_PREPROCESSING

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def main(input_path, output_dir, dataset_type='train',preprocessing_version='latest',previous_version = None):
    logging.info(f"Preprocessing version: ----{preprocessing_version}----")
    logging.info("Starting to build the dataset")
    logging.info(f"Dataset path: {input_path}")
    
    articles = pl.read_parquet(os.path.join(input_path, 'articles.parquet'))
    files_path = os.path.join(input_path, dataset_type)
    behaviors = pl.read_parquet(os.path.join(files_path, 'behaviors.parquet'))
    history = pl.read_parquet(os.path.join(files_path, 'history.parquet'))
    slices_dir = output_dir + '/Sliced_ds'
    os.makedirs(slices_dir)
    logging.info('Finished to build parquet files. Starting feature engineering')
    
    is_test_data = dataset_type == 'test'
    #sample = dataset_type == 'train'
    sample = False
    
    if is_test_data and previous_version is not None:
        build_features_iterator = TEST_BATCH_PREPROCESSING[preprocessing_version]
    else:    
        build_features_iterator = BATCH_PREPROCESSING[preprocessing_version]
    
    dataset_complete = []
    i = 0
    for dataset, vectorizer, unique_entities in build_features_iterator(behaviors, history, articles, test=is_test_data, 
                                                                        sample=sample, n_batches=100 ,previous_version = previous_version):
        dataset.write_parquet(os.path.join(slices_dir, f'{dataset_type}_slice_{i}.parquet'))
        if not is_test_data:
            dataset_complete.append(dataset)
        logging.info(f'Slice {i+1} preprocessed.')
        i += 1
    
    if not is_test_data:
        dataset_complete = pl.concat(dataset_complete, how='vertical_relaxed')
    
    categorical_columns = get_categorical_columns(preprocessing_version)
    
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
    parser.add_argument("-preprocessing_version", choices=['68f', '94f', '115f','127f', 'latest'], default='latest', type=str,
                        help="Specifiy the preprocessing version to use. Default is 'latest' valuses are ['68f', '94f', '115f','127f','latest']")
    parser.add_argument("-previous_version", default = None, type=str,
                        help="Specify the path of a previous version of the dataset to use as a reference for the new one. Default is None.\n YOU MUST GUARANTEE THE COMPATIBILITY BETWEEN VERSIONS. ")
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    DATASET_TYPE = args.dataset_type
    PREPROCESSING_VERSION = args.preprocessing_version
    PREVIOUS_VERSION = args.previous_version
    
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
    
    main(DATASET_DIR, output_dir, DATASET_TYPE, PREPROCESSING_VERSION, PREVIOUS_VERSION)