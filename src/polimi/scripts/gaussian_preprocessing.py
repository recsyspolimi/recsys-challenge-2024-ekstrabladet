import polars as pl
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import os
import logging
from datetime import datetime
import argparse
import pandas as pd
import json
import numpy as np
import seaborn as sns
import gc
import tqdm
import joblib

from polimi.utils.tf_models import *
from polimi.utils.tf_models.utils import *


LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"

def main(dataset_path, numerical_transform, dataset_type, fit, load_path, output_dir):
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")
    
    dataset = pl.read_parquet(os.path.join(dataset_path, f'{dataset_type}_ds.parquet'))
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Data info: {data_info}')
    
    categorical_columns = data_info['categorical_columns']
    ignore_columns = ['impression_id', 'article', 'user_id', 'impression_time', 'target']
    numerical_columns = [c for c in dataset.columns if c not in categorical_columns + ignore_columns]
    
    logging.info(f'Features ({len(dataset.columns)}): {np.array(list(dataset.columns))}')
    logging.info(f'Categorical features: {np.array(data_info["categorical_columns"])}')
    
    if fit:
        if numerical_transform == 'quantile-normal':
            xformer = QuantileTransformer(output_distribution='normal')
        elif numerical_transform == 'yeo-johnson':
            xformer = PowerTransformer(method='yeo-johnson')
        else:
            raise ValueError('Not recognized numerical transformer')
        X_train_numerical = xformer.fit_transform(
            dataset.select(numerical_columns).to_pandas().replace(
                [-np.inf, np.inf], np.nan).astype(np.float32)).astype(np.float32)
    else:
        xformer = joblib.load(load_path)
        missing_columns = [x for x in numerical_columns if x not in xformer.feature_names_in_]
        if len(missing_columns):
            raise ValueError(f'These columns are missing in the dataset: {missing_columns}')
        X_train_numerical = xformer.transform(
            dataset.select(xformer.feature_names_in_).to_pandas().replace(
                [-np.inf, np.inf], np.nan).astype(np.float32)).astype(np.float32)
    
    logging.info('Preprocessing complete.')
    for i, col in tqdm.tqdm(enumerate(numerical_columns)):
        dataset = dataset.with_columns(pl.Series(X_train_numerical[:, i]).fill_nan(0).fill_null(0).alias(col))
        
    logging.info(f'Saving dataset and numerical transformer at: {output_dir}')
    data_info_path = os.path.join(output_dir, 'data_info.json')
    with open(data_info_path, 'w') as data_info_file:
        json.dump(data_info, data_info_file)
        
    joblib.dump(xformer, os.path.join(output_dir, 'numerical_transformer.joblib'))
    dataset.write_parquet(os.path.join(output_dir, f'{dataset_type}_ds.parquet'))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for neural networks")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the dataset will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the preprocessed dataset is placed")
    parser.add_argument("-dataset_type", choices=['train', 'validation', 'test'], default='train', type=str,
                        help="Specify the type of dataset: ['train', 'validation', 'test']")
    parser.add_argument("-numerical_transform", type=str, required=True,
                        choices=['yeo-johnson', 'quantile-normal'],
                        help="The type of numerical transformer to be used")
    parser.add_argument('--fit', action='store_true', default=False, 
                        help='Whether to fit the preprocessor or not')
    parser.add_argument("-load_path", default=None, type=str, required=False,
                        help="Directory where the numerical transformer is placed")
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_TYPE = args.dataset_type
    DATASET_DIR = args.dataset_path
    NUMERICAL_TRANSFORM = args.numerical_transform
    LOAD_PATH = args.load_path
    FIT = args.fit
    
    if not FIT and LOAD_PATH is None:
        raise ValueError('Fit is set to false but load path is not provided')
    elif FIT and NUMERICAL_TRANSFORM is None:
        raise ValueError('Fit is set to true but no numerical transform is provided')
    
    if FIT and LOAD_PATH is not None:
        logging.warn('Load path will be ignored since fit is set to true')
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f'preprocessing_{NUMERICAL_TRANSFORM}_{DATASET_TYPE}_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, dir_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_DIR, NUMERICAL_TRANSFORM, DATASET_TYPE, FIT, LOAD_PATH, output_dir)
