import os
import logging
from xgboost import XGBClassifier, XGBRanker, callback
from datetime import datetime
import argparse
import pandas as pd
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing_extensions import List
import polars as pl

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def save_feature_importances_plot(model: XGBClassifier | XGBRanker, feature_names: List[str], output_dir):
    feature_importances = model.feature_importances_
    
    sorted_importances = np.argsort(feature_importances)[::-1]
    output_path = os.path.join(output_dir, 'feature_importances.png')

    plt.figure(figsize=(10, 20))
    sns.barplot(x=feature_importances[sorted_importances], y=np.array(feature_names)[sorted_importances])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title(f"XGB Feature Importances")

    # plt.title(f"XGB {'_ranker' if model == XGBRanker else '_cls'} Feature Importances")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
        
class CustomLogger:
    def __init__(self, logger_name='xgb_custom_logger', log_file=None):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(LOGGING_FORMATTER)
        
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
    
    def write(self, msg):
        if msg.strip():
            self.logger.debug(msg.strip())
    
    def info(self, msg:str):
        return self.logger.info(msg)
    def debug(self, msg:str):
        return self.logger.debug(msg)
    
    def flush(self):
        pass


def main(dataset_path, params_path, output_dir, use_ranker, verbosity):
    log_file_path = os.path.join(output_dir, 'log.txt')
    logging = CustomLogger(log_file=log_file_path)
    
    
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")
    
    file_path = os.path.join(dataset_path, 'train_ds.parquet')
    logging.info(f"Reading the dataset from {file_path}")
    train_ds = pl.read_parquet(file_path)
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Data info: {data_info}')
    
    if use_ranker:
        train_ds = train_ds.sort(by='impression_id')
        groups = train_ds.select(['impression_id']).to_pandas().groupby('impression_id').size()
        
    if 'postcode' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in train_ds.columns:
        train_ds = train_ds.drop(['impression_time'])
    
    train_ds = train_ds.drop(['impression_id', 'article', 'user_id']).to_pandas()
    train_ds[data_info['categorical_columns']] = train_ds[data_info['categorical_columns']].astype('category')

    X = train_ds.drop(columns=['target'])
    y = train_ds['target']
    
    
    
    logging.info(f'Features ({len(X.columns)}): {np.array(list(X.columns))}')
    logging.info(f'Categorical features: {np.array(data_info["categorical_columns"])}')
    logging.info(f'Reading xgb parameters from path: {params_path}')
    
    with open(params_path, 'r') as params_file:
        params = json.load(params_file)
        
    logging.info(f'XGB params: {params}')
    logging.info(f"Starting to train the XGB {'ranker' if use_ranker else 'cls'} model, verbosity: {verbosity}")
       
    import sys
    sys.stdout = logging
    sys.stderr = logging
    
    verbose=1
    if use_ranker:
        model = XGBRanker(**params, enable_categorical=True, verbosity=verbosity)
        model.fit(X, y, group=groups, verbose=verbose, eval_set=[(X.iloc[0:1], y.iloc[0:1])], eval_metric='error')
    else:
        model = XGBClassifier(**params, verbosity=verbosity, enable_categorical=True)
        model.fit(X, y, verbose=verbose, eval_set=[(X.iloc[0:1], y.iloc[0:1])], eval_metric='error')
    
    # Ripristina stdout originale
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
        
    logging.info(f'Model fitted. Saving the model and the feature importances at: {output_dir}')
    joblib.dump(model, os.path.join(output_dir, 'model.joblib'))
    save_feature_importances_plot(model, X.columns, output_dir)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for XGB")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the preprocessed dataset is placed")
    parser.add_argument("-params_file", default=None, type=str, required=True,
                        help="File path where the XGB hyperparameters are placed")
    parser.add_argument("-model_name", default=None, type=str,
                        help="The name of the model")
    parser.add_argument('--ranker', action='store_true', default=False, 
                        help='Whether to use XGBRanker or not')
    parser.add_argument('-verbosity', choices=['0', '1', '2', '3'], default='0', 
                        help='XGB verbosity')
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    HYPERPARAMS_PATH = args.params_file
    USE_RANKER = args.ranker
    model_name = args.model_name
    VERBOSITY = int(args.verbosity)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if model_name is None:
        model_name = f'XGB_Training_{timestamp}' if not USE_RANKER else f'XGBRanker_Training_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    
    main(DATASET_DIR, HYPERPARAMS_PATH, output_dir, USE_RANKER, VERBOSITY)
