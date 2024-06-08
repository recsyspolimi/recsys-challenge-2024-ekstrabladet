import os
import logging
from matplotlib.path import Path
from xgboost import XGBClassifier, XGBRanker, callback
import xgboost as xgb
from datetime import datetime
from polimi.utils._polars import check_for_inf, reduce_polars_df_memory_size
import argparse
import pandas as pd
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing_extensions import List
import polars as pl
import gc
import polars.selectors as cs

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
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    
def main(dataset_path, params_path, output_dir, verbosity):        
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")
    
    file_path = os.path.join(dataset_path, 'train_ds.parquet')
    logging.info(f"Reading the dataset from {file_path}")
    train_ds = pl.read_parquet(file_path)
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Data info: {data_info}')
    
        
    if 'postcode' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in train_ds.columns:
        train_ds = train_ds.drop(['impression_time'])
        
    logging.info('Checking for inf in train_ds...')
    rows_with_inf, cols_with_inf = check_for_inf(train_ds)
    logging.info(f'Rows with inf: {rows_with_inf}')
    logging.info(f'Columns with inf: {cols_with_inf}')
    
    logging.info('Replacing inf from train_ds...')
    train_ds = train_ds.with_columns(pl.when(~(cs.numeric().is_infinite())).then(cs.numeric()))
    
    train_ds = reduce_polars_df_memory_size(train_ds)

    train_ds = train_ds.drop(['impression_id', 'article', 'user_id']).to_pandas()
    train_ds[data_info['categorical_columns']] = train_ds[data_info['categorical_columns']].astype('category')        
    X = train_ds.drop(columns=['target'])
    y = train_ds['target']
    del train_ds
    gc.collect()
    
    with open(params_path, 'r') as params_file:
        params = json.load(params_file)
        
    logging.info(f'XGB params: {params}')
    
    dtrain = xgb.QuantileDMatrix(X, label=y, enable_categorical=True, max_bin=params['max_bin'])
    logging.info('finishing building the QuantileDMatrix')
    dtrain_evals = xgb.DMatrix(X.iloc[0:1], label=y.iloc[0:1], enable_categorical=True)
    
    booster_params = params.copy()
    num_boost_round = params['n_estimators']
    booster_params['eval_metric'] = 'error' 
    booster_params['objective'] = 'binary:logistic'
    del booster_params['n_estimators']
    
    del X, y
    gc.collect()
    
    logging.info(f'Features ({len(dtrain.feature_names)}): {np.array(list(dtrain.feature_names))}')
    logging.info(f'Categorical features: {np.array(data_info["categorical_columns"])}')
    logging.info(f'Reading xgb parameters from path: {params_path}')
    logging.info(f"Starting to train the XGB cls model, verbosity: {verbosity}")
          
    ##TODO: implement ranker 
    booster = xgb.train(booster_params, dtrain, num_boost_round=num_boost_round, evals=[(dtrain_evals, 'train')], verbose_eval=1)
    model = XGBClassifier()
    model._Booster = booster
    model.set_params(**booster.attributes())
    model.n_classes_ = None
        
    logging.info(f'Model fitted. Saving the model and the feature importances at: {output_dir}')
    joblib.dump(model, os.path.join(output_dir, 'model.joblib'))
    save_feature_importances_plot(model, dtrain.feature_names, output_dir)
    
    
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
    parser.add_argument('-verbosity', choices=['0', '1', '2', '3'], default='0', 
                        help='XGB verbosity')

    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    HYPERPARAMS_PATH = args.params_file
    USE_RANKER = args.ranker
    model_name = args.model_name
    VERBOSITY = int(args.verbosity)
    DMATRIX_PATH = Path(args.dmatrix) if args.dmatrix_path is not None else None
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if model_name is None:
        model_name = f'XGB_Training_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_DIR, HYPERPARAMS_PATH, output_dir, VERBOSITY)
        
