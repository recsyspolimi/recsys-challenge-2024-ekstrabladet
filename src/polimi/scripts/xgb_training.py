import os
import logging
import shutil
from typing import Callable
from pathlib import Path
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
import math

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
    
    
def process_train_ds(train_ds: pd.DataFrame, data_info: dict):
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
    return X, y

def create_slices(dataset_path: Path, data_info: dict, n_slices: int = 3):
    logging.info(f"Reading the dataset from {dataset_path / 'train_ds.parquet'}")
    train_ds = pl.read_parquet(dataset_path / 'train_ds.parquet')
    
    slices_path = dataset_path / 'slices'
    if slices_path.exists() and slices_path.is_dir():
        logging.info(f'Removing existing directory {slices_path}...')
        shutil.rmtree(slices_path)
    
    BATCH_SIZE = math.ceil(len(train_ds) / n_slices)
    for i, slice in enumerate(train_ds.iter_slices(BATCH_SIZE)):
        dir_path = slices_path / str(i)
        dir_path.mkdir(exist_ok=True, parents=True)
        logging.info(f'Processing slice {i}...')
        X_i, y_i = process_train_ds(slice, data_info)
        logging.info(f'Saving slice {i} X.parquet...')
        X_i.to_parquet(dir_path / 'X.parquet')
        logging.info(f'Saving slice {i} y.parquet...')
        y_i.to_frame().to_parquet(dir_path / 'y.parquet')
            
class Iterator(xgb.DataIter):
    def __init__(self, file_paths: List[str], data_info: dict):
        self._file_paths = file_paths
        self.data_info = data_info
        self._it = 0
        super().__init__()


    def next(self, input_data: Callable):
        """Advance the iterator by 1 step and pass the data to XGBoost.  This function is
        called by XGBoost during the construction of ``DMatrix``

        """
        if self._it == len(self._file_paths):
            # return 0 to let XGBoost know this is the end of iteration
            return 0

        # input_data is a function passed in by XGBoost who has the exact same signature of
        # ``DMatrix``
        logging.info(f"Reading the dataset from {self._file_paths[self._it]}")
        X = pd.read_parquet(self._file_paths[self._it] / 'X.parquet')
        y = pd.read_parquet(self._file_paths[self._it] / 'y.parquet')['target']
        input_data(data=X, label=y)
        self._it += 1
        # Return 1 to let XGBoost know we haven't seen all the files yet.
        return 1

    def reset(self):
        """Reset the iterator to its beginning"""
        self._it = 0
    
    def get_eval(self):
        X = pd.read_parquet(self._file_paths[self._it] / 'X.parquet').iloc[0:1]
        y = pd.read_parquet(self._file_paths[self._it] / 'y.parquet')['target'].iloc[0:1]
        return xgb.DMatrix(X, label=y, enable_categorical=True)
        
        
    
def main(dataset_path: Path, params_path: Path, output_dir: Path, verbosity, preprocess_slices:bool = True):        
    with open(dataset_path / 'data_info.json') as data_info_file:
        data_info = json.load(data_info_file)        
    logging.info(f'Data info: {data_info}')
    
    if preprocess_slices:
        create_slices(dataset_path, data_info, n_slices=3)
        
    slices_path = dataset_path / 'slices'
    # List all folders of type dataset_path / 'slices' / i
    folder_paths = sorted(
        [p for p in slices_path.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda x: int(x.name)
    )
        
    logging.info(f"Reading the dataset from {folder_paths}")
    it = Iterator(folder_paths, data_info)
    deval = it.get_eval()

    with open(params_path, 'r') as params_file:
        params = json.load(params_file)
    logging.info(f'XGB params: {params}')
    
    dtrain = xgb.QuantileDMatrix(it, enable_categorical=True, max_bin=params['max_bin'])
    logging.info('Finished building the QuantileDMatrix')
    
    booster_params = params.copy()
    num_boost_round = params['n_estimators']
    booster_params['eval_metric'] = 'error' 
    booster_params['objective'] = 'binary:logistic'
    del booster_params['n_estimators']
    
    logging.info(f'Features ({len(dtrain.feature_names)}): {np.array(list(dtrain.feature_names))}')
    logging.info(f'Categorical features: {np.array(data_info["categorical_columns"])}')
    logging.info(f'Reading xgb parameters from path: {params_path}')
    logging.info(f"Starting to train the XGB cls model, verbosity: {verbosity}")
          
    ##TODO: implement ranker 
    booster = xgb.train(booster_params, dtrain, num_boost_round=num_boost_round, evals=[(deval, 'train')], verbose_eval=1)
    model = XGBClassifier()
    model._Booster = booster
    model.set_params(**booster.attributes())
    model.n_classes_ = None
        
    logging.info(f'Model fitted. Saving the model and the feature importances at: {output_dir}')
    joblib.dump(model, output_dir / 'model.joblib')
    save_feature_importances_plot(model, dtrain.feature_names, output_dir)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for XGB")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the preprocessed dataset is placed")
    parser.add_argument("-params_file", default=None, type=str, required=True,
                        help="File path where the XGB hyperparameters are placed")
    parser.add_argument('-verbosity', choices=['0', '1', '2', '3'], default='0', 
                        help='XGB verbosity')
    
    
    args = parser.parse_args()
    OUTPUT_DIR = Path(args.output_dir)
    DATASET_DIR = Path(args.dataset_path)
    HYPERPARAMS_PATH = Path(args.params_file)
    VERBOSITY = int(args.verbosity)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = OUTPUT_DIR / f'XGB_Training_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = output_dir / "log.txt"
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_DIR, HYPERPARAMS_PATH, output_dir, VERBOSITY)
        
