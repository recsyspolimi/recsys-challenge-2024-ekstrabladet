import os
import logging
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from datetime import datetime
import argparse
import pandas as pd
import polars as pl
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc


LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def save_feature_importances_plot(X, y, model, output_dir, categorical_columns, group_id=None):
    train_pool = Pool(X, y, cat_features=categorical_columns, group_id=group_id)
    feature_importances = model.get_feature_importance(train_pool)
    
    sorted_importances = np.argsort(feature_importances)[::-1]
    output_path = os.path.join(output_dir, 'feature_importances.png')

    plt.figure(figsize=(10, 30))
    sns.barplot(x=feature_importances[sorted_importances], y=np.array(X.columns)[sorted_importances])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title('Catboost Feature Importances')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def main(dataset_path, catboost_params_path, output_dir, catboost_verbosity, use_ranker=False):
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")
    
    train_ds = pl.read_parquet(os.path.join(dataset_path, 'train_ds.parquet'))
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Data info: {data_info}')
    
    if use_ranker:
        train_ds = train_ds.sort(by='impression_id')
        groups = train_ds.select('impression_id').to_numpy().flatten()
        
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
    
    del train_ds
    gc.collect()
    
    logging.info(f'Features ({len(X.columns)}): {np.array(list(X.columns))}')
    logging.info(f'Categorical features: {np.array(data_info["categorical_columns"])}')
    logging.info(f'Reading catboost parameters from path: {catboost_params_path}')
    
    with open(catboost_params_path, 'r') as params_file:
        params = json.load(params_file)
        
    logging.info(f'Catboost params: {params}')
    logging.info(f'Starting to train the catboost model')
    
    if use_ranker:
        model = CatBoostRanker(**params, cat_features=data_info['categorical_columns'])
        model.fit(X, y, group_id=groups, verbose=catboost_verbosity)
    else:
        model = CatBoostClassifier(**params, cat_features=data_info['categorical_columns'])
        model.fit(X, y, verbose=catboost_verbosity, 
                  save_snapshot= True,
                  snapshot_file= os.path.join(output_dir, 'snapshot'),
                  snapshot_interval = 1800
        )
        
    logging.info(f'Model fitted. Saving the model and the feature importances at: {output_dir}')
    joblib.dump(model, os.path.join(output_dir, 'model.joblib'))
    # save_feature_importances_plot(X, y, model, output_dir, data_info['categorical_columns'])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for catboost")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the preprocessed dataset is placed")
    parser.add_argument("-catboost_params_file", default=None, type=str, required=True,
                        help="File path where the catboost hyperparameters are placed")
    parser.add_argument("-catboost_verbosity", default=50, type=int,
                        help="An integer representing how many iterations will pass between two catboost logs")
    parser.add_argument("-model_name", default=None, type=str,
                        help="An integer representing how many iterations will pass between two catboost logs")
    parser.add_argument('--ranker', action='store_true', default=False, 
                        help='Whether to use CarBoostRanker or not')
    parser.add_argument('--resume_training', action='store_true', default=False, 
                        help='Whether to resume an existing training')
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    CATBOOST_HYPERPARAMS_PATH = args.catboost_params_file
    CATBOOST_VERBOSITY = args.catboost_verbosity
    RESUME_TRAINING = args.resume_training
    USE_RANKER = args.ranker
    model_name = args.model_name
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if model_name is None:
        model_name = f'Catboost_Training_{timestamp}' if not USE_RANKER else f'CatboostRanker_Training_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    if os.path.exists(output_dir):
        if not (RESUME_TRAINING and not USE_RANKER):
            raise ValueError(f'Model with name {model_name} already exists, try another name.')
        else :
            print('Resuming Training for Catboost Classifier')
    else :
        os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_DIR, CATBOOST_HYPERPARAMS_PATH, output_dir, CATBOOST_VERBOSITY, USE_RANKER)
