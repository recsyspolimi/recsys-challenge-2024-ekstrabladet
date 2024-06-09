import os
import logging
from lightgbm import LGBMClassifier, LGBMRanker, log_evaluation
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
import gc

from polimi.preprocessing_pipelines.pre_68f import strip_new_features

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def save_feature_importances_plot(model: LGBMClassifier, feature_names: List[str], output_dir):
    feature_importances = model.feature_importances_
    
    sorted_importances = np.argsort(feature_importances)[::-1]
    output_path = os.path.join(output_dir, 'feature_importances.png')

    plt.figure(figsize=(10, 20))
    sns.barplot(x=feature_importances[sorted_importances], y=np.array(feature_names)[sorted_importances])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title('Lightgbm Feature Importances')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def main(dataset_path, lgbm_params_path, output_dir, use_ranker, early_stopping_path, es_rounds):
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")
    
    train_ds = pl.read_parquet(os.path.join(dataset_path, 'train_ds.parquet'))
    val_ds = pl.read_parquet(os.path.join(early_stopping_path, 'validation_ds.parquet')) if early_stopping_path else None
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Data info: {data_info}')
    
    if use_ranker:
        train_ds = train_ds.sort(by='impression_id')
        groups = train_ds.select(['impression_id']).to_pandas().groupby('impression_id').size()
        if early_stopping_path:
            val_ds = val_ds.sort(by='impression_id')
            eval_group = val_ds.select(['impression_id']).to_pandas().groupby('impression_id').size()
        
    if 'postcode' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('postcode').fill_null(5))
        if early_stopping_path:
            val_ds = val_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('article_type').fill_null('article_default'))
        if early_stopping_path:
            val_ds = val_ds.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in train_ds.columns:
        train_ds = train_ds.drop(['impression_time'])
    
    train_ds = train_ds.drop(['impression_id', 'article', 'user_id']).to_pandas()
    train_ds[data_info['categorical_columns']] = train_ds[data_info['categorical_columns']].astype('category')

    X = train_ds.drop(columns=['target'])
    y = train_ds['target']
    
    del train_ds
    gc.collect()
    
    if early_stopping_path:
        val_ds = val_ds.drop(['impression_id', 'article', 'user_id']).to_pandas()
        val_ds[data_info['categorical_columns']] = val_ds[data_info['categorical_columns']].astype('category')
        eval_set = (val_ds[X.columns], val_ds['target'])
        
        del val_ds
        gc.collect()
    
    logging.info(f'Features ({len(X.columns)}): {np.array(list(X.columns))}')
    logging.info(f'Categorical features: {np.array(data_info["categorical_columns"])}')
    logging.info(f'Reading lightgbm parameters from path: {lgbm_params_path}')
    
    with open(lgbm_params_path, 'r') as params_file:
        params = json.load(params_file)
        
    logging.info(f'Lightgbm params: {params}')
    logging.info(f'Starting to train the lightgbm model')
    
    verbosity = 100 if early_stopping_path else 1
    if use_ranker:
        model = LGBMRanker(**params, verbosity=verbosity, early_stopping_round=es_rounds)
        model.fit(
            X, 
            y, 
            group=groups,
            eval_set=eval_set if early_stopping_path else None, 
            eval_group=eval_group if early_stopping_path else None,
            eval_metric='ndcg' if early_stopping_path else None,
            callbacks=[log_evaluation(period=20)])
    else:
        model = LGBMClassifier(**params, verbosity=verbosity, early_stopping_round=es_rounds)
        model.fit(
            X, 
            y,
            eval_set=eval_set if early_stopping_path else None,
            eval_metric='auc' if early_stopping_path else None,
            callbacks=[log_evaluation(period=20)])
        
    if early_stopping_path:
        logging.info(f'Best iteration: {model.best_iteration_}')
        
    logging.info(f'Model fitted. Saving the model and the feature importances at: {output_dir}')
    joblib.dump(model, os.path.join(output_dir, 'model.joblib'))
    save_feature_importances_plot(model, X.columns, output_dir)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for catboost")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the preprocessed dataset is placed")
    parser.add_argument("-lgbm_params_file", default=None, type=str, required=True,
                        help="File path where the catboost hyperparameters are placed")
    parser.add_argument("-model_name", default=None, type=str,
                        help="The name of the model")
    parser.add_argument('--ranker', action='store_true', default=False, 
                        help='Whether to use LGBMRanker or not')
    parser.add_argument("-early_stopping_path", default=None, type=str,
                        help="Directory where the early stopping dataset is placed")
    parser.add_argument("-early_stopping_rounds", default=50, type=int,
                        help="The patience for early stopping")
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    LGBM_HYPERPARAMS_PATH = args.lgbm_params_file
    EARLY_STOPPING_PATH = args.early_stopping_path
    EARLY_STOPPING_ROUNDS = args.early_stopping_rounds if EARLY_STOPPING_PATH else None
    USE_RANKER = args.ranker
    model_name = args.model_name
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if model_name is None:
        model_name = f'LGBM_Training_{timestamp}' if not USE_RANKER else f'LGBMRanker_Training_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_DIR, LGBM_HYPERPARAMS_PATH, output_dir, USE_RANKER, EARLY_STOPPING_PATH, EARLY_STOPPING_ROUNDS)
