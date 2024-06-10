import os
import logging
import lightgbm as lgb
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


def save_feature_importances_plot(model: lgb.Booster, feature_names: List[str], output_dir):
    feature_importances = model.feature_importance(importance_type='gain')
    
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
    
    train_ds = lgb.Dataset(data=dataset_path)
    
    logging.info(f'Reading lightgbm parameters from path: {lgbm_params_path}')
    
    with open(lgbm_params_path, 'r') as params_file:
        params = json.load(params_file)
        
    logging.info(f'Lightgbm params: {params}')
    logging.info(f'Starting to train the lightgbm model')
        
    num_boost_rounds = params['n_estimators']
    del params['n_estimators']
    
    params['verbosity'] = 100
    params['objective'] = 'lambdarank' if use_ranker else 'binary'
    
    booster = lgb.train(params=params, num_boost_round=num_boost_rounds, train_set=train_ds)
    booster.save_model(os.path.join(output_dir, 'booster.txt'))
    
    if use_ranker:
        model = lgb.LGBMRanker()
        model.booster_ = booster
        model.feature_name_ = booster.feature_name()
        model.n_features_in_ = booster.num_feature()
    else:
        model = lgb.LGBMClassifier()
        model.booster_ = booster
        model.feature_name_ = booster.feature_name()
        model.n_features_in_ = booster.num_feature()
        model.n_classes_ = 2
        
    logging.info(f'Model fitted. Saving the model and the feature importances at: {output_dir}')
    joblib.dump(model, os.path.join(output_dir, 'model.joblib'))
    save_feature_importances_plot(booster, booster.feature_name(), output_dir)
    
    
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
