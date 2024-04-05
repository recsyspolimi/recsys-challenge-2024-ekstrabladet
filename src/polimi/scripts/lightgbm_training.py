import os
import logging
from lightgbm import LGBMClassifier
from datetime import datetime
import argparse
import pandas as pd
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing_extensions import List


LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def save_feature_importances_plot(model: LGBMClassifier, feature_names: List[str], output_dir):
    feature_importances = model.feature_importances_
    
    sorted_importances = np.argsort(feature_importances)[::-1]
    output_path = os.path.join(output_dir, 'feature_importances.png')

    plt.figure(figsize=(10, 10))
    sns.barplot(x=feature_importances[sorted_importances], y=np.array(feature_names)[sorted_importances])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title('Lightgbm Feature Importances')
    plt.savefig(output_path)
    plt.close()


def main(dataset_path, lgbm_params_path, output_dir):
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")
    
    train_ds = pd.read_parquet(os.path.join(dataset_path, 'train_ds.parquet'))
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Data info: {data_info}')
    
    train_ds = train_ds.drop(columns=['impression_id', 'article', 'user_id', 'article_id'])
    train_ds[data_info['categorical_columns']] = train_ds[data_info['categorical_columns']].astype('category')

    X = train_ds.drop(columns=['target'])
    y = train_ds['target']
    
    logging.info(f'Features ({len(X.columns)}): {np.array(list(X.columns))}')
    logging.info(f'Categorical features: {np.array(data_info["categorical_columns"])}')
    logging.info(f'Reading lightgbm parameters from path: {lgbm_params_path}')
    
    with open(lgbm_params_path, 'r') as params_file:
        params = json.load(params_file)
        
    logging.info(f'Lightgbm params: {params}')
    logging.info(f'Starting to train the lightgbm model')
    model = LGBMClassifier(**params)
    model.fit(X, y)
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
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    LGBM_HYPERPARAMS_PATH = args.lgbm_params_file
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'Lightgbm_Training_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_DIR, LGBM_HYPERPARAMS_PATH, output_dir)
