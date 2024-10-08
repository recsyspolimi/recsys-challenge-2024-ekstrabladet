
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
from sklearn.model_selection import GroupKFold

import sys
sys.path.append('/home/ubuntu/RecSysChallenge2024/src')

from polimi.preprocessing_pipelines.preprocessing_versions import PREPROCESSING
from polimi.preprocessing_pipelines.categorical_dict import get_categorical_columns

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def main(input_path, output_dir, preprocessing_version='latest', n_folds=5,
         urm_path=None, recsys_urm_path=None, recsys_models_path=None, ners_models_path=None):
    logging.info(f"Preprocessing version: ----{preprocessing_version}----")
    logging.info("Starting to build the dataset")
    logging.info(f"Dataset path: {input_path}")

    articles = pl.read_parquet(os.path.join(input_path, 'articles.parquet'))
    train_path = os.path.join(input_path, 'train')
    validation_path = os.path.join(input_path, 'validation')
    behaviors_train = pl.read_parquet(os.path.join(train_path, 'behaviors.parquet'))
    history_train = pl.read_parquet(os.path.join(train_path, 'history.parquet'))
    behaviors_val = pl.read_parquet(os.path.join(validation_path, 'behaviors.parquet'))
    history_val = pl.read_parquet(os.path.join(validation_path, 'history.parquet'))

    logging.info(
        'Finished to build parquet files. Starting feature engineering')
    
    # even if users have the same id in the train and val, they are considered two "different" users
    # since a different history is used for them
    behaviors_train = behaviors_train.with_columns(
        pl.concat_str([pl.col('user_id').cast(pl.String), pl.lit('1')], separator='_').alias('user_id'),
        pl.concat_str([pl.col('impression_id').cast(pl.String), pl.lit('1')], separator='_').alias('impression_id'),
    )
    history_train = history_train.with_columns(
        pl.concat_str([pl.col('user_id').cast(pl.String), pl.lit('1')], separator='_').alias('user_id')
    )
    behaviors_val = behaviors_val.with_columns(
        pl.concat_str([pl.col('user_id').cast(pl.String), pl.lit('2')], separator='_').alias('user_id'),
        pl.concat_str([pl.col('impression_id').cast(pl.String), pl.lit('2')], separator='_').alias('impression_id'),
    )
    history_val = history_val.with_columns(
        pl.concat_str([pl.col('user_id').cast(pl.String), pl.lit('2')], separator='_').alias('user_id')
    )
    user_info = pl.concat([behaviors_train.select(['user_id', 'impression_id']), 
                           behaviors_val.select(['user_id', 'impression_id'])]).to_pandas()
    
    for i, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=n_folds).split(user_info, groups=user_info['user_id'])):
        logging.info(f'Preprocessing fold {i}')
        fold_path = os.path.join(output_dir, f'fold_{i+1}')
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
        
        user_info_train = pl.from_pandas(user_info.iloc[train_idx])
        user_info_val = pl.from_pandas(user_info.iloc[test_idx])
        
        behaviors_train_train = user_info_train.join(behaviors_train, on=['user_id', 'impression_id'], how='inner')
        behaviors_train_val = user_info_val.join(behaviors_train, on=['user_id', 'impression_id'], how='inner')
        behaviors_val_train = user_info_train.join(behaviors_val, on=['user_id', 'impression_id'], how='inner')
        behaviors_val_val = user_info_val.join(behaviors_val, on=['user_id', 'impression_id'], how='inner')
        
        features_train_train, _, unique_entities = PREPROCESSING[preprocessing_version](
            behaviors_train_train, history_train, articles, test=False, sample=False, previous_version=None,
            urm_path=urm_path, ners_models_path=ners_models_path, split_type='train', output_path=output_dir, 
            recsys_models_path=recsys_models_path, recsys_urm_path=recsys_urm_path)
        
        features_train_val, _, _ = PREPROCESSING[preprocessing_version](
            behaviors_train_val, history_train, articles, test=False, sample=False, previous_version=None,
            urm_path=urm_path, ners_models_path=ners_models_path, split_type='validation', output_path=output_dir, 
            recsys_models_path=recsys_models_path, recsys_urm_path=recsys_urm_path)
        
        features_val_train, _, _ = PREPROCESSING[preprocessing_version](
            behaviors_val_train, history_val, articles, test=False, sample=False, previous_version=None,
            urm_path=urm_path, ners_models_path=ners_models_path, split_type='train', output_path=output_dir, 
            recsys_models_path=recsys_models_path, recsys_urm_path=recsys_urm_path)
        
        features_val_val, _, _ = PREPROCESSING[preprocessing_version](
            behaviors_val_val, history_val, articles, test=False, sample=False, previous_version=None,
            urm_path=urm_path, ners_models_path=ners_models_path, split_type='validation', output_path=output_dir, 
            recsys_models_path=recsys_models_path, recsys_urm_path=recsys_urm_path)
        
        features_train = pl.concat([features_train_train, features_val_train], how='diagonal_relaxed')
        features_val = pl.concat([features_train_val, features_val_val], how='diagonal_relaxed')
        
        features_train.write_parquet(os.path.join(fold_path, f'train_ds.parquet'))
        features_val.write_parquet(os.path.join(fold_path, f'validation_ds.parquet'))
        
    categorical_columns = get_categorical_columns(preprocessing_version)
    categorical_columns += [f'Entity_{entity}_Present' for entity in unique_entities]

    logging.info(f'Preprocessing complete. There are {len(features_train.columns)} columns: {np.array(features_train.columns)}')

    dataset_info = {
        'type': 'group_kfold',
        'categorical_columns': categorical_columns,
        'unique_entities': unique_entities,
        'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }
    data_info_path = os.path.join(output_dir, 'data_info.json')
    with open(data_info_path, 'w') as data_info_file:
        json.dump(dataset_info, data_info_file)
    logging.info(f'Saved data info at: {data_info_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing group k fold")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the dataset is placed")
    parser.add_argument("-preprocessing_version", default='latest', type=str,
                        choices=['68f', '94f', '115f', '127f', '142f', '147f', 'new', 'latest'],
                        help="Specifiy the preprocessing version to use. Default is 'latest' valuses are ['68f', '94f', '115f','latest']")
    parser.add_argument("-urm_ner_path", default=None, type=str, required=False,
                        help="Specify the path of the already created urm to use to generate ners features.")
    parser.add_argument("-ners_models_path", default=None, type=str, required=False,
                        help="Specify the path of the already created urm to use to generate ners features.")
    parser.add_argument("-recsys_models_path", default = None, type=str,required=False,
                        help="Specify the path of the already trained recsys to use to generate recsys features.")
    parser.add_argument("-recsys_urm_path", default = None, type=str,required=False,
                        help="Specify the path of the already created urm to use to generate recsys features.")
    parser.add_argument("-n_folds", default=5, type=int, required=False,
                        help="Number of folds for cross validation")
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    PREPROCESSING_VERSION = args.preprocessing_version
    URM_PATH = args.urm_ner_path
    NER_MODELS_PATH = args.ners_models_path
    RECSYS_MODELS_PATH = args.recsys_models_path
    RECSYS_URM_PATH = args.recsys_urm_path
    N_FOLDS = args.n_folds

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f'preprocessing_user_group_{N_FOLDS}folds_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, experiment_name)
    os.makedirs(output_dir)

    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w",
                        format=LOGGING_FORMATTER, level=logging.INFO, force=True)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)

    main(DATASET_DIR, output_dir, PREPROCESSING_VERSION, N_FOLDS, 
         URM_PATH, NER_MODELS_PATH, RECSYS_URM_PATH, RECSYS_MODELS_PATH)
