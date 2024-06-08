
from polimi.utils._strategies import _behaviors_to_history, moving_window_split_iterator
from polimi.preprocessing_pipelines.categorical_dict import get_categorical_columns
from polimi.preprocessing_pipelines.preprocessing_versions import BATCH_PREPROCESSING, PREPROCESSING
import os
from datetime import datetime
from pathlib import Path
import polars as pl
import json
import matplotlib.pyplot as plt
import gc
import time

import sys
sys.path.append('/home/ubuntu/RecSysChallenge2024/src')


LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"

PREPROCESSING_VERSION = 'new'
INPUT_PATH = '/home/ubuntu/dataset/ebnerd_small' 
OUTPUT_PATH = '/home/ubuntu/experiments/level_2_folds'

def main(input_path, output_dir, preprocessing_version='new'):
    print(f"Preprocessing version: ----{preprocessing_version}----")
    print("Starting to build the dataset")
    print(f"Dataset path: {input_path}")

    articles = pl.read_parquet(os.path.join(input_path, 'articles.parquet'))
    train_path = os.path.join(input_path, 'train')
    behaviors_train = pl.read_parquet(
        os.path.join(train_path, 'behaviors.parquet'))
    history_train = pl.read_parquet(
        os.path.join(train_path, 'history.parquet'))

    history_all = pl.concat([
        history_train.explode(pl.all().exclude('user_id')),
        _behaviors_to_history(behaviors_train).explode(
            pl.all().exclude('user_id')),
    ]).sort(['user_id', 'impression_time_fixed'])\
        .group_by('user_id').agg(pl.all())

    del history_train
    gc.collect()

    behaviors_all = pl.concat([
        behaviors_train,
    ]).sort('impression_time')
    del behaviors_train
    gc.collect()

    dataset_path = Path(input_path).parent
    output_dir = Path(output_dir)

    print(
        'Finished to build parquet files. Starting feature engineering')
    start_time = time.time()
    for i, (history_k_train, behaviors_k_train, history_k_val, behaviors_k_val) in enumerate(
        moving_window_split_iterator(
            history_all, behaviors_all, window=3.5, window_val=3.5, stride=5, verbose=True)
    ):
        print(history_k_train)
        print(behaviors_k_train)
        print(history_k_val)
        print(behaviors_k_val)
        print(f'Preprocessing fold {i}')
        fold_path = output_dir / f'train_{i+1}'
        fold_path.mkdir(parents=True, exist_ok=True)

        print(f'Starting training fold {i}...')
        features_k_train, _, unique_entities = PREPROCESSING[preprocessing_version](
            behaviors_k_train, history_k_train, articles, test=False, sample=False, previous_version=None,
            split_type='train', output_path=output_dir, emb_scores_path=None,
            urm_ner_scores_path=None, dataset_path=dataset_path)
        features_k_train.write_parquet(fold_path / 'train_ds.parquet')
        del features_k_train
        gc.collect()

        print(f'Starting validation fold {i}...')
        features_k_val, _, unique_entities = PREPROCESSING[preprocessing_version](
            behaviors_k_val, history_k_val, articles, test=False, sample=False, previous_version=None,
            split_type='validation', output_path=output_dir, emb_scores_path=None,
            urm_ner_scores_path=None, dataset_path=dataset_path)

        features_k_val.write_parquet(fold_path / 'validation_ds.parquet')
        del features_k_val
        gc.collect()

    categorical_columns = get_categorical_columns(preprocessing_version)
    categorical_columns += [
        f'Entity_{entity}_Present' for entity in unique_entities]

    print(
        f'Preprocessing moving window finished. {i + 1} folds completed in {((time.time() - start_time) / 60):.2f} minutes.')

    dataset_info = {
        'type': f'mw_w4_wval2_st2_kfold',
        'categorical_columns': categorical_columns,
        'unique_entities': unique_entities,
        'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }
    data_info_path = os.path.join(output_dir, 'data_info.json')
    with open(data_info_path, 'w') as data_info_file:
        json.dump(dataset_info, data_info_file)
    print(f'Saved data info at: {data_info_path}')

if __name__ == '__main__':
    main(INPUT_PATH, OUTPUT_PATH, PREPROCESSING_VERSION)