import polars as pl
import pandas as pd
from tqdm import tqdm
import os
import json
import numpy as np
from catboost import CatBoostClassifier, CatBoostRanker, Pool, sum_models
from sklearn.utils import resample
from polimi.utils._inference import _inference
from ebrec.evaluation.metrics_protocols import *
from ebrec.utils._behaviors import sampling_strategy_wu2019
from polimi.utils._polars import reduce_polars_df_memory_size
from polimi.test.level_2_ensemble.build_model_predictions import require_subsampled_set, train_predict_model


original_dataset_path = '/home/ubuntu/dataset/ebnerd_small/train/behaviors.parquet'
output_path = '/home/ubuntu/experiments/hybrid_level2'

LEVEL_1_MODELS = ['catboost_ranker', 'catboost_classifier', 'light_gbm_classifier']
NPRATIO = 2

def prediction_feature_eng(df, models):
    '''
        Do feature engineering of the predictions
    '''
    df = df.with_columns(
        *[
            (pl.col(f'prediction_{model}')-pl.col(f'prediction_{model}').min().over('impression_id')) / 
            (pl.col(f'prediction_{model}').max().over('impression_id')-pl.col(f'prediction_{model}').min().over('impression_id'))
            for model in models
        ]
    )
    
    return df


def load_dataset(dataset_path, original_dataset_path):
    '''
        load the dataset
        return train, validation and subsampled train
    '''
    train_ds = reduce_polars_df_memory_size(pl.read_parquet(os.path.join(dataset_path, 'train_ds.parquet')), verbose=False)
    
    starting_dataset =  pl.read_parquet(original_dataset_path).select(['impression_id','user_id','article_ids_inview','article_ids_clicked'])
    starting_dataset = starting_dataset.filter(pl.col('impression_id').is_in(train_ds['impression_id']))
    
    behaviors = pl.concat(
        rows.pipe(
            sampling_strategy_wu2019, npratio=NPRATIO, shuffle=False, with_replacement=True, seed=123
        ).explode('article_ids_inview').drop(columns =['article_ids_clicked']).rename({'article_ids_inview' : 'article'})\
        .with_columns(pl.col('user_id').cast(pl.UInt32),
                      pl.col('article').cast(pl.Int32))\
        
         for rows in tqdm(starting_dataset.iter_slices(1000), total=starting_dataset.shape[0] // 1000)
    )
    
    subsampled_train_ds = behaviors.join(train_ds, on = ['impression_id','user_id','article'], how = 'left')
    
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
    val_ds = reduce_polars_df_memory_size(pl.read_parquet(os.path.join(dataset_path, 'validation_ds.parquet')), verbose=False) 
    
    return subsampled_train_ds, train_ds, val_ds, data_info

def fit_models_save_predictions(models, subsampled_train_ds, train_ds, val_ds, data_info, output_path):
    '''
        fit the models in models list and return the dataframe containing the predictions of each model
        return prediction dataframe
    '''
    prediction_all = []
    for model in models:
        if require_subsampled_set(model):
            prediction = train_predict_model(subsampled_train_ds, val_ds, data_info, model)
        else : 
            prediction = train_predict_model(train_ds, val_ds, data_info, model)
        prediction = prediction.rename({'prediction' : f'prediction_{model}'})
        if 'user_id' in prediction.columns:
            prediction = prediction.drop('user_id')
        prediction.write_parquet(output_path + f'/{model}_predictions.parquet')
        prediction_all.append(prediction)
    
    merged_df = prediction_all[0]
    
    for df in range(1, len(prediction_all)):
        merged_df = merged_df.join(prediction_all[df], on=['impression_id','article','target'])
        
    merged_df.write_parquet(output_path + '/prediction.parquet')
    return merged_df
            
if __name__ == '__main__':
    
    # READ TRAIN DATASET AND BUILD THE SUBSAMPLED DATASET
    subsampled_train_ds, train_ds, val_ds, data_info = load_dataset(
        dataset_path='/home/ubuntu/experiments/hybrid_level2/train_1',
        original_dataset_path=original_dataset_path
    )
    
    # FIT THE MODELS AND GET THE PREDICTION
    level2_train_df = fit_models_save_predictions(LEVEL_1_MODELS, subsampled_train_ds, train_ds, 
                                                  val_ds, data_info, 
                                                  output_path + '/prediction_level_1_train')

    # DO FEATURE ENGINEERING OF THE PREDICTIONS
    level2_train_df = prediction_feature_eng(level2_train_df, LEVEL_1_MODELS)
    # READ THE TRAINING ALL TOGETHER 
    subsampled_train_ds, train_ds, val_ds, data_info = load_dataset(
        dataset_path='/home/ubuntu/experiments/preprocessing_train_small_new',
        original_dataset_path=original_dataset_path
    )
    
    # BUILD THE PREDICTIONS 
    level2_val_df = fit_models_save_predictions(LEVEL_1_MODELS, subsampled_train_ds, train_ds, 
                                                  val_ds, data_info, 
                                                  output_path + '/prediction_level_1_validation')

    # DO FEATURE ENGINNERING OF THE PREDICTIONS
    level2_train_df = prediction_feature_eng(level2_val_df, LEVEL_1_MODELS)
    # NOW YOU CAN DO THE TUNING OF THE MODEL
    
    