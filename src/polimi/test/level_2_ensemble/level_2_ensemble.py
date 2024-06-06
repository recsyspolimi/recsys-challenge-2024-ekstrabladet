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
from fastauc.fastauc.fast_auc import CppAuc
from polars import testing

original_dataset_path = '/home/ubuntu/dataset/ebnerd_small/train/behaviors.parquet'
output_path = '/home/ubuntu/experiments/hybrid_level2'

TRAIN_MODELS = True
LEVEL_1_MODELS_TRAIN = ['catboost_classifier', 'mlp', 'GANDALF', 'wd', 'dcn'] #'catboost_ranker', 
LEVEL_1_MODELS_PREDICTIONS = [ 'catboost_classifier','mlp', 'GANDALF', 'wd', 'dcn'] #'catboost_ranker',
NPRATIO = 2
TRAIN_TRAIN_PATH = '/home/ubuntu/experiments/hybrid_level2/train_1'
TRAIN_VAL_PATH = '/home/ubuntu/experiments/hybrid_level2/train_1'
VAL_TRAIN_PATH = '/home/ubuntu/experiments/preprocessing_train_small_new'
VAL_VAL_PATH = '/home/ubuntu/experiments/preprocessing_validation_small_new'
RANKER = True
params = {
    'iterations': 2000,
    'depth': 8,
    'colsample_bylevel': 0.5
}

def prediction_feature_eng(df, models):
    '''
        Do feature engineering of the predictions
    '''
    for model in models:
        if model in ['mlp', 'GANDALF', 'wd', 'dcn']:
            df = df.with_columns(
                    pl.col(f'prediction_{model}').list.first()    
                )
    df = df.with_columns(
        *[
            (pl.col(f'prediction_{model}')-pl.col(f'prediction_{model}').min().over('impression_id')) / 
            (pl.col(f'prediction_{model}').max().over('impression_id')-pl.col(f'prediction_{model}').min().over('impression_id'))
            for model in models
        ]
    )
    
    return df

def load_dataset(train_path, val_path, original_dataset_path):
    '''
        load the dataset
        return train, validation and subsampled train
    '''
    train_ds = reduce_polars_df_memory_size(pl.read_parquet(os.path.join(train_path, 'train_ds.parquet')), verbose=False)
    
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
    
    with open(os.path.join(train_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
    val_ds = reduce_polars_df_memory_size(pl.read_parquet(os.path.join(val_path, 'validation_ds.parquet')), verbose=False) 
    
    return subsampled_train_ds, train_ds, val_ds, data_info

def fit_models_save_predictions(models, subsampled_train_ds, train_ds, val_ds, data_info, output_path):
    '''
        fit the models in models list and return the dataframe containing the predictions of each model
        return prediction dataframe
    '''
    for model in models:
        if require_subsampled_set(model):
            prediction = train_predict_model(subsampled_train_ds, val_ds, data_info, model)
        else : 
            prediction = train_predict_model(train_ds, val_ds, data_info, model)
        prediction = prediction.rename({'prediction' : f'prediction_{model}'})
        if 'user_id' in prediction.columns:
            prediction = prediction.drop('user_id')
        prediction.write_parquet(output_path + f'/{model}_predictions.parquet')

def load_predictions(dir, model_list):
    model_name = model_list[0]
    print(f'Loading Predictions for {model_name}')
    merged_df = reduce_polars_df_memory_size(pl.read_parquet(dir + f'/{model_name}_predictions.parquet'), verbose=0)\
        .sort(by=['impression_id','article'])
    original_shape = merged_df.shape[0]
    for df in range(1, len(model_list)):
        model_name = model_list[df]
        print(f'Loading Predictions for {model_name}')
        model_predictions = reduce_polars_df_memory_size(pl.read_parquet(dir + f'/{model_name}_predictions.parquet'),verbose=0).sort(by=['impression_id','article'])
        testing.assert_frame_equal(merged_df.select(['impression_id','article','target']), 
                                   model_predictions.select(['impression_id','article','target']))
        merged_df = merged_df.with_columns(
            model_predictions[f'prediction_{model_name}'].alias(f'prediction_{model_name}')
        )
        assert original_shape == merged_df.shape[0]
        
    return merged_df
            
if __name__ == '__main__':
    
    if TRAIN_MODELS:
        # # READ TRAIN DATASET AND BUILD THE SUBSAMPLED DATASET
        # subsampled_train_ds, train_ds, val_ds, data_info = load_dataset(
        #     train_path=TRAIN_TRAIN_PATH,
        #     val_path=TRAIN_VAL_PATH,
        #     original_dataset_path=original_dataset_path
        # )
        
        # # FIT THE MODELS AND GET THE PREDICTION
        # fit_models_save_predictions(LEVEL_1_MODELS_TRAIN, subsampled_train_ds, train_ds, 
        #                                             val_ds, data_info, 
        #                                             output_path + '/prediction_level_1_train')
        
        # READ THE TRAINING ALL TOGETHER 
        subsampled_train_ds, train_ds, val_ds, data_info = load_dataset(
            train_path=VAL_TRAIN_PATH,
            val_path=VAL_VAL_PATH,
            original_dataset_path=original_dataset_path
        )
        
        # BUILD THE PREDICTIONS 
        fit_models_save_predictions(LEVEL_1_MODELS_TRAIN, subsampled_train_ds, train_ds, 
                                                    val_ds, data_info, 
                                                    output_path + '/prediction_level_1_validation')
        
    # LOAD PREDICTIONS
    level2_train_df = load_predictions(output_path + '/prediction_level_1_train', LEVEL_1_MODELS_PREDICTIONS,
                                       output_path + '/prediction_level_1_train')
    level2_train_df.write_parquet(output_path + '/prediction_level_1_train' + '/prediction.parquet')

    # DO FEATURE ENGINEERING OF THE PREDICTIONS 
    level2_train_df = prediction_feature_eng(level2_train_df, LEVEL_1_MODELS_PREDICTIONS)
    
    # LOAD PREDICTIONS
    level2_val_df = load_predictions(output_path + '/prediction_level_1_validation', LEVEL_1_MODELS_PREDICTIONS, 
                                     output_path + '/prediction_level_1_validation')
    
    level2_train_df.write_parquet(output_path + '/prediction_level_1_validation' + '/prediction.parquet')

    # DO FEATURE ENGINNERING OF THE PREDICTIONS
    # level2_val_df = prediction_feature_eng(level2_val_df, LEVEL_1_MODELS_PREDICTIONS)
    
    # # NOW YOU CAN DO THE TUNING OF THE MODEL
    # if RANKER:
    #     level2_train_df = level2_train_df.sort(by='impression_id')
    #     groups = level2_train_df.select('impression_id').to_numpy().flatten()
        
    # level2_train_df = level2_train_df.to_pandas()
    # group_ids = level2_train_df['impression_id'].to_frame()
    # level2_train_df = level2_train_df.drop(columns=['impression_id', 'article', 'user_id'])
    
    # X_train = level2_train_df.drop(columns=['target'])
    # y_train = level2_train_df['target']
    
    # model = CatBoostRanker(**params)
    # model.fit(X_train, y_train, group_id=groups, verbose=50)
    
    # level2_val_df = level2_val_df.to_pandas()
    # X_val = level2_val_df[X_train.columns]
    # evaluation_ds = pl.from_pandas(level2_val_df[['impression_id', 'article', 'target']])
    
    # prediction_ds = evaluation_ds.with_columns(pl.Series(model.predict(X_val)).alias('prediction'))
    
    # cpp_auc = CppAuc()
    # result = np.mean(
    #         [cpp_auc.roc_auc_score(np.array(y_t).astype(bool), np.array(y_s).astype(np.float32)) 
    #             for y_t, y_s in zip(prediction_ds['target'].to_list(), 
    #                                 prediction_ds['prediction'].to_list())]
    #     )
    # print(result)
    
    