import polars as pl
import pandas as pd
import os
import json
import numpy as np
from sklearn.utils import resample
from polimi.utils._inference import _inference
from ebrec.evaluation.metrics_protocols import *
from fastauc.fastauc.fast_auc import CppAuc
from tqdm import tqdm
import optuna

predictions = [
    '/home/ubuntu/experiments/hybrid_level2/prediction_level_1_validation/catboost_classifier_predictions.parquet',
    '/home/ubuntu/experiments/hybrid_level2/prediction_level_1_validation/catboost_ranker_predictions.parquet',
    '/home/ubuntu/experiments/hybrid_level2/prediction_level_1_validation/dcn_predictions.parquet',
    '/home/ubuntu/experiments/hybrid_level2/prediction_level_1_validation/GANDALF_predictions.parquet',
    '/home/ubuntu/experiments/hybrid_level2/prediction_level_1_validation/light_gbm_classifier_predictions.parquet',
    '/home/ubuntu/experiments/hybrid_level2/prediction_level_1_validation/mlp_predictions.parquet',
    '/home/ubuntu/experiments/hybrid_level2/prediction_level_1_validation/wd_predictions.parquet',
    # '/home/ubuntu/experiments_1/models_predictions/logistic_regression_predictions.parquet',
    # '/home/ubuntu/experiments_1/models_predictions/wide_deep_predictions.parquet'
]

names = ['catboost_classifier', 'catboost_ranker', 'dcn', 'GANDALF', 'light_gbm_classifier', 'mlp', 'wd']
# names = ['ranker', 'catboost', 'mlp', 'deep', 'fast', 'gandalf', 'lgbm', 'logistic', 'wide_deep']

N_TRIALS = 500

def prepare_data(df, index):
    if index in ['mlp', 'GANDALF', 'wd', 'dcn']:
        df = df.with_columns(
                pl.col(f'prediction_{index}').list.first()    
            )
    df = df.with_columns(
        (pl.col(f'prediction_{index}')-pl.col(f'prediction_{index}').min().over('impression_id')) / 
        (pl.col(f'prediction_{index}').max().over('impression_id')-pl.col(f'prediction_{index}').min().over('impression_id'))
    )
    
    if 'user_id' in df.columns:
        df = df.drop('user_id')
    print(df)
    return df

def join_dfs(dfs):
    df = dfs[0]
    for i in range(1, len(dfs)):
        df = df.join(dfs[i], on=['impression_id','article','target'])
    return df

def build_weighted_sum(dfs_pred, weights):
    order = len(weights)
    return dfs_pred.with_columns(
                    *[(weights[i] * pl.col(f'prediction_{names[i]}')).alias(f'prediction_{names[i]}') for i in range(order)]
                ).with_columns(
                    pl.sum_horizontal([f"prediction_{names[i]}" for i in range(order)]).alias('final_pred')
                ).drop([f"prediction_{names[i]}" for i in range(order)]).rename({'final_pred' : 'prediction'})
                
def eval_pred(pred ,cpp_auc):
    pred = pred.group_by('impression_id').agg(pl.col('target'), pl.col('prediction'))     
    return np.mean(
                [cpp_auc.roc_auc_score(np.array(y_t).astype(bool), np.array(y_s).astype(np.float32)) 
                    for y_t, y_s in zip(pred['target'].to_list(), 
                                        pred['prediction'].to_list())]
            )

def build_weights(perc_list):
    n_weights = len(perc_list)
    
    residual = 1.0
    
    weights = []
    for i in range(n_weights):
        weights.append(residual * perc_list[i])
        residual = residual - residual*perc_list[i]
    weights.append(residual)
    return weights

def optimize_parameters(df, n_models, cpp_auc, study_name: str = 'lightgbm_tuning', n_trials: int = 100, storage: str = None):
    
    def objective_function(trial: optuna.Trial):     
        perc_list = [trial.suggest_float(f"residual_{i}", 0, 1) for i in range(n_models - 1)]
        weights = build_weights(perc_list)
        pred = build_weighted_sum(df, weights)
        return eval_pred(pred,cpp_auc)
        
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)
    study.optimize(objective_function, n_trials=n_trials, n_jobs=1)
    return study.best_params, study.trials_dataframe()

if __name__ == "__main__":
    cpp_auc = CppAuc()
    n_models = len(predictions)
    dfs_pred = []
    for pred in range(n_models):
        dfs_pred.append(prepare_data(pl.read_parquet(predictions[pred]), names[pred]))
    
    df_pred = join_dfs(dfs_pred)
    
    best_param, df = optimize_parameters(df_pred, n_models, cpp_auc, n_trials=N_TRIALS)
    
    print('Best_param : ')
    print(best_param)
    print('Associated weights :')
    print(build_weights(best_param))
    
       