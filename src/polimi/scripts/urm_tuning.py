import os
import logging
from datetime import datetime
import argparse
from pathlib import Path
import pandas as pd
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing_extensions import List, Tuple, Dict
import optuna
import polars as pl
import scipy.sparse as sps

import sys



sys.path.append('/home/ubuntu/RecSysChallenge2024/src')

from ebrec.evaluation.metrics_protocols import *
from RecSys_Course_AT_PoliMi.Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from RecSys_Course_AT_PoliMi.Recommenders.BaseRecommender import BaseRecommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from RecSys_Course_AT_PoliMi.Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from RecSys_Course_AT_PoliMi.Evaluation.Evaluator import EvaluatorHoldout
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased import P3alphaRecommender, RP3betaRecommender
from polimi.utils._custom import ALGORITHMS
from polimi.utils._custom import load_sparse_csr, save_json

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def get_params(trial: optuna.Trial, model: BaseRecommender):
    if model in [ItemKNNCFRecommender, UserKNNCFRecommender]:
        params = {
            "similarity": trial.suggest_categorical("similarity", ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky', 'euclidean']),
            "topK": trial.suggest_int("topK", 5, 1000),
            "shrink": trial.suggest_int("shrink", 0, 1000),
        }
        if params['similarity'] == "asymmetric":
            params["asymmetric_alpha"] = trial.suggest_float("asymmetric_alpha", 0, 2, log=False)
            params["normalize"] = True     

        elif params['similarity'] == "tversky":
            params["tversky_alpha"] = trial.suggest_float("tversky_alpha", 0, 2, log=False)
            params["tversky_beta"] = trial.suggest_float("tversky_beta", 0, 2, log=False)
            params["normalize"] = True 

        elif params['similarity'] == "euclidean":
            params["normalize_avg_row"] = trial.suggest_categorical("normalize_avg_row", [True, False])
            params["similarity_from_distance_mode"] = trial.suggest_categorical("similarity_from_distance_mode", ["lin", "log", "exp"])
            params["normalize"] = trial.suggest_categorical("normalize", [True, False])
        
    elif model == SLIMElasticNetRecommender:
        params = {
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 1e-5, 1e-1, log=True),
            "positive_only": True,
            "topK": trial.suggest_int("topK", 5, 100),
        }
    elif model == PureSVDRecommender:
        params = {
            "num_factors": trial.suggest_int("num_factors", 1, 100),
        }
    elif model == P3alphaRecommender:
        params = {
            "topK": trial.suggest_int("topK", 20, 100),
            'normalize_similarity': trial.suggest_categorical("normalize_similarity", [True]),
            'alpha': trial.suggest_float("alpha", 0.05, 0.5),
        }   
    elif model == RP3betaRecommender:
        params = {
            "topK": trial.suggest_int("topK", 20, 100),
            'normalize_similarity': trial.suggest_categorical("normalize_similarity", [True]),
            'alpha': trial.suggest_float("alpha", 0.05, 0.5),
            'beta': trial.suggest_float("beta", 0.05, 0.5),
        }  
    else:
        raise ValueError(f"Model {model.RECOMMENDER_NAME} not recognized")
    return params

def get_sampler_from_name(sampler_name: str):
    if sampler_name == 'RandomSampler':
        return optuna.samplers.RandomSampler()
    elif sampler_name == 'TPESampler':
        return optuna.samplers.TPESampler()
    else:
        raise ValueError(f"Sampler {sampler_name} not recognized")


def optimize_parameters(URM_train: sps.csr_matrix, URM_val: sps.csr_matrix, 
                        model_name: str, metric: str, 
                        cutoff:int, study_name: str, 
                        n_trials: int, storage: str, 
                        sampler:str) -> Tuple[Dict, pd.DataFrame]:
    
    model = ALGORITHMS[model_name][0]
    evaluator = EvaluatorHoldout(URM_val, cutoff_list=[cutoff], exclude_seen=False)

    def objective_function(trial: optuna.Trial):
        params = get_params(trial, model)
        rec_instance = model(URM_train)
        rec_instance.fit(**params)
        result_df, _ = evaluator.evaluateRecommender(rec_instance)    
        return result_df.loc[cutoff][metric]

    study = optuna.create_study(direction='maximize', 
                                study_name=study_name, 
                                storage=storage, 
                                sampler=get_sampler_from_name(sampler),
                                load_if_exists=True)
    study.optimize(objective_function, n_trials=n_trials, n_jobs=1)
    return study.best_params, study.trials_dataframe()
    

def main(urm_folder: Path, output_dir: Path,
         model_name:str, study_name: str, 
         n_trials: int, storage: str,
         sampler: str):
    
    urm_train_path = urm_folder.joinpath('URM_train.npz')
    urm_val_path = urm_folder.joinpath('URM_validation.npz')    
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr (prevents double logs)
    
    URM_train = load_sparse_csr(urm_train_path, logger=logging)
    URM_val =  load_sparse_csr(urm_val_path, logger=logging)
    best_params, trials_df = optimize_parameters(URM_train, URM_val,
                                                 metric='MAP',
                                                 model_name=model_name,
                                                 study_name=study_name, 
                                                 n_trials=n_trials, 
                                                 storage=storage,
                                                 sampler=sampler)
    
    
    params_file_path = output_dir.joinpath(f'{study_name}_best_params.json')
    logging.info(f'Best parameters: {best_params}')
    logging.info(f'Saving the best parameters at: {params_file_path}')
    save_json(best_params, params_file_path)
        
    trials_file_path = output_dir.joinpath('trials_dataframe.csv')
    logging.info(f'Saving the trials dataframe at: {trials_file_path}')
    trials_df.to_csv(trials_file_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="URM tuning")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-urm_folder", default=None, type=str, required=True,
                        help="Folder where URM sps.csr_matrix are placed")
    parser.add_argument("-model_name", default='ItemKNNCFRecommender', type=str, required=True,
                        help="Folder where URM sps.csr_matrix are placed")
    parser.add_argument("-n_trials", default=100, type=int, required=False,
                        help="Number of optuna trials to perform")
    parser.add_argument("-study_name", default=None, type=str, required=False,
                        help="Optional name of the study. Should be used if a storage is provided")
    parser.add_argument("-storage", default=None, type=str, required=False,
                        help="Optional storage url for saving the trials")
    
    parser.add_argument("-sampler", choices=['TPESampler', 'RandomSampler'], default='TPESampler', type=str, required=True,
                        help="Optuna sampler")
    
    
    
    args = parser.parse_args()
    OUTPUT_DIR = Path(args.output_dir)
    URM_FOLDER = Path(args.urm_folder)
    MODEL_NAME = args.model_name
    N_TRIALS = args.n_trials
    STUDY_NAME = args.study_name
    STORAGE = args.storage
    SAMPLER = args.sampler
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = OUTPUT_DIR.joinpath(f'urm_tuning_{MODEL_NAME}_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = output_dir.joinpath('log.txt')    
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(URM_FOLDER, output_dir, 
         model_name=MODEL_NAME, study_name=STUDY_NAME, 
         n_trials=N_TRIALS, storage=STORAGE, 
         sampler=SAMPLER)