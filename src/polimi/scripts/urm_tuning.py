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

from ebrec.evaluation.metrics_protocols import *
from RecSys_Course_AT_PoliMi.Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender
from RecSys_Course_AT_PoliMi.Recommenders.BaseRecommender import BaseRecommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from RecSys_Course_AT_PoliMi.Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from RecSys_Course_AT_PoliMi.Evaluation.Evaluator import EvaluatorHoldout
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from polimi.utils._custom import ALGORITHMS
from polimi.utils._custom import load_sparse_csr, save_json, get_algo_params
import time

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"

def get_sampler_from_name(sampler_name: str, constant_liar: bool = False):
    if sampler_name == 'RandomSampler':
        return optuna.samplers.RandomSampler()
    elif sampler_name == 'TPESampler':
        return optuna.samplers.TPESampler(constant_liar=constant_liar)
    else:
        raise ValueError(f"Sampler {sampler_name} not recognized")


def optimize_parameters(URM_train: sps.csr_matrix, URM_val: sps.csr_matrix, 
                        model_name: str, metric: str, 
                        cutoff:int, study_name: str, 
                        n_trials: int, storage: str, 
                        sampler:str, jobs:int) -> Tuple[Dict, pd.DataFrame]:
    
    model = ALGORITHMS[model_name][0]
    evaluator = EvaluatorHoldout(URM_val, cutoff_list=[cutoff], exclude_seen=False)

    def objective_function(trial: optuna.Trial):
        start_time = time.time()
        params = get_algo_params(trial, model, eval=evaluator, eval_metric=metric)
        rec_instance = model(URM_train)
        rec_instance.fit(**params)
        
        if 'epochs' in params:
            epochs = rec_instance.get_early_stopping_final_epochs_dict()["epochs"]
            trial.set_user_attr("epochs", epochs)
        
        trial.set_user_attr("train_time (min)", f'{((time.time() - start_time)/60):.1f}')
        
        result_df, _ = evaluator.evaluateRecommender(rec_instance)
        return result_df.loc[cutoff][metric.upper()]

    study = optuna.create_study(direction='maximize', 
                                study_name=study_name, 
                                storage=storage, 
                                sampler=get_sampler_from_name(sampler),
                                load_if_exists=True)
    study.optimize(objective_function, n_trials=n_trials, n_jobs=jobs)
    return study.best_params, study.trials_dataframe()
    

def main(urm_folder: Path, output_dir: Path,
         model_name:str, study_name: str, 
         n_trials: int, storage: str,
         sampler: str, metric: str, jobs: int, cutoff:int):
    
    urm_train_path = urm_folder.joinpath('URM_train.npz')
    urm_val_path = urm_folder.joinpath('URM_validation.npz')    
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr (prevents double logs)
    
    URM_train = load_sparse_csr(urm_train_path, logger=logging)
    URM_val =  load_sparse_csr(urm_val_path, logger=logging)
    best_params, trials_df = optimize_parameters(URM_train=URM_train,
                                                 URM_val=URM_val,
                                                 model_name=model_name,
                                                 study_name=study_name, 
                                                 n_trials=n_trials, 
                                                 storage=storage,
                                                 cutoff=cutoff,
                                                 metric=metric,
                                                 sampler=sampler, 
                                                 jobs=jobs)
    
    
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
    parser.add_argument("-model_name", type=str, required=True,
                        help="Folder where URM sps.csr_matrix are placed")
    parser.add_argument("-n_trials", default=100, type=int, required=False,
                        help="Number of optuna trials to perform")
    parser.add_argument("-study_name", default=None, type=str, required=False,
                        help="Optional name of the study. Should be used if a storage is provided")
    parser.add_argument("-storage", default=None, type=str, required=False,
                        help="Optional storage url for saving the trials")
    parser.add_argument("-sampler", choices=['TPESampler', 'RandomSampler'], default='TPESampler', type=str, required=False,
                        help="Optuna sampler")
    parser.add_argument("-metric", choices=['NDCG', 'MAP'], default='NDCG', type=str, required=True,
                        help="Optimization metric")
    parser.add_argument("-n_jobs", default=1, type=int, required=False,
                        help="Optimization jobs")
    parser.add_argument("-cutoff", default=10, type=int, required=False,
                        help="Cutoff for the evaluation metric")
    
    
    args = parser.parse_args()
    OUTPUT_DIR = Path(args.output_dir)
    URM_FOLDER = Path(args.urm_folder)
    METRIC = args.metric
    MODEL_NAME = args.model_name
    N_TRIALS = args.n_trials
    STUDY_NAME = args.study_name
    STORAGE = args.storage
    SAMPLER = args.sampler
    JOBS = args.n_jobs
    CUTOFF = args.cutoff
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = OUTPUT_DIR.joinpath(f'{STUDY_NAME}_{timestamp}')
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
         metric=METRIC,
         sampler=SAMPLER, 
         jobs=JOBS, 
         cutoff=CUTOFF)