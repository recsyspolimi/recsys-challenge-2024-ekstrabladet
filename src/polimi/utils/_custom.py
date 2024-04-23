import polars as pl
import numpy as np
from typing_extensions import List
from RecSys_Course_AT_PoliMi.Recommenders.BaseRecommender import BaseRecommender
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from RecSys_Course_AT_PoliMi.Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_AsySVD_Cython, MatrixFactorization_BPR_Cython
from RecSys_Course_AT_PoliMi.Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from RecSys_Course_AT_PoliMi.Recommenders.Neural.MultVAERecommender import MultVAERecommender
from RecSys_Course_AT_PoliMi.Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender

from os import getpid
from psutil import Process
from colorama import Fore, Style
from pathlib import Path
import numpy as np
import scipy.sparse as sps
import json
import optuna

_PARQUET_TYPE = 'parquet'

def load_history(base_path, type, split, lazy=False):
    path = base_path / f'ebnerd_{type}' / split / f'history.{_PARQUET_TYPE}'
    if lazy:
        return pl.scan_parquet(path)
    return pl.read_parquet(path)


def load_behaviors(base_path, type, split, lazy=False):
    path = base_path / f'ebnerd_{type}' / split / f'behaviors.{_PARQUET_TYPE}'
    if lazy:
        return pl.scan_parquet(path)
    return pl.read_parquet(path)


def load_articles(base_path, type, lazy=False):
    path = base_path / f'ebnerd_{type}' / f'articles.{_PARQUET_TYPE}'
    if lazy:
        return pl.scan_parquet(path)
    return pl.read_parquet(path)


def load_dataset(base_path, type, split, lazy=False):
    return {
        'history': load_history(base_path, type, split, lazy),
        'behaviors': load_behaviors(base_path, type, split, lazy),
        'articles': load_articles(base_path, type, lazy)
    }
    



def cosine_similarity(x: List[float], y: List[float]):
    x = np.array(x)
    y = np.array(y)
    normalization = np.linalg.norm(x, 2) * np.linalg.norm(y, 2)
    return np.dot(x, y) / normalization if normalization > 0 else 0


def PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):
    print(style + color + text + Style.RESET_ALL)
    
def GetMemUsage():   
    pid = getpid()
    py = Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return f"RAM memory GB usage = {memory_use :.4}"




def save_sparse_csr(path: Path, array: sps.csr_matrix, logger=None):
    directory_path = path.parents[0]    
    directory_path.mkdir(parents=True, exist_ok=True)
    
    np.savez(path, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)
    
    if logger:
        logger.info(f"File saved at: {path}")
    else:
        print('File saved at:', path)

def load_sparse_csr(path: Path, logger=None) -> sps.csr_matrix:
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")
    
    loader = np.load(path)
    if logger:
        logger.info(f"File loaded at: {path}")
    else:
        print('File loaded at:', path)
    return sps.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
    
def read_json(file_path: Path):
    res = {}
    try:
        with open(file_path) as file:
            res = json.load(file)
    except FileNotFoundError:
        pass
    return res

def save_json(data: dict, file_path: Path):
    with open(file_path, 'w') as file:
        json.dump(data, file)
        
        
ALGORITHMS = {
    'RP3betaRecommender': [
        RP3betaRecommender,
    ],
    'P3alphaRecommender': [
        P3alphaRecommender,
    ],
    'ItemKNNCFRecommender': [
        ItemKNNCFRecommender,
    ],
    'UserKNNCFRecommender': [
        UserKNNCFRecommender,
    ],
    'PureSVDRecommender': [
        PureSVDRecommender,
    ],
    'MultiThreadSLIM_SLIMElasticNetRecommender': [
        MultiThreadSLIM_SLIMElasticNetRecommender
    ],
    'SLIMElasticNetRecommender': [
        SLIMElasticNetRecommender
    ]
}
        
def get_algo_params(trial: optuna.Trial, model: BaseRecommender):
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
        
    elif model in [SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender]:
        params = {
            "alpha": trial.suggest_float("alpha", 1e-3, 1, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 1e-5, 1, log=True),
            "positive_only": True,
            "topK": trial.suggest_int("topK", 5, 1000),
        }
    elif model == PureSVDRecommender:
        params = {
            "num_factors": trial.suggest_int("num_factors", 1, 1000),
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
    elif model == MatrixFactorization_AsySVD_Cython:
        params = {
            "sgd_mode": trial.suggest_categorical("sgd_mode", ["sgd", "adagrad", "adam"]),
            "epochs": 500,
            "use_bias": trial.suggest_categorical("use_bias", [True, False]),
            "batch_size": trial.suggest_categorical("batch_size", [1]),
            "num_factors": trial.suggest_int("num_factors", 1, 200),
            "item_reg": trial.suggest_float("item_reg", 1e-5, 1e-2, log=True),
            "user_reg": trial.suggest_float("user_reg", 1e-5, 1e-2, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "negative_interactions_quota": trial.suggest_float("negative_interactions_quota", 0.0, 0.5),
            "epochs": 500,
        }
    elif model == MatrixFactorization_BPR_Cython:
        params = {
            "sgd_mode": trial.suggest_categorical("sgd_mode", ["sgd", "adagrad", "adam"]),
            "num_factors": trial.suggest_int("num_factors", 1, 200),
            "batch_size": trial.suggest_categorical("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
            "positive_reg": trial.suggest_float("positive_reg", 1e-5, 1e-2, log=True),
            "negative_reg": trial.suggest_float("negative_reg", 1e-5, 1e-2, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "positive_threshold_BPR": None,
            "epochs": 1000,
        }
    elif model == MultVAERecommender:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True),
            "l2_reg": trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0., 0.8),
            "total_anneal_steps": trial.suggest_int("total_anneal_steps", 100000, 600000),
            "anneal_cap": trial.suggest_float("anneal_cap", 0., 0.6),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
            "encoding_size": trial.suggest_int("encoding_size", 1, 512),
            "next_layer_size_multiplier": trial.suggest_int("next_layer_size_multiplier", 2, 10),
            "max_n_hidden_layers": trial.suggest_int("max_n_hidden_layers", 1, 4),
            "max_parameters": trial.suggest_categorical("max_parameters", [7*1e9*8/32]),            
            "epochs": 500
        }
    else:
        raise ValueError(f"Model {model.RECOMMENDER_NAME} not recognized")
    return params