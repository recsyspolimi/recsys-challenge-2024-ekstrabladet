import polars as pl
import numpy as np
from typing_extensions import List
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from RecSys_Course_AT_PoliMi.Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from os import getpid
from psutil import Process
from colorama import Fore, Style
from pathlib import Path
import numpy as np
import scipy.sparse as sps
import json
import optuna


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
    ]
}
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