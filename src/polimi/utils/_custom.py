import polars as pl
import numpy as np
from typing_extensions import List
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from RecSys_Course_AT_PoliMi.Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender


ALGORITHMS = {
    'RP3betaRecommender': [
        RP3betaRecommender,
        {

        }
    ],
    'ItemKNNCFRecommender': [
        ItemKNNCFRecommender,
        {
            
        }
    ],
    'PureSVDRecommender': [
        PureSVDRecommender,
        {
            
        }
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
        'articles': load_articles(base_path, type, split, lazy)
    }
    



def cosine_similarity(x: List[float], y: List[float]):
    x = np.array(x)
    y = np.array(y)
    normalization = np.linalg.norm(x, 2) * np.linalg.norm(y, 2)
    return np.dot(x, y) / normalization if normalization > 0 else 0


