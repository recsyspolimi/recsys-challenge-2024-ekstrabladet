

from pathlib import Path
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
from polimi.utils._custom import load_sparse_csr, load_best_optuna_params, load_articles, load_behaviors, load_history
from polimi.utils._urm import train_recommender, build_ner_scores_features, load_recommender
import polars as pl
import scipy.sparse as sps
    
def build_ner_score_features(feature_ouput_dir: Path, rec_dir: Path, URM: sps.csr_matrix,
         algo_dict: dict, history: pl.DataFrame, behaviors: pl.DataFrame, articles: pl.DataFrame):
    
    recs = []
    for rec, info in algo_dict.items():
        params = info['params']
        study_name = info['study_name']
        if not params:
            print(f'Params are missing, loading best params for {study_name}...')
            params = load_best_optuna_params(study_name)
            print(f'Loaded params: {params}')
        
        if 'load' in info and info['load']:
            rec_instance = load_recommender(URM, rec, rec_dir, file_name=study_name)
        else:
            rec_instance = train_recommender(URM, rec, params, file_name=study_name, output_dir=rec_dir) #also saves the model
            
        recs.append(rec_instance)
        
    build_ner_scores_features(history=history, behaviors=behaviors, articles=articles, recs=recs, save_path=feature_ouput_dir)
    
if __name__ == '__main__':
    algo_dict = {
        UserKNNCFRecommender: {
            'params': None,
            'study_name': 'UserKNNCFRecommender-ner-small-ndcg100_new',
            'load': False
        },
    }
    
    URM_TYPE = 'ner'
    DTYPE = 'small'
    DSPLIT = 'train'
    
    d_path = Path('/mnt/ebs_volume/recsys2024/dataset')
    
    urm_path = Path('/mnt/ebs_volume/urm').joinpath(URM_TYPE).joinpath(DTYPE)
    URM = load_sparse_csr(urm_path.joinpath(f'URM_{DSPLIT}.npz') )

    algo_path = urm_path.joinpath('algo').joinpath(DSPLIT)
    algo_path.mkdir(parents=True, exist_ok=True)
    
    features_path = d_path.joinpath('features').joinpath(DTYPE).joinpath(DSPLIT)
    features_path.mkdir(parents=True, exist_ok=True)
    
    history = load_history(d_path, DTYPE, DSPLIT, lazy=False)
    behaviors = load_behaviors(d_path, DTYPE, DSPLIT, lazy=False)
    articles = load_articles(d_path, DTYPE, lazy=False)

    build_ner_score_features(feature_ouput_dir=features_path, rec_dir=algo_path, 
         URM=URM, algo_dict=algo_dict, history=history, behaviors=behaviors, articles=articles)
