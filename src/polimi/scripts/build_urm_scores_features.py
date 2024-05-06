

from pathlib import Path
from ebrec.evaluation.metrics_protocols import *
from RecSys_Course_AT_PoliMi.Recommenders.BaseRecommender import BaseRecommender
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from RecSys_Course_AT_PoliMi.Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_AsySVD_Cython, MatrixFactorization_BPR_Cython
from RecSys_Course_AT_PoliMi.Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender, PureSVDItemRecommender
from RecSys_Course_AT_PoliMi.Recommenders.Neural.MultVAERecommender import MultVAERecommender
from RecSys_Course_AT_PoliMi.Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender
from RecSys_Course_AT_PoliMi.Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from RecSys_Course_AT_PoliMi.Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from RecSys_Course_AT_PoliMi.Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_AsySVD_Cython, MatrixFactorization_BPR_Cython
from RecSys_Course_AT_PoliMi.Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender, PureSVDItemRecommender
from RecSys_Course_AT_PoliMi.Recommenders.Neural.MultVAERecommender import MultVAERecommender
from RecSys_Course_AT_PoliMi.Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender
from RecSys_Course_AT_PoliMi.Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from RecSys_Course_AT_PoliMi.Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from RecSys_Course_AT_PoliMi.Evaluation.Evaluator import EvaluatorHoldout
from polimi.utils._custom import ALGORITHMS
from polimi.utils._custom import load_sparse_csr, load_best_optuna_params, load_articles, load_behaviors, load_history, save_json
from polimi.utils._custom import load_sparse_csr, load_best_optuna_params, load_articles, load_behaviors, load_history, save_json
from polimi.utils._urm import train_recommender, build_ner_scores_features, load_recommender
import polars as pl
import scipy.sparse as sps
import time
import os
import os
    
    
def train_ner_score_algo(URM: sps.csr_matrix, rec_dir: Path, algo_dict:dict):
def train_ner_score_algo(URM: sps.csr_matrix, rec_dir: Path, algo_dict:dict):
    start_time = time.time()
    recs = []
    for rec, info in algo_dict.items():
        params = info['params']
        study_name = info['study_name']
        
        is_study_in_dir = f'{study_name}.zip' in os.listdir(rec_dir)
        is_load = 'load' in info and info['load']
        if is_load and is_study_in_dir:
            rec_instance = load_recommender(URM, rec, rec_dir, file_name=study_name)
        else:
            if is_load and not is_study_in_dir:
                print(f'Study {study_name} not found in {rec_dir}, loading params...')
            if not params:
                print(f'Params are missing, loading best params for {study_name}...')
                params = load_best_optuna_params(study_name)
                print(f'Loaded params: {params}')
            
            save_json(params, rec_dir.joinpath(f'{study_name}_params.json'))
        
        is_study_in_dir = f'{study_name}.zip' in os.listdir(rec_dir)
        is_load = 'load' in info and info['load']
        if is_load and is_study_in_dir:
            rec_instance = load_recommender(URM, rec, rec_dir, file_name=study_name)
        else:
            if is_load and not is_study_in_dir:
                print(f'Study {study_name} not found in {rec_dir}, loading params...')
            if not params:
                print(f'Params are missing, loading best params for {study_name}...')
                params = load_best_optuna_params(study_name)
                print(f'Loaded params: {params}')
            
            save_json(params, rec_dir.joinpath(f'{study_name}_params.json'))
            rec_instance = train_recommender(URM, rec, params, file_name=study_name, output_dir=rec_dir) #also saves the model
            
        recs.append(rec_instance)
    print(f'Loaded/trained ner scores algorithms in {((time.time() - start_time)/60):.1f} minutes')
    return recs
    
    
def build_ner_score_features(feature_ouput_dir: Path, recs: list, 
                             history: pl.DataFrame, behaviors: pl.DataFrame, articles: pl.DataFrame):
    
    print(f'Loaded/trained ner scores algorithms in {((time.time() - start_time)/60):.1f} minutes')
    return recs
    
    
def build_ner_score_features(feature_ouput_dir: Path, recs: list, 
                             history: pl.DataFrame, behaviors: pl.DataFrame, articles: pl.DataFrame):
    
    start_time = time.time()
    build_ner_scores_features(history=history, behaviors=behaviors, articles=articles, recs=recs, save_path=feature_ouput_dir)
    print(f'Built ner scores features in {((time.time() - start_time)/60):.1f} minutes')
    print(f'Built ner scores features in {((time.time() - start_time)/60):.1f} minutes')
    
if __name__ == '__main__':
    algo_ner_dict = {
        # PureSVDItemRecommender: {
        #     'params': None,
        #     'study_name': 'PureSVDItemRecommender-ner-small-ndcg100_new',
        #     'load': False
        # },
        # UserKNNCFRecommender: {
        #     'params': None,
        #     'study_name': 'UserKNNCFRecommender-ner-small-ndcg100_new',
        #     'load': False
        # },
        # ItemKNNCFRecommender: {
        #     'params': None,
        #     'study_name': 'ItemKNNCFRecommender-ner-small-ndcg100_new',
        #     'load': False
        # },
        # P3alphaRecommender: {
        #     'params': None,
        #     'study_name': 'P3alphaRecommender-ner-small-ndcg100_new',
        #     'load': False
        # },
        # RP3betaRecommender: {
        #     'params': None,
        #     'study_name': 'RP3betaRecommender-ner-small-ndcg100_new',
        #     'load': False
        # },
        # SLIM_BPR_Cython: {
        #     'params': None,
        #     'study_name': 'SLIM_BPR_Cython-ner-small-ndcg100_new',
        #     'load': False
        # },
         MatrixFactorization_BPR_Cython: {
            'params': None,
            'study_name': 'MatrixFactorization_BPR_Cython-ner-small-ndcg100_new',
            'load': False
        },
        # ItemKNNCFRecommender: {
        #     'params': None,
        #     'study_name': 'ItemKNNCFRecommender-ner-small-ndcg100_new',
        #     'load': False
        # },
        # P3alphaRecommender: {
        #     'params': None,
        #     'study_name': 'P3alphaRecommender-ner-small-ndcg100_new',
        #     'load': False
        # },
        # RP3betaRecommender: {
        #     'params': None,
        #     'study_name': 'RP3betaRecommender-ner-small-ndcg100_new',
        #     'load': False
        # },
        # SLIM_BPR_Cython: {
        #     'params': None,
        #     'study_name': 'SLIM_BPR_Cython-ner-small-ndcg100_new',
        #     'load': False
        # },
        #  MatrixFactorization_BPR_Cython: {
        #     'params': None,
        #     'study_name': 'MatrixFactorization_BPR_Cython-ner-small-ndcg100_new',
        #     'load': False
        # },
    }
    
    URM_TYPE = 'ner'
    DTYPE = 'large'
    DSPLIT = 'train'
    
    d_path = Path('/mnt/ebs_volume/recsys2024/dataset')
    
    urm_path = d_path.parent.joinpath('urm').joinpath(URM_TYPE).joinpath(DTYPE)
    urm_path = d_path.parent.joinpath('urm').joinpath(URM_TYPE).joinpath(DTYPE)
    URM = load_sparse_csr(urm_path.joinpath(f'URM_{DSPLIT}.npz') )

    algo_path = urm_path.joinpath('algo').joinpath(DSPLIT)
    algo_path.mkdir(parents=True, exist_ok=True)
    
    features_path = d_path.parent.joinpath('features').joinpath(DTYPE).joinpath(DSPLIT)
    features_path.mkdir(parents=True, exist_ok=True)
    
    history = load_history(d_path, DTYPE, DSPLIT, lazy=False)
    behaviors = load_behaviors(d_path, DTYPE, DSPLIT, lazy=False)
    articles = load_articles(d_path, DTYPE, lazy=False)

    recs = train_ner_score_algo(URM, algo_path, algo_ner_dict)
    
    # build_ner_score_features(feature_ouput_dir=features_path, recs=recs, 
    #                          history=history, behaviors=behaviors, articles=articles)
    recs = train_ner_score_algo(URM, algo_path, algo_ner_dict)
    
    # build_ner_score_features(feature_ouput_dir=features_path, recs=recs, 
    #                          history=history, behaviors=behaviors, articles=articles)
