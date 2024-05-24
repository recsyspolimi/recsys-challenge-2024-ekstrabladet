

from pathlib import Path
import os
import logging
from datetime import datetime
import argparse
import polars as pl
import gc
from pathlib import Path
import math
from tqdm import tqdm
from polimi.utils._polars import reduce_polars_df_memory_size, stack_slices
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
from polimi.utils._custom import load_sparse_csr, load_best_optuna_params, load_articles, load_behaviors, load_history, save_json, read_json
from polimi.utils._custom import load_sparse_csr, load_best_optuna_params, load_articles, load_behaviors, load_history, save_json
from polimi.utils._urm import train_recommender, build_urm_ner_scores, load_recommender
import polars as pl
import scipy.sparse as sps
import time
import os
from polimi.utils._urm import build_user_id_mapping, build_articles_with_processed_ner, build_ner_mapping
LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"

    
ALGO_NER_TRAIN_DICT = {
        RP3betaRecommender: {
            'params': None,
            'study_name': 'RP3betaRecommender-ner-small-ndcg100_new',
            'load': True
        },
        PureSVDItemRecommender: {
            'params': None,
            'study_name': 'PureSVDItemRecommender-ner-small-ndcg100_new',
            'load': True
        },
        ItemKNNCFRecommender: {
            'params': None,
            'study_name': 'ItemKNNCFRecommender-ner-small-ndcg100_new',
            'load': True
        },
        MatrixFactorization_BPR_Cython: {
            'params': None,
            'study_name': 'MatrixFactorization_BPR_Cython-ner-small-ndcg100_new',
            'load': True
        },
        P3alphaRecommender: {
            'params': None,
            'study_name': 'P3alphaRecommender-ner-small-ndcg100_new',
            'load': True
        },
    }
    
def train_ner_score_algo(URM: sps.csr_matrix, rec_dir: Path, algo_dict:dict=ALGO_NER_TRAIN_DICT, save_algo:bool=True):
    start_time = time.time()
    recs = []
    for rec, info in algo_dict.items():
        params = info['params']
        study_name = info['study_name']
           
        is_study_in_dir = f'{study_name}.zip' in os.listdir(rec_dir)
        is_load = 'load' in info and info['load']
        if is_load and is_study_in_dir:
            params = read_json(rec_dir.joinpath(f'{study_name}_params.json'))
            logging.info(f'Loading {rec} with params {params}...')
            rec_instance = load_recommender(URM, rec, rec_dir, file_name=study_name)
        else:
            if is_load and not is_study_in_dir:
                logging.info(f'Study {study_name} not found in {rec_dir}, loading params...')
            if not params:
                logging.info(f'Params are missing, loading best params for {study_name}...')
                params = load_best_optuna_params(study_name)
                logging.info(f'Loaded params: {params}')
            
            if save_algo and rec_dir:
                save_json(params, rec_dir / f'{study_name}_params.json')
            rec_instance = train_recommender(URM, rec, params, file_name=study_name, output_dir=rec_dir if save_algo else None)
            
        recs.append(rec_instance)
    logging.info(f'Loaded/Trained ner scores algorithms in {((time.time() - start_time)/60):.1f} minutes')
    return recs
    

    
def main(dataset_path: Path, urm_path: Path, output_path: Path):    
    
    dtype = str(dataset_path).split('/')[-1]
    is_testset = dtype == 'ebnerd_testset'
    if is_testset:
        splits = ['test']
        
    else:
        splits = ['train', 'validation']
    
    
    articles = pl.read_parquet(dataset_path / 'articles.parquet')
    unique_articles = articles['article_id'].unique().to_numpy()
    
    for split in splits:
        save_path = output_path / dtype / split
        save_path.mkdir(parents=True, exist_ok=True)
        
        URM_train = load_sparse_csr(urm_path / split / f'URM_train.npz')
        
        algo_path = urm_path / split / 'algo'
        algo_path.mkdir(parents=True, exist_ok=True)
        recs = train_ner_score_algo(URM_train, algo_path, ALGO_NER_TRAIN_DICT)
        assert len(recs) == len(ALGO_NER_TRAIN_DICT.keys()), 'Some algorithms were not loaded/trained'
        
        del URM_train
        gc.collect()

        history = pl.read_parquet(dataset_path / split / 'history.parquet')
        unique_users = history['user_id'].unique().to_numpy()
        behaviors = pl.read_parquet(dataset_path / split / 'behaviors.parquet')
                

        start_time = time.time()
        behaviors = behaviors.sort(['user_id', 'impression_id'])
        BATCH_SIZE = int(1e6)
        n_slices = math.ceil(len(behaviors) / BATCH_SIZE)
        for i, slice in enumerate(tqdm(behaviors.iter_slices(BATCH_SIZE), total=n_slices)):
            logging.info(f'Starting urm ner scores slice {i}...')
            
            slice = build_urm_ner_scores(slice, history, articles, recs, batch_size=BATCH_SIZE)
            slice = reduce_polars_df_memory_size(slice)
            
            logging.info(f'Saving urm ner scores slice {i} to {save_path}')
            slice.write_parquet(save_path / f'urm_ner_scores_slice_{i}.parquet')
            
            assert np.all(np.in1d(slice['user_id'].unique().to_numpy(), unique_users)), 'Some users were not found in history'
            assert np.all(np.in1d(slice['article'].explode().unique().to_numpy(), unique_articles)), 'Some candidates were not found in articles'
        logging.info(f'Built ner scores features slices in {((time.time() - start_time)/60):.1f} minutes')
        agg_slices_paths = list(save_path.glob('urm_ner_scores_slice_*.parquet'))
        assert len(agg_slices_paths) == n_slices, 'Some slices were not saved'
        stack_slices(agg_slices_paths, save_path, 'urm_ner_scores', delete_all_slices=True)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Training script for generating embeddings scores for the dataset")

    parser.add_argument("-output_dir", default="~/experiments", type=str,
                        help="The directory where the dataset will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the dataset is placed")
    parser.add_argument("-urm_path", default=None, type=str, required=True,
                        help="Directory where the URM are placed")

    args = parser.parse_args()
    OUTPUT_DIR = Path(args.output_dir)
    DATASET_DIR = Path(args.dataset_path)
    URM_DIR = Path(args.urm_path)


    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_name = f'preprocessing_urm_ner_scores_{timestamp}'
    output_dir = OUTPUT_DIR / out_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "log.txt"
    logging.basicConfig(filename=log_path, filemode="w",
                        format=LOGGING_FORMATTER, level=logging.INFO, force=True)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)

    main(DATASET_DIR, URM_DIR, output_dir)
