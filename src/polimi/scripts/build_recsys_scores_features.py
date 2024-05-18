from polimi.utils._urm import build_recsys_features

import logging
from pathlib import Path
from datetime import datetime
import argparse
import polars as pl
import os
from polimi.utils._custom import load_articles,load_behaviors,load_history,load_recommenders,load_sparse_csr



LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"

def main(dataset_split: str , output_dir: Path, dataset_type: str, base_path: str, urm_type: str):

    #d_path = base_path.joinpath('dataset')
    dataset_path = Path('/home/ubuntu/dataset')

    if dataset_type in ['demo', 'small', 'large']:
        dataset_path = dataset_path.joinpath('ebnerd_{}'.format(dataset_type))
        train_path = dataset_path.joinpath('train')
        val_path = dataset_path.joinpath('validation')
        articles = pl.read_parquet(dataset_path.joinpath('articles.parquet'))
        if dataset_split == 'train':
            history_train = pl.read_parquet(train_path.joinpath('history.parquet'))
            history_val = pl.read_parquet(val_path.joinpath('history.parquet'))
            history = history_train.vstack(history_val)
            behaviors = pl.read_parquet(train_path.joinpath('behaviors.parquet'))
        elif dataset_split == 'validation':
            history_train = pl.read_parquet(train_path.joinpath('history.parquet'))
            history_val = pl.read_parquet(val_path.joinpath('history.parquet'))
            history = history_train.vstack(history_val)
            behaviors = pl.read_parquet(val_path.joinpath('behaviors.parquet'))
    elif dataset_type == 'testset': # Build final train URM for inference
        dataset_path = dataset_path.joinpath('ebnerd_{}'.format(dataset_type))
        # Load all the other large datasets
        train_path_large = dataset_path.parent.joinpath('ebnerd_large').joinpath('train')
        val_path_large = dataset_path.parent.joinpath('ebnerd_large').joinpath('validation')

        history_train_large = pl.read_parquet(train_path_large.joinpath('history.parquet'))
        history_val_large = pl.read_parquet(val_path_large.joinpath('history.parquet'))
        history_testset = pl.read_parquet(dataset_path.joinpath('test').joinpath('history.parquet'))

        history = history_train_large.vstack(history_val_large).vstack(history_testset)
        behaviors = pl.read_parquet(dataset_path.joinpath('test').joinpath('behaviors.parquet'))
        articles = pl.read_parquet(dataset_path.joinpath('articles.parquet'))

    


    

    recs_path = base_path.joinpath('algo').joinpath(urm_type).joinpath(dataset_type).joinpath(dataset_split)
    urm_path = base_path.joinpath('urm').joinpath(urm_type).joinpath(dataset_type).joinpath(f'URM_{dataset_split}.npz')
    
    logging.info(f"Loading recommenders from {recs_path}")
    recs = load_recommenders(URM=load_sparse_csr(path=urm_path), file_path=recs_path)

    logging.info(f"Building scores...")
    build_recsys_features(history=history, articles=articles, behaviors=behaviors,recs=recs, save_path=output_dir)
    
   
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creating recsys scores ...")
    parser.add_argument("-dataset_split", choices=['train', 'validation', 'test'], default='train',required=True, type=str,
                        help="Specify the type of dataset split: ['train', 'validation', 'test']")
    parser.add_argument("-output_dir", default=None, type=str,required=True,
                        help="The directory where the scores parquet will be placed")
    parser.add_argument("-dataset_type", choices=['demo', 'small', 'large', 'testset'],required=True, default='small', type=str,
                        help="Specify the type of dataset: ['demo', 'small', 'large']")
    parser.add_argument("-base_path", required=True, type=str,
                        help="Specify the base path")
    parser.add_argument("-urm_type", choices=['ner', 'recsys'],required=True, default='ner', type=str,
                        help="Specify the type of URM: ['ner', 'recsys']")
    
    args = parser.parse_args()
    DATASET_SPLIT = args.dataset_split
    OUTPUT_DIR = Path(args.output_dir)
    DATASET_TYPE = args.dataset_type
    BASE_PATH = Path(args.base_path)
    URM_TYPE = args.urm_type
       
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_DIR = OUTPUT_DIR.joinpath(URM_TYPE).joinpath(DATASET_TYPE).joinpath(DATASET_SPLIT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR.joinpath("log.txt")    
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_SPLIT, OUTPUT_DIR, DATASET_TYPE, BASE_PATH, URM_TYPE)
