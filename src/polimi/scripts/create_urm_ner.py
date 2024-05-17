import logging
from pathlib import Path
from datetime import datetime
import argparse
import gc
import polars as pl
from polimi.utils._urm import build_user_id_mapping, build_ner_mapping, build_ner_urm,build_recsys_urm,build_item_mapping, build_articles_with_processed_ner, compute_sparsity_ratio
from polimi.utils._custom import save_sparse_csr, read_json
import time


LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"

def main(dataset_path: Path, output_dir: Path, urm_type:str):
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")
    
    dtype = str(dataset_path).split('/')[-1]
    is_testset = dtype == 'ebnerd_testset'
    if is_testset:
        splits = ['test']
    else:
        splits = ['train', 'validation']
    
    articles = pl.read_parquet(dataset_path.joinpath('articles.parquet'))

    for split in splits:
        save_path = output_dir / dtype / split
        save_path.mkdir(parents=True, exist_ok=True)
        
        history = pl.read_parquet(dataset_path / split / 'history.parquet')
        behaviors = pl.read_parquet(dataset_path / split / 'behaviors.parquet')
        

        logging.info(f'Building the URM {dtype} {split}')
        start_time = time.time()
        if urm_type == 'ner':
            ap = build_articles_with_processed_ner(articles)
            ner_mapping = build_ner_mapping(ap)
            # output_file_path = output_dir.joinpath(f'URM_{split}')
            user_id_mapping = build_user_id_mapping(history)
            URM_train = build_ner_urm(history, ap, user_id_mapping, ner_mapping, 'article_id_fixed')
            logging.info(f'Sparsity ratio of the URM_train: {compute_sparsity_ratio(URM_train)}')
            save_sparse_csr(save_path / f'URM_train', URM_train, logger=logging)

            del URM_train
            gc.collect()

            URM_test = build_ner_urm(behaviors, ap, user_id_mapping, ner_mapping, 'article_ids_clicked')
            logging.info(f'Sparsity ratio of the URM_train: {compute_sparsity_ratio(URM_test)}')
            save_sparse_csr(save_path / f'URM_test', URM_test, logger=logging)
        
        logging.info(f'Built urm for split {split} in {((time.time() - start_time)/60):.1f} minutes')
            


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Building URM matrixes")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the dataset is placed")
    parser.add_argument("-output_dir", default="../../urm/", type=str,
                        help="The directory where the URMs will be placed")
    parser.add_argument("-urm_type", choices=['ner', 'recsys'], default='ner', type=str,
                        help="Specify the type of URM: ['ner', 'recsys']")
    
    args = parser.parse_args()
    DATASET_PATH = Path(args.dataset_path)
    OUTPUT_DIR = Path(args.output_dir)
    URM_TYPE = args.urm_type
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_DIR = OUTPUT_DIR.joinpath(URM_TYPE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR.joinpath("log.txt")    
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(dataset_path=DATASET_PATH, urm_type=URM_TYPE, output_dir=OUTPUT_DIR)
