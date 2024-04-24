import logging
from pathlib import Path
from datetime import datetime
import argparse
import polars as pl
from polimi.utils._urm import build_user_id_mapping, build_ner_mapping, build_ner_urm,build_recsys_urm, build_articles_with_processed_ner, compute_sparsity_ratio
from polimi.utils._custom import save_sparse_csr, read_json


LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"

def main(dataset_path: Path, dataset_type:str, urm_type:str, urm_split:str, output_dir: Path):
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")
    articles = pl.read_parquet(dataset_path.joinpath('articles.parquet'))
    train_path = dataset_path.joinpath('train')
    history_train = pl.read_parquet(train_path.joinpath('history.parquet'))
    behaviors_train = pl.read_parquet(train_path.joinpath('behaviors.parquet'))
    
    val_path = dataset_path.joinpath('validation')
    history_val = pl.read_parquet(val_path.joinpath('history.parquet'))
    behaviors_val = pl.read_parquet(val_path.joinpath('behaviors.parquet'))
    
    
    data_info = read_json(dataset_path.joinpath('data_info.json'))
    logging.info(f'Data info: {data_info}')
    
    logging.info(f'Building the URM {dataset_type} {urm_type} {urm_split}')
    if urm_type == 'ner':
        user_id_mapping = build_user_id_mapping(history_train.vstack(history_val))
        ap = build_articles_with_processed_ner(articles)
        ner_mapping = build_ner_mapping(ap)
        output_file_path = output_dir.joinpath(f'URM_{urm_split}')
        if urm_split == 'train':
            URM_ner = build_ner_urm(history_train, ap, user_id_mapping, ner_mapping, 'article_id_fixed')
        elif urm_split == 'validation':
            URM_ner = build_ner_urm(history_val, ap, user_id_mapping, ner_mapping, 'article_id_fixed')
        elif urm_split == 'test':
            URM_ner = build_ner_urm(behaviors_train.vstack(behaviors_val), ap, user_id_mapping, ner_mapping, 'article_ids_clicked')
        
        logging.info(f'Sparsity ratio of the URM: {compute_sparsity_ratio(URM_ner)}')
        save_sparse_csr(output_file_path, URM_ner, logger=logging)
    
    if urm_type == 'recsys':
        user_id_mapping_train = history_train.select('user_id').sort('user_id').unique().with_row_index().rename({'index': 'user_index'})
        user_id_mapping_val = history_val.select('user_id').sort('user_id').unique().with_row_index().rename({'index': 'user_index'})
        item_mapping = articles.select('article_id').sort('article_id').unique().with_row_index().rename({'index': 'item_index'})
        output_file_path = output_dir.joinpath(f'URM_{urm_split}')
        if urm_split == 'train':
            URM_recsys = build_recsys_urm(history_train, user_id_mapping_train, item_mapping)
        elif urm_split == 'validation':
            URM_recsys = build_recsys_urm(history_val, user_id_mapping_val, item_mapping)
        
        logging.info(f'Sparsity ratio of the URM: {compute_sparsity_ratio(URM_recsys)}')
        save_sparse_csr(output_file_path, URM_recsys, logger=logging)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Building URM matrixes")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the dataset is placed")
    parser.add_argument("-dataset_type", choices=['demo', 'small', 'large'], default='small', type=str,
                        help="Specify the type of dataset: ['demo', 'small', 'large']")
    parser.add_argument("-output_dir", default="../../urm/", type=str,
                        help="The directory where the URMs will be placed")
    parser.add_argument("-urm_split", choices=['train', 'validation', 'test'], default='train', type=str,
                        help="Specify the type of URM split: ['train', 'validation', 'test']")
    parser.add_argument("-urm_type", choices=['ner', 'recsys'], default='ner', type=str,
                        help="Specify the type of URM: ['ner', 'recsys']")
    
    args = parser.parse_args()
    DATASET_DIR = Path(args.dataset_path)
    DATASET_TYPE = args.dataset_type
    OUTPUT_DIR = Path(args.output_dir)
    URM_TYPE = args.urm_type
    URM_SPLIT = args.urm_split
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    DATASET_DIR = DATASET_DIR.joinpath(f'ebnerd_{DATASET_TYPE}')
    OUTPUT_DIR = OUTPUT_DIR.joinpath(URM_TYPE).joinpath(DATASET_TYPE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR.joinpath("log.txt")    
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_DIR, DATASET_TYPE, URM_TYPE, URM_SPLIT, OUTPUT_DIR)
