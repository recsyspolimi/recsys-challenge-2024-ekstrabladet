import os
import logging
from datetime import datetime
import argparse
import polars as pl

import sys
#sys.path.append('/home/ubuntu/RecSysChallenge2024/src')

from polimi.utils._embeddings import build_embeddings_similarity

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"

def main(input_path, output_dir, embedding_file, feature_name, test_path = None):
    logging.info("Starting to build the dataset")
    logging.info(f"Dataset path: {input_path}")
    
    dataset_types = ['train','test']
    
    for dataset_type in dataset_types:
        files_path = os.path.join(input_path, dataset_type)
        behaviors = pl.read_parquet(os.path.join(files_path, 'behaviors.parquet'))
        history = pl.read_parquet(os.path.join(files_path, 'history.parquet'))
        
        embeddings = pl.read_parquet(embedding_file)
        
        ds = behaviors.select(['impression_id','user_id','article_ids_inview'])\
                    .explode('article_ids_inview')\
                    .rename({'article_ids_inview': 'article'})
                    
        ds = build_embeddings_similarity(ds, history, embeddings, feature_name)
        ds.write_parquet( os.path.join(output_dir, dataset_types +'.parquet'))
        
    if not test_path is None:
        behaviors = pl.read_parquet(os.path.join(test_path, 'behaviors.parquet'))
        history = pl.read_parquet(os.path.join(test_path, 'history.parquet'))
        embeddings = pl.read_parquet(embedding_file)
        
        ds = behaviors.select(['impression_id','user_id','article_ids_inview'])\
                    .explode('article_ids_inview')\
                    .rename({'article_ids_inview': 'article'})
                    
        ds = build_embeddings_similarity(ds, history, embeddings, feature_name)
        ds.write_parquet( os.path.join(output_dir, 'test.parquet'))
        
    return


if __name__ == '__main__':
    print('here')
    parser = argparse.ArgumentParser(description="Training script for catboost")
    
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the dataset will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the dataset is placed")
    parser.add_argument("-test_path", default=None, type=str,
                        help="Directory where the test dataset is placed")
    parser.add_argument("-feature_name",default=None, type=str, required=True,
                        help="the name of the new feature")
    parser.add_argument("-emdeddings_file", default=None, type=str, required=True,
                        help="Path of the embeddings parquet file")
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    TEST_PATH = args.test_path
    FEATURE_NAME = args.feature_name
    EMBEDDINGS_FILE = args.emdeddings_file
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_name = f'preprocessing_{FEATURE_NAME}_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR,out_name )
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_DIR, output_dir, EMBEDDINGS_FILE, FEATURE_NAME, TEST_PATH)