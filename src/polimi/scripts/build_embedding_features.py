from polimi.utils._embeddings import _build_user_embeddings
from polimi.utils._embeddings import iterator_build_embeddings_similarity
import os
import logging
from datetime import datetime
import argparse
import polars as pl
import gc

import sys
sys.path.append('/home/ubuntu/RecSysChallenge2024/src')


LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def main(input_path, output_dir, embedding_file, feature_name, test_path=None):
    logging.info("Starting to build the dataset")
    logging.info(f"Dataset path: {input_path}")

    dataset_types = ['train', 'validation']

    for data_type in dataset_types:

        logging.info(f"Loading {data_type}")
        files_path = os.path.join(input_path, data_type)
        behaviors = pl.read_parquet(
            os.path.join(files_path, 'behaviors.parquet'))
        history = pl.read_parquet(os.path.join(files_path, 'history.parquet'))

        embeddings = pl.read_parquet(embedding_file)

        embeddings = embeddings.rename(
            {embeddings.columns[0]: 'article_id', embeddings.columns[1]: 'item_embedding'})

        ds = behaviors.select(['impression_id', 'user_id', 'article_ids_inview'])\
            .explode('article_ids_inview')\
            .rename({'article_ids_inview': 'article'})

        logging.info(f"Building similarities for {data_type}")

        users_embeddings = _build_user_embeddings(history, embeddings)

        dataset_complete = []
        i = 0
        for dataset in iterator_build_embeddings_similarity(ds, users_embeddings, embeddings, feature_name, n_batches=100):
            dataset_complete.append(dataset)
            logging.info(f'Slice {i+1} preprocessed.')
            i += 1

        dataset_complete = pl.concat(dataset_complete, how='vertical_relaxed')
        dataset_complete = dataset_complete.unique(
            subset=['user_id', 'article'])
        ds = ds.join(dataset_complete, on=[
                     'user_id', 'article'], how='left').fill_null(value=0)
        ds.write_parquet(os.path.join(output_dir, data_type + '.parquet'))

        del behaviors, history, embeddings, dataset_complete, ds, users_embeddings
        gc.collect()

    if not test_path is None:
        logging.info(f"Dataset path: {test_path}")
        logging.info(f"Loading Test")
        test_path = test_path + '/test'
        behaviors = pl.read_parquet(
            os.path.join(test_path, 'behaviors.parquet'))
        history = pl.read_parquet(os.path.join(test_path, 'history.parquet'))
        embeddings = pl.read_parquet(embedding_file)

        embeddings = pl.read_parquet(embedding_file)

        embeddings = embeddings.rename(
            {embeddings.columns[0]: 'article_id', embeddings.columns[1]: 'item_embedding'})

        ds = behaviors.select(['impression_id', 'user_id', 'article_ids_inview'])\
            .explode('article_ids_inview')\
            .rename({'article_ids_inview': 'article'})

        logging.info(f"Building similarities for Test")

        users_embeddings = build_user_embeddings(history, embeddings)

        dataset_complete = []
        i = 0
        for dataset in iterator_build_embeddings_similarity(ds, users_embeddings, embeddings, feature_name, n_batches=100):
            dataset_complete.append(dataset)
            logging.info(f'Slice {i+1} preprocessed.')
            i += 1

        dataset_complete = pl.concat(dataset_complete, how='vertical_relaxed')
        dataset_complete = dataset_complete.unique(
            subset=['user_id', 'article'])
        ds = ds.join(dataset_complete, on=[
                     'user_id', 'article'], how='left').fill_null(value=0)
        ds.write_parquet(os.path.join(output_dir, 'test.parquet'))

        del behaviors, history, embeddings, dataset_complete, ds, users_embeddings
        gc.collect()

    return


if __name__ == '__main__':
    print('here')
    parser = argparse.ArgumentParser(
        description="Training script for catboost")

    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the dataset will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the dataset is placed")
    parser.add_argument("-test_path", default=None, type=str,
                        help="Directory where the test dataset is placed")
    parser.add_argument("-feature_name", default=None, type=str, required=True,
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
    output_dir = os.path.join(OUTPUT_DIR, out_name)
    os.makedirs(output_dir)

    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w",
                        format=LOGGING_FORMATTER, level=logging.INFO, force=True)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)

    main(DATASET_DIR, output_dir, EMBEDDINGS_FILE, FEATURE_NAME, TEST_PATH)
