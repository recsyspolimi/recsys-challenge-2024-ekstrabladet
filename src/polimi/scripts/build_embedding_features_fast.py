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
from polimi.utils._embeddings import _build_user_embeddings
from polimi.utils._embeddings import iterator_build_embeddings_similarity, build_normalized_embeddings_matrix, build_embeddings_scores, build_embeddings_agg_scores, build_embeddings_scores_test

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def main(input_path, output_dir):
    logging.info("Starting to build the dataset")
    logging.info(f"Dataset path: {input_path}")
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if str(input_path).split('/')[-1] == 'ebnerd_testset':
        dataset_types = ['test']
    else:
        dataset_types = ['train', 'validation']
    
    emb_name_dict = {'Ekstra_Bladet_contrastive_vector': 'contrastive_vector',
                 'FacebookAI_xlm_roberta_base': 'xlm_roberta_base',
                 'Ekstra_Bladet_image_embeddings': 'image_embeddings',
                 'google_bert_base_multilingual_cased': 'bert_base_multilingual_cased'}
    
    articles = pl.read_parquet(input_path / 'articles.parquet')    
    norm_m_dict = {}
    article_emb_mapping = articles.select('article_id').unique().with_row_index()
    for dir, file_name in emb_name_dict.items():
        logging.info(f'Processing {file_name} embedding matrix...')
        emb_df = pl.read_parquet(input_path.parent / dir / f'{file_name}.parquet')
        emb_df.columns = ['article_id', 'embedding']
        logging.info(f'Building normalized embeddings matrix for {file_name}...')
        m = build_normalized_embeddings_matrix(emb_df, article_emb_mapping)
        norm_m_dict[file_name] = m
    
    for data_type in dataset_types:
        logging.info(f"Loading {data_type}")
        files_path = input_path / data_type
        behaviors = pl.read_parquet(files_path / 'behaviors.parquet')
        history = pl.read_parquet(files_path / 'history.parquet')
        
        history_m = history\
            .select('user_id', pl.col('article_id_fixed').list.eval(
                        pl.element().replace(article_emb_mapping['article_id'], article_emb_mapping['index'], default=None)))\
            .with_row_index('user_index')

        user_history_map = history_m.select('user_id', 'user_index')
        history_m = history_m['article_id_fixed'].to_numpy()
        train_ds = behaviors.select('impression_id', 'user_id', pl.col('article_ids_inview').alias('article'))\
            .join(user_history_map, on='user_id')\
            .with_columns(
                pl.col('article').list.eval(pl.element().replace(article_emb_mapping['article_id'], article_emb_mapping['index'], default=None)).name.suffix('_index'),
            ).drop('impression_time_fixed', 'scroll_percentage_fixed', 'read_time_fixed')
            
        save_path = output_dir / data_type
        save_path.mkdir(parents=True, exist_ok=True)
         
        train_ds = reduce_polars_df_memory_size(train_ds)
        train_ds = train_ds.sort('user_id')
        BATCH_SIZE = 1e6
        slices = math.ceil(len(train_ds) / BATCH_SIZE)
        for i, slice in enumerate(tqdm(train_ds.iter_slices(BATCH_SIZE), total=slices)):
            logging.info(f'Starting embeddings scores slice {i}...')
            slice = build_embeddings_scores(slice, history_m, m_dict=norm_m_dict)
            logging.info(f'Saving embeddings scores plain slice {i} to {save_path}...')
            slice.write_parquet(save_path / f'embeddings_scores_slice_{i}.parquet')
                
        for i in range(0, slices):
            slice = pl.read_parquet(save_path / f'embeddings_scores_slice_{i}.parquet')
            logging.info(f'Building embeddings scores slice {i} standard aggregations...')
            slice = build_embeddings_agg_scores(slice, history, emb_names=emb_name_dict.values())
            logging.info(f'Saving embeddings scores slice {i} aggregations to {save_path}')
            slice.write_parquet(save_path / f'embeddings_scores_agg_slice_{i}.parquet')
        
        
        scores_slices_paths = list(save_path.glob('embeddings_scores_slice_*.parquet'))
        stack_slices(scores_slices_paths, save_path, 'embeddings_scores.parquet', delete_all_slices=True)
        
        agg_scores_slices_paths = list(save_path.glob('embeddings_scores_agg_slice_*.parquet'))
        stack_slices(agg_scores_slices_paths, save_path, 'agg_embeddings_scores.parquet', delete_all_slices=True)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Training script for generating embeddings scores for the dataset")

    parser.add_argument("-output_dir", default="~/experiments", type=str,
                        help="The directory where the dataset will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the dataset is placed")

    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_name = f'preprocessing_embedding_scores_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, out_name)
    os.makedirs(output_dir)

    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w",
                        format=LOGGING_FORMATTER, level=logging.INFO, force=True)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)

    main(DATASET_DIR, output_dir)
