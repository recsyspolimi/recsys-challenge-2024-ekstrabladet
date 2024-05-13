import os
import logging
from datetime import datetime
import argparse
import polars as pl
import gc
from pathlib import Path
import math
from tqdm import tqdm
import time
from polimi.utils._polars import reduce_polars_df_memory_size, stack_slices
from polimi.utils._embeddings import weight_scores, build_normalized_embeddings_matrix, build_embeddings_scores, build_embeddings_agg_scores, build_history_w

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def main(input_path, output_dir):
    logging.info("Starting to build the dataset")
    logging.info(f"Dataset path: {input_path}")
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    dtype = str(input_path).split('/')[-1]
    if dtype == 'ebnerd_testset':
        dataset_splits = ['test']
    else:
        dataset_splits = ['train', 'validation']
    
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
    
    for data_split in dataset_splits:
        logging.info(f"Loading {input_path / data_split}")
        files_path = input_path / data_split
        behaviors = pl.read_parquet(files_path / 'behaviors.parquet')
        history = pl.read_parquet(files_path / 'history.parquet')
        history_w = build_history_w(history, articles)
        selected_weight_col = 'read_time_fixed_article_len_ratio_l1_w'
        history_w = history_w.select('user_id', selected_weight_col)
        
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
            
        save_path = output_dir / dtype / data_split
        save_path.mkdir(parents=True, exist_ok=True)
         
        train_ds = reduce_polars_df_memory_size(train_ds)
        train_ds = train_ds.sort('user_id')
        BATCH_SIZE = int(5e5)
        n_slices = math.ceil(len(train_ds) / BATCH_SIZE)
        for i, slice in enumerate(tqdm(train_ds.iter_slices(BATCH_SIZE), total=n_slices)):
            logging.info(f'Start building embeddings scores slice {i}...')
            t = time.time()
            slice = build_embeddings_scores(slice, history_m=history_m, m_dict=norm_m_dict)
            logging.info(f'Completed in {((time.time() - t) / 60):.2f} minutes')
            
            logging.info(f'Weightening embeddings scores for slice {i} ...')
            t = time.time()
            slice = slice.join(history_w, on='user_id', how='left')
            weights_cols = [col for col in slice.columns if col.endswith('_l1_w')]
            scores_cols = [col for col in slice.columns if col.endswith('_scores')]
            slice = weight_scores(slice, scores_cols=scores_cols, weights_cols=weights_cols)    
            logging.info(f'Completed in {((time.time() - t) / 60):.2f} minutes')
     
            slice = reduce_polars_df_memory_size(slice)
            logging.info(f'Building embeddings scores slice {i} aggregations...')
            agg_cols = [col for col in slice.columns if '_scores' in col]
            slice = build_embeddings_agg_scores(slice, agg_cols=agg_cols, last_k=[25, 50, 100])
            logging.info(f'Completed in {((time.time() - t) / 60):.2f} minutes')
            
            logging.info(f'Saving embeddings scores slice {i} aggregations to {save_path}')
            slice = reduce_polars_df_memory_size(slice)
            slice.write_parquet(save_path / f'embeddings_scores_agg_slice_{i}.parquet')
        
        t = time.time()
        agg_scores_slices_paths = list(save_path.glob('embeddings_scores_agg_slice_*.parquet'))
        stack_slices(agg_scores_slices_paths, save_path, save_name=f'agg_embeddings_scores_{selected_weight_col}', delete_all_slices=True)
        logging.info(f'Completed in {((time.time() - t) / 60):.2f} minutes')

        
        
        
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
