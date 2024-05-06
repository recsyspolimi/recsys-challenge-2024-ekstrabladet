import polars as pl
from polimi.utils._catboost import _preprocessing_article_endorsement_feature
from tqdm import tqdm
import numpy as np
import logging
from multiprocessing import Pool
import time
from polimi.utils._embeddings import fast_distance
import simsimd
from polimi.utils._polars import reduce_polars_df_memory_size
import multiprocessing
from polimi.utils._embeddings import get_distance_function

pool = multiprocessing.Pool()

from multiprocessing import cpu_count
from polimi.utils._embeddings import (
    build_weighted_timestamps_embeddings,
    build_weighted_SP_embeddings,
    build_weighted_readtime_embeddings,
)
import pandas as pd
WEIGHTED_EMB_FUNCTIONS = [
    build_weighted_timestamps_embeddings,
    build_weighted_SP_embeddings,
    build_weighted_readtime_embeddings,
]
cores = cpu_count() #Number of CPU cores on your system
partitions = cores - 2  #Define as many partitions as you want

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
log_path = '/home/ubuntu/tmp/log.txt'



def dist_func(x):
    return simsimd.cosine(np.asarray(x[:768]), np.asarray(x[768:]))

def parallelized_distance(data, function):
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(function, data_split))
    pool.close()
    pool.join()
    return data


def compute_multiple_weighted_embeddings(train_ds, history, embedding, name):
    users_embeddings = []
    column_names = []
    new_col_names = []

    for fun in range(len(WEIGHTED_EMB_FUNCTIONS)):
        logging.info(f'Weighted function N :{fun}')
        users_embedding, column_name, new_col_name = WEIGHTED_EMB_FUNCTIONS[fun](
            train_ds, history, embedding, name)
        users_embeddings.append(users_embedding)
        column_names.append(column_name)
        new_col_names.append(new_col_name)

    users_embeddings = users_embeddings[0].join(users_embeddings[1], on='user_id', how='left').join(
        users_embeddings[2], on='user_id', how='left')
    return users_embeddings, column_names, new_col_names


def iterator_weighted_embedding(train_ds, column_name, new_column_name, distance_function ,n_batches=100):

    for sliced_df in train_ds.iter_slices(train_ds.shape[0] // n_batches):

        users_embeddings_col = sliced_df.select(column_name).with_columns(pl.col(column_name).list.to_struct()).unnest(column_name)\
            .to_numpy(order='c').astype(np.float32)
        item_embedding_col = sliced_df.select('item_embedding').with_columns(pl.col("item_embedding").list.to_struct()).unnest("item_embedding")\
            .to_numpy(order='c').astype(np.float32)

        col = np.hstack((users_embeddings_col, item_embedding_col))

        out = pool.map(distance_function, col)
        # out = map(get_distance_function(emb_len), col)
        new_col = pl.Series(new_column_name, list(out))
        sliced_df = sliced_df.hstack([new_col])
        
        yield reduce_polars_df_memory_size(sliced_df.select(['user_id', 'article', new_column_name]), verbose=False)
    

if __name__ == '__main__':
    start = time.time()
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    emb = [  #'/home/ubuntu/dataset/distilbert_title_embedding.parquet',
             '/home/ubuntu/dataset/google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet',
            '/home/ubuntu/dataset/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet',
        '/home/ubuntu/dataset/Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet',
        '/home/ubuntu/dataset/Ekstra_Bladet_word2vec/document_vector.parquet']
        #'/home/ubuntu/dataset/emotions_embedding.parquet']

    #'distilbert', 'bert', 'roberta',
    names = ['bert', 'roberta','contrastive', 'w2v'] #,'emotion']

    types = ['validation']

    for type in types:
        logging.info(f'Preprocessing {type}')
        train_ds = pl.read_parquet(
            f'/home/ubuntu/dataset/ebnerd_large/{type}/behaviors.parquet')
        history = pl.read_parquet(
            f'/home/ubuntu/dataset/ebnerd_large/{type}/history.parquet')

        articles = pl.read_parquet(
            '/home/ubuntu/dataset/ebnerd_large/articles.parquet')

        train_ds = train_ds.select(['impression_id', 'user_id', 'article_ids_inview'])\
            .explode('article_ids_inview')\
            .rename({'article_ids_inview': 'article'}).unique(subset=['user_id', 'article'])
        
        output_ds = train_ds.select(['user_id', 'article'])
            
        output_dir = f'/home/ubuntu/tmp/{type}_ds.parquet'

        for em in range(len(emb)):
            logging.info(f'Embedding {names[em]}')
            embedding = pl.read_parquet(emb[em])
            users_embeddings, column_names, new_col_names = compute_multiple_weighted_embeddings(
                train_ds, history, embedding, names[em])

            print('here_0')
            embedding = embedding.rename(
                {embedding.columns[0]: 'article', embedding.columns[1]: 'item_embedding'})
            
            print(train_ds.columns)
            print(embedding.columns)
            ds = pl.DataFrame()
            
            for rows in tqdm(train_ds.iter_slices(10000), total=train_ds.shape[0] // 10000):
                ds = ds.vstack( 
                        rows.join(users_embeddings, on='user_id', how='left') \
                            .join(embedding, on='article', how='left'))
            
            train_ds = ds
            
            start = time.time()
            emb_len = len(embedding.select('item_embedding').limit(1).item())
            print(train_ds)
            n_batches = 1000
            output_ds = train_ds
            for col in range(len(column_names)):
                logging.info(f'Computing distance for {column_names[col]}')
                new_ds = []
                count = 0
                for slice in tqdm(iterator_weighted_embedding(train_ds, column_names[col], new_col_names[col], get_distance_function(emb_len), n_batches=n_batches), total= n_batches):
                    new_ds.append(slice)
                    logging.info(f'Slice {count+1} preprocessed.')
                    count += 1
                new_ds = reduce_polars_df_memory_size(pl.concat(new_ds, how='vertical_relaxed'))
                output_ds = output_ds.join(new_ds, on=['user_id', 'article'], how='left')
            
            output_ds.write_parquet(output_dir)

    end = time.time()
    print(end - start)

        # start = time.time()
        # train_ds = train_ds.with_columns(
        #                 pl.struct(['user_embedding_weighted_TS', 'contrastive_vector']).map_elements(
        #                 lambda x: fast_distance(x['user_embedding_weighted_TS'], x['contrastive_vector']), return_dtype=pl.Float32).cast(pl.Float32).alias('new_col')
        #         )
        # end = time.time()
        # print(end - start)

        # distance_function = get_distance_function(emb_len)
        # out = np.apply_along_axis(distance_function, 1, col)
