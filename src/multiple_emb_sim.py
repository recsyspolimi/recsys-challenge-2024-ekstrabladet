import polars as pl
from polimi.utils._catboost import _preprocessing_article_endorsement_feature
from tqdm import tqdm
import logging
from polimi.utils._embeddings import fast_distance

from polimi.utils._embeddings import (
    build_weighted_timestamps_embeddings,
    build_weighted_SP_embeddings,
    build_weighted_readtime_embeddings,
    iterator_weighted_embedding
    )

from polimi.utils._polars import reduce_polars_df_memory_size
import gc

WEIGHTED_EMB_FUNCTIONS = [
    build_weighted_timestamps_embeddings,
    build_weighted_SP_embeddings,
    build_weighted_readtime_embeddings,
]

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
log_path = '/home/ubuntu/tmp/log.txt'


def batch_loop(ds, embeddings, users_embeddings, column_names, new_col_names, output_dir, logging):
    dataset_complete = []
    i = 0
    embeddings = embeddings.rename({embeddings.columns[0]: 'article_id', embeddings.columns[1]: 'item_embedding'})
    
    for dataset in iterator_weighted_embedding(ds, users_embeddings, embeddings, column_names, new_col_names, n_batches=200):
        dataset_complete.append(dataset)
        # print(f'Slice {i+1} preprocessed.')
        if i % 20 == 0:
            logging.info(f'Slice {i+1} preprocessed.')
        i += 1

    dataset_complete = pl.concat(dataset_complete, how='vertical_relaxed')
    dataset_complete = dataset_complete.unique(subset=['user_id', 'article'])
    ds = ds.with_columns(pl.col('user_id').cast(pl.UInt32),
                         pl.col('article').cast(pl.Int32)).join(dataset_complete, on=[
                     'user_id', 'article'], how='left').fill_null(value=0)
    print(ds)
    ds.write_parquet(output_dir)
    return ds
        
        
def compute_multiple_weighted_embeddings(train_ds, history, embedding, name):
    users_embeddings = []
    column_names = []
    new_col_names = []
    
    for fun in range(len(WEIGHTED_EMB_FUNCTIONS)):
                logging.info(f'Weighted function N :{fun}')
                users_embedding, column_name, new_col_name = WEIGHTED_EMB_FUNCTIONS[fun](train_ds, history, embedding, name)
                users_embeddings.append(users_embedding)
                column_names.append(column_name)
                new_col_names.append(new_col_name)
    
    users_embeddings = users_embeddings[0].join(users_embeddings[1], on='user_id', how='left').join(users_embeddings[2], on='user_id', how='left')
    return users_embeddings, column_names, new_col_names


def compute_similarity_parallel(df, users_embeddings, embeddings, column_names, new_col_names):
    
    df = df.with_columns(pl.col('user_id').cast(pl.UInt32),
                         pl.col('article').cast(pl.Int32))
    user_item_emb_similarity = pl.concat(
                rows.select(['user_id','article']).join(embeddings, left_on = 'article', right_on = 'article_id')\
                .join(users_embeddings, on = 'user_id')\
                .with_columns(
                     pl.struct([column_names[0], 'item_embedding']).map_elements(
                                        lambda x: fast_distance(x[column_names[0]], x['item_embedding']), return_dtype=pl.Float32).cast(pl.Float32).alias(new_col_names[0]),
                     pl.struct([column_names[1], 'item_embedding']).map_elements(
                                        lambda x: fast_distance(x[column_names[1]], x['item_embedding']), return_dtype=pl.Float32).cast(pl.Float32).alias(new_col_names[1]),
                     pl.struct([column_names[2], 'item_embedding']).map_elements(
                                        lambda x: fast_distance(x[column_names[2]], x['item_embedding']), return_dtype=pl.Float32).cast(pl.Float32).alias(new_col_names[2])
                ).drop(['item_embedding_right','user_embedding_right'])
            for rows in tqdm(df.iter_slices(1000), total=df.shape[0] // 1000)).unique(subset=['user_id', 'article'])
    return user_item_emb_similarity.select(['user_id','article', new_col_names[0], new_col_names[1], new_col_names[2]])
  

def iterator_weighted_embedding(df, users_embeddings, embeddings, column_names, new_col_names, n_batches: int = 10):

    for sliced_df in df.iter_slices(df.shape[0] // n_batches):
        slice_features = sliced_df.pipe(compute_similarity_parallel, users_embeddings=users_embeddings,
                                        embeddings=embeddings, column_names= column_names, new_col_names=new_col_names)
    
        yield slice_features
                
if __name__ == '__main__': 
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    emb = [# '/home/ubuntu/dataset/distilbert_title_embedding.parquet',
           # '/home/ubuntu/dataset/google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet',
           # '/home/ubuntu/dataset/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet',
           '/home/ubuntu/dataset/Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet',
           '/home/ubuntu/dataset/Ekstra_Bladet_word2vec/document_vector.parquet',
           '/home/ubuntu/dataset/emotions_embedding.parquet']
    
    names = [ 'contrastive', 'w2v', 'emotion'] #'distilbert', 'bert', 'roberta',
    
    types = ['train', 'validation']
    
    for type in types:
        logging.info(f'Preprocessing {type}')
        train_ds = pl.read_parquet(f'/home/ubuntu/dataset/ebnerd_large/{type}/behaviors.parquet')
        history = pl.read_parquet(f'/home/ubuntu/dataset/ebnerd_large/{type}/history.parquet')
        train_ds = train_ds.select(['impression_id', 'user_id', 'article_ids_inview'])\
                .explode('article_ids_inview')\
                .rename({'article_ids_inview': 'article'})
                
        articles = pl.read_parquet('/home/ubuntu/dataset/ebnerd_large/articles.parquet')
        
        output_dir = f'/home/ubuntu/tmp/{type}_ds.parquet'
        
        for i in range(len(emb)):
            logging.info(f'Embedding {names[i]}')
            embedding = pl.read_parquet(emb[i])
            users_embeddings, column_names, new_col_names = compute_multiple_weighted_embeddings(train_ds, history, embedding, names[i])
            train_ds = batch_loop(ds=train_ds,  embeddings=embedding, users_embeddings=users_embeddings, 
                                  column_names=column_names, new_col_names=new_col_names, output_dir=output_dir, logging=logging)
          
        del train_ds, history, articles
        gc.collect()