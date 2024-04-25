try:
    import polars as pl
except ImportError:
    print("polars not available")

import scipy.sparse as sps
from tqdm import tqdm
import numpy as np
from polimi.utils._catboost import reduce_polars_df_memory_size


"""
Utils for building URM tables.
""" 
BATCH_SIZE=70000
def build_articles_with_processed_ner(articles: pl.DataFrame):        
        return articles.select('article_id', 'ner_clusters')\
            .with_columns(pl.col('ner_clusters').list.eval(pl.element().str.strip_chars_start('\"')))\
            .with_columns(pl.col('ner_clusters').list.eval(pl.element().str.strip_chars(' ')))\
            .with_columns(pl.col('ner_clusters').list.eval(pl.element().str.to_lowercase()))\
            .with_columns(pl.col('ner_clusters').list.eval(pl.element().filter(pl.element().str.len_chars() > 0)))\
            .with_columns(pl.col('ner_clusters').list.drop_nulls())\
            .with_columns(pl.col('ner_clusters').list.unique())\
            .with_columns(pl.col('ner_clusters').list.sort())\
            .sort('article_id')\
            .set_sorted('article_id')

def compute_sparsity_ratio(URM: sps.csr_matrix):
    total_elements = URM.shape[0] * URM.shape[1]
    nonzero_elements = URM.count_nonzero()
    sparsity_ratio = 1 - (nonzero_elements / total_elements)
    return sparsity_ratio


def _build_implicit_urm(df: pl.DataFrame, x_col: str, y_col: str, x_mapping: pl.DataFrame, y_mapping: pl.DataFrame):
    return sps.csr_matrix((np.ones(df.shape[0]),
                          (df[x_col].to_numpy(), df[y_col].to_numpy())),
                         shape=(x_mapping.shape[0], y_mapping.shape[0]))
    
def build_user_id_mapping(history: pl.DataFrame):
    return history.select('user_id')\
        .unique('user_id') \
        .cast(pl.UInt32)\
        .drop_nulls() \
        .sort('user_id') \
        .with_row_index() \
        .select(['index', 'user_id'])\
        .rename({'index': 'user_index'})        
        
def build_ner_mapping(articles: pl.DataFrame):
    return articles\
        .select('ner_clusters')\
        .explode('ner_clusters') \
        .rename({'ner_clusters': 'ner'})\
        .unique('ner')\
        .drop_nulls()\
        .sort('ner')\
        .with_row_index()\
        .rename({'index': 'ner_index'})     
     
def _build_batch_ner_interactions(df: pl.DataFrame, 
                  articles: pl.DataFrame, 
                  user_id_mapping: pl.DataFrame,
                  ner_mapping: pl.DataFrame,
                  articles_id_col = 'article_id_fixed',
                  batch_size=BATCH_SIZE):
    
    articles_ner_index = articles\
        .with_columns(pl.col('ner_clusters').list.eval(pl.element().replace(ner_mapping['ner'], ner_mapping['ner_index'], default=None).cast(pl.UInt32)).list.drop_nulls())
    
    df = df.select('user_id', articles_id_col)\
        .group_by('user_id')\
        .agg(pl.col(articles_id_col).flatten().unique())
        
    df = pl.concat([
        slice.with_columns(
            pl.col(articles_id_col).list.eval(pl.element().replace(articles_ner_index['article_id'], articles_ner_index['ner_clusters'], default=None).cast(pl.List(pl.UInt32)))\
                .list.eval(pl.element().flatten())\
                .list.drop_nulls()\
                .list.unique()\
                .list.sort()\
        ).rename({articles_id_col: 'ner_index'})\
        .filter(pl.col('ner_index').list.len() > 0)\
        .with_columns(pl.col('user_id').replace(user_id_mapping['user_id'], user_id_mapping['user_index']).cast(pl.UInt32))
        for slice in tqdm(df.iter_slices(batch_size), total=df.shape[0]//batch_size)
    ]).rename({'user_id': 'user_index'})\
    .sort('user_index')\
    .explode('ner_index')
        
    return reduce_polars_df_memory_size(df)
    
    

def build_ner_urm(history: pl.DataFrame, 
                  articles: pl.DataFrame, 
                  user_id_mapping: pl.DataFrame,
                  ner_mapping: pl.DataFrame,
                  articles_id_col = 'article_id_fixed',
                  batch_size=BATCH_SIZE):
        
    ner_interactions = _build_batch_ner_interactions(history, articles, user_id_mapping, ner_mapping, articles_id_col, batch_size=batch_size)
    return _build_implicit_urm(ner_interactions, 'user_index', 'ner_index', user_id_mapping, ner_mapping)






def build_recsys_urm(history: pl.DataFrame,
                     user_id_mapping: pl.DataFrame,
                     item_mapping: pl.DataFrame
                    ):
    interactions = reduce_polars_df_memory_size(history.select('user_id','article_id_fixed').explode('article_id_fixed').unique().join(user_id_mapping,on='user_id').join(item_mapping,left_on='article_id_fixed',right_on='article_id').unique(['user_index','item_index']).rename({'article_id_fixed':'article_id'}))
    return sps.csr_matrix((np.ones(interactions.shape[0]),
                          (interactions['user_index'].to_numpy(), interactions['item_index'].to_numpy())),
                         shape=(user_id_mapping.shape[0], item_mapping.shape[0]))
