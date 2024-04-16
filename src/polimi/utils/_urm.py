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

_BATCH_SIZE = 100000
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
        .drop_nulls() \
        .sort('user_id') \
        .with_row_index() \
        .select(['index', 'user_id'])\
        .rename({'index': 'user_index'})
        
        
def build_ner_mapping(articles: pl.DataFrame):
    return articles\
        .select('article_id', 'ner_clusters')\
        .explode('ner_clusters') \
        .rename({'ner_clusters': 'ner'})\
        .unique('ner')\
        .drop_nulls()\
        .sort('ner')\
        .with_row_index()\
        .drop('article_id')\
        .rename({'index': 'ner_index'})
     
     
def _build_batch_ner_interactions(df: pl.DataFrame, 
                  articles: pl.DataFrame, 
                  user_id_mapping: pl.DataFrame,
                  ner_mapping: pl.DataFrame,
                  articles_id_col = 'article_id_fixed',
                  batch_size=_BATCH_SIZE):
    
    articles_ner_index = articles\
        .with_columns(pl.col('ner_clusters').list.eval(pl.element().replace(ner_mapping['ner'], ner_mapping['ner_index']).cast(pl.Int32)))

    df = pl.concat([
        slice_df.select('user_id', articles_id_col)\
            .group_by('user_id')\
            .agg(pl.col(articles_id_col).flatten())\
            .with_columns(
                pl.col(articles_id_col).list.eval(pl.element().replace(articles_ner_index['article_id'], articles_ner_index['ner_clusters']).cast(pl.List(pl.Int32)))\
                    .list.eval(pl.element().flatten())\
                    .list.drop_nulls()\
                    .list.unique()\
                    .list.sort()\
            ).rename({articles_id_col: 'ner_index'})\
            .with_columns(pl.col('user_id').replace(user_id_mapping['user_id'], user_id_mapping['user_index']).cast(pl.Int32))\
            .rename({'user_id': 'user_index'})\
        for slice_df in tqdm(df.iter_slices(batch_size), total=df.shape[0]//batch_size)
        ]).sort('user_index').explode('ner_index')
    
    return reduce_polars_df_memory_size(df)
    
    

def build_ner_urm(history: pl.DataFrame, 
                  articles: pl.DataFrame, 
                  user_id_mapping: pl.DataFrame,
                  ner_mapping: pl.DataFrame,
                  articles_id_col = 'article_id_fixed',
                 batch_size=_BATCH_SIZE):
        
    ner_interactions = _build_batch_ner_interactions(history, articles, user_id_mapping, ner_mapping, articles_id_col, batch_size=batch_size)
    return _build_implicit_urm(ner_interactions, 'user_index', 'ner_index', user_id_mapping, ner_mapping)