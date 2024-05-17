try:
    import polars as pl
except ImportError:
    print("polars not available")

from pathlib import Path
import scipy.sparse as sps
from tqdm import tqdm
import numpy as np
from RecSys_Course_AT_PoliMi.Recommenders.BaseRecommender import BaseRecommender
from polimi.utils._polars import reduce_polars_df_memory_size


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
    return sps.csr_matrix((np.ones(df.shape[0]), (df[x_col].to_numpy(), df[y_col].to_numpy())),
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
        .rename({'index': 'ner_index'})\
        .cast({'ner_index': pl.UInt32})


def build_item_mapping(articles: pl.DataFrame):
    return articles\
        .select('article_id')\
        .unique()\
        .drop_nulls()\
        .sort('article_id')\
        .with_row_index()\
        .rename({'index': 'item_index'})

def _build_batch_ner_interactions(df: pl.DataFrame, 
                  articles: pl.DataFrame, 
                  user_id_mapping: pl.DataFrame,
                  ner_mapping: pl.DataFrame,
                  articles_id_col = 'article_id_fixed',
                  batch_size=BATCH_SIZE):
    
    articles_ner_index = articles\
        .with_columns(pl.col('ner_clusters').list.eval(pl.element().replace(ner_mapping['ner'], ner_mapping['ner_index'], default=None, return_dtype=pl.UInt32)).list.drop_nulls())
    
    df = df.select('user_id', articles_id_col)\
        .group_by('user_id')\
        .agg(pl.col(articles_id_col).flatten().unique())
        
    df = pl.concat([
        slice.with_columns(
            pl.col(articles_id_col).list.eval(pl.element().replace(articles_ner_index['article_id'], articles_ner_index['ner_clusters'], default=None, return_dtype=pl.List(pl.UInt32)))\
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
    
    

def build_ner_urm(df: pl.DataFrame, 
                  articles: pl.DataFrame, 
                  user_id_mapping: pl.DataFrame,
                  ner_mapping: pl.DataFrame,
                  articles_id_col = 'article_id_fixed',
                  batch_size=BATCH_SIZE):
        
    ner_interactions = _build_batch_ner_interactions(df, articles, user_id_mapping, ner_mapping, articles_id_col, batch_size=batch_size)
    return _build_implicit_urm(ner_interactions, 'user_index', 'ner_index', user_id_mapping, ner_mapping)

def _build_recsys_interactions(
                history: pl.DataFrame,
                user_id_mapping: pl.DataFrame,
                item_mapping: pl.DataFrame,
                articles_id_col = 'article_id_fixed'):
    
    df = history\
            .select('user_id',articles_id_col)\
            .explode(articles_id_col)\
            .unique()\
            .join(user_id_mapping,on='user_id')\
            .join(item_mapping,left_on=articles_id_col,right_on='article_id')\
            .unique(['user_index','item_index'])\
            .rename({articles_id_col:'article_id'})
    
    return reduce_polars_df_memory_size(df)
       

def build_recsys_urm(history: pl.DataFrame,
                     user_id_mapping: pl.DataFrame,
                     item_mapping: pl.DataFrame,
                     articles_id_col = 'article_id_fixed'
                    ):
    recsys_interactions = _build_recsys_interactions(history, user_id_mapping, item_mapping, articles_id_col)
    return _build_implicit_urm(recsys_interactions,'user_index', 'item_index', user_id_mapping, item_mapping)
        

def train_recommender(URM: sps.csr_matrix, recommender: BaseRecommender, params: dict, file_name:str = None, output_dir: Path = None):
    rec_instance= recommender(URM)
    rec_instance.fit(**params)

    if output_dir:
        rec_instance.save_model(folder_path=str(output_dir), file_name=file_name)

    return rec_instance


def load_recommender(URM: sps.csr_matrix, recommender: BaseRecommender, file_path: Path, file_name: str=None):
    rec_instance = recommender(URM)
    rec_instance.load_model(folder_path=str(file_path), file_name=file_name)
    return rec_instance

def build_urm_ner_score_features(df: pl.DataFrame, ner_mapping: pl.DataFrame, recs: list[BaseRecommender], batch_size=BATCH_SIZE):
    df = pl.concat([ #remove caidates with empty ner_index list
        slice.explode(['candidate_ids', 'candidate_ner_index'])\
            .filter(pl.col('candidate_ner_index').list.len() > 0)\
            .group_by(['impression_id', 'user_id', 'user_index']).agg(pl.all())
        for slice in tqdm(df.iter_slices(batch_size), total=df.shape[0]//batch_size)
    ])
    
    all_items = ner_mapping['ner_index'].unique().sort().to_list()
    df = pl.concat([
            slice.with_columns(
                    *[pl.col('candidate_ner_index').list.eval(
                        pl.element().list.eval(pl.element().replace(all_items, rec._compute_item_score(user_index, all_items)[0, all_items], default=None)).cast(pl.List(pl.Float32))
                    ).alias(f"{rec.RECOMMENDER_NAME}_ner_scores") for rec in recs]
                ).drop('user_index', 'candidate_ner_index')
        for user_index, slice in tqdm(df.partition_by(['user_index'], as_dict=True).items(), total=df['user_index'].n_unique())
        ])
    
    scores_cols = [col for col in df.columns if '_ner_scores' in col]
    df = df.with_columns(
            *[pl.col(col).list.eval(pl.element().list.sum()).alias(f'sum_{col}') for col in scores_cols],
            *[pl.col(col).list.eval(pl.element().list.max()).alias(f'max_{col}') for col in scores_cols],
            *[pl.col(col).list.eval(pl.element().list.mean()).alias(f'mean_{col}') for col in scores_cols],
        ).drop(scores_cols)\
        .rename({'candidate_ids': 'article'})
                    
    return df
    