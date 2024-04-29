try:
    import polars as pl
except ImportError:
    print("polars not available")

from pathlib import Path
import scipy.sparse as sps
from tqdm import tqdm
import numpy as np
from RecSys_Course_AT_PoliMi.Recommenders.BaseRecommender import BaseRecommender
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
        .rename({'index': 'ner_index'})     

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
        rec_instance.save_model(folder_path=output_dir.resolve(), file_name=file_name)
    
    return rec_instance

def build_recsys_algorithms(history: pl.DataFrame, behaviors: pl.DataFrame, articles: pl.DataFrame, recs: list[BaseRecommender]):

    user_id_mapping = build_user_id_mapping(history)
    item_mapping = build_item_mapping(articles)
    
    df = behaviors\
            .select('impression_id', 'article_id', 'user_id')\
            .unique()\
            .join(item_mapping, on='article_id')\
            .join(user_id_mapping, on='user_id')\
            .sort(['user_index', 'item_index'])\
            .group_by('user_index').map_groups(lambda df: df.pipe(_compute_recommendations, recommenders=recs))

    return reduce_polars_df_memory_size(df)
            

def _compute_recommendations(user_items_df, recommenders):
    user_index = user_items_df['user_index'].to_list()[0]
    items = user_items_df['item_index'].to_numpy()

    scores = {}
    for name, model in recommenders.items():
        scores[name] = model._compute_item_score([user_index], items)[0, items]

    return user_items_df.with_columns(
        [
            pl.Series(model).alias(name) for name, model in scores.items()
        ]
    )


def load_recommender(URM: sps.csr_matrix, recommender: BaseRecommender, file_path: Path, file_name: str=None):
    rec_instance = recommender(URM)
    rec_instance.load_model(folder_path=str(file_path), file_name=file_name)
    return rec_instance

def build_ner_scores_features(history: pl.DataFrame, behaviors: pl.DataFrame, articles: pl.DataFrame, recs: list[BaseRecommender], batch_size=BATCH_SIZE, save_path:Path=None):
    '''
    Builds the score features for ner interactions. For each (impression_id, user_id, article), 
    it computes the sum, max and mean of the scores for each recommender in recs. Note that each score
    is normalized by the maximum value of the scores for that particular inview (candidate) list (inf norm).
    Args:
        history: the (raw) users history dataframe
        behaviors: the (raw) behaviors dataframe
        articles: the (raw) articles dataframe
        recs: a list containing all the recommenders to use for computing the scores
    Returns:
        pl.DataFrame: the dataframe containing the triple (impression_id, user_id, article) and the scores features for each rec in recs.
    '''
    
    user_id_mapping = build_user_id_mapping(history)
    ap = build_articles_with_processed_ner(articles)
    ner_mapping = build_ner_mapping(ap)
    ap = ap.with_columns(
        pl.col('ner_clusters').list.eval(pl.element().replace(ner_mapping['ner'], ner_mapping['ner_index'], default=None).drop_nulls()).alias('ner_clusters_index'),
    )
    
    df = behaviors.rename({'article_ids_inview': 'candidate_ids'})\
        .with_columns(
            pl.col('candidate_ids').list.eval(pl.element().replace(ap['article_id'], ap['ner_clusters_index'], default=None)).alias('candidate_ner_index'),
            pl.col('user_id').replace(user_id_mapping['user_id'], user_id_mapping['user_index'], default=None).alias('user_index'),
        ).select('impression_id', 'user_id', 'user_index', 'candidate_ids', 'candidate_ner_index')        
    
    all_items = ner_mapping['ner_index'].unique().sort().to_list()
    
    df = pl.concat([ #remove empty ner_index
        slice.explode(['candidate_ids', 'candidate_ner_index'])\
            .filter(pl.col('candidate_ner_index').list.len() > 0)\
            .group_by(['impression_id', 'user_id', 'user_index']).agg(pl.all())
        for slice in tqdm(df.iter_slices(batch_size), total=df.shape[0]//batch_size)
    ])
    
    df = pl.concat([
            slice.with_columns(
                    *[pl.col('candidate_ner_index').list.eval(
                        pl.element().list.eval(pl.element().replace(all_items, rec._compute_item_score(user_index)[0], default=None))
                    ).alias(f"{rec.RECOMMENDER_NAME}_ner_scores") for rec in recs]
                ).drop('user_index', 'candidate_ner_index')
        for user_index, slice in tqdm(df.partition_by(['user_index'], as_dict=True).items(), total=df['user_index'].n_unique())
        ])
    
    df = reduce_polars_df_memory_size(df)
    scores_cols = [col for col in df.columns if '_ner_scores' in col]
    df = df.with_columns(
            *[pl.col(col).list.eval(pl.element().list.sum()).alias(f'sum_{col}') for col in scores_cols],
            *[pl.col(col).list.eval(pl.element().list.max()).alias(f'max_{col}') for col in scores_cols],
            *[pl.col(col).list.eval(pl.element().list.mean()).alias(f'mean_{col}') for col in scores_cols],
        ).with_columns(
            pl.all().exclude(['impression_id', 'user_id', 'candidate_ids'] + scores_cols).list.eval(pl.element().truediv(pl.element().max()).fill_nan(0.0)), #inf norm
        ).drop(scores_cols)
        
    df = reduce_polars_df_memory_size(df)
    df = df.sort(['impression_id', 'user_id'])\
        .explode(pl.all().exclude(['impression_id', 'user_id']))\
        .rename({'candidate_ids': 'article'})
    
    if save_path:
        print(f'Saving ner scores features ... [{save_path}]')
        df.write_parquet(save_path / 'ner_scores_features.parquet')
        
    return df