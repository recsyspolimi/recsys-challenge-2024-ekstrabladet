import os
from pathlib import Path
import polars as pl
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from polimi.utils._catboost import (
    _preprocess_articles
)
from polimi.utils._topic_model import _compute_topic_model, add_topic_model_features
from polimi.utils._polars import reduce_polars_df_memory_size, inflate_polars_df
from polimi.utils._norm_and_stats import *
from polimi.preprocessing_pipelines.pre_new import build_features_iterator as _old_build_features_iterator
from polimi.preprocessing_pipelines.pre_new import build_features as _old_build_features
from polimi.utils._urm import build_articles_with_processed_ner, build_ner_mapping, build_ner_urm, build_user_id_mapping
from polimi.scripts.build_urm_ner_scores_features import ALGO_NER_TRAIN_DICT, train_ner_score_algo
from polimi.utils._urm import build_urm_ner_score_features
from polimi.utils._embeddings import weight_scores, build_normalized_embeddings_matrix, build_embeddings_scores, build_embeddings_agg_scores, build_history_w

import gc
import logging


def _get_urm_ner(urm_ner_path: Path, history: pl.DataFrame, behaviors: pl.DataFrame, articles: pl.DataFrame):
    if urm_ner_path:
        logging.info('Collecting URM for NER...')
        ner_path = f'{urm_ner_path}/urm_ner_scores.parquet'
        urm_ner_df = pl.read_parquet(ner_path)
    else:
        # Create URM
        ap = build_articles_with_processed_ner(articles)
        ner_mapping = build_ner_mapping(ap)
        user_id_mapping = build_user_id_mapping(history)
        URM_train = build_ner_urm(history, ap, user_id_mapping, ner_mapping, 'article_id_fixed')
        
        logging.info('Train NER score algorithm...')
        recs = train_ner_score_algo(URM_train, urm_ner_path, ALGO_NER_TRAIN_DICT, save_algo=False)
        
        ap = ap.with_columns(
            pl.col('ner_clusters').list.eval(pl.element().replace(ner_mapping['ner'], ner_mapping['ner_index'], default=None)).list.drop_nulls().alias('ner_clusters_index'),
        )
        urm_ner_df = behaviors.rename({'article_ids_inview': 'candidate_ids'})\
            .with_columns(
                pl.col('candidate_ids').list.eval(pl.element().replace(ap['article_id'], ap['ner_clusters_index'], default=[])).alias('candidate_ner_index'),
                pl.col('user_id').replace(user_id_mapping['user_id'], user_id_mapping['user_index'], default=None).alias('user_index'),
            ).select('impression_id', 'user_id', 'user_index', 'candidate_ids', 'candidate_ner_index') 
            
        logging.info('Building URM NER score features...')
        urm_ner_df = build_urm_ner_score_features(urm_ner_df, ner_mapping=ner_mapping, recs=recs)
        urm_ner_df = reduce_polars_df_memory_size(urm_ner_df)

    
    logging.info('Normalizing NER scores...')
    ner_features = [col for col in urm_ner_df.columns if '_ner_scores' in col]
    urm_ner_df = urm_ner_df.explode(pl.all().exclude(['impression_id', 'user_id'])).with_columns(
            *[(pl.col(c) / pl.col(c).max().over('impression_id')
        ).alias(f'{c}_l_inf_impression') for c in ner_features],
    ).drop(ner_features)

    return reduce_polars_df_memory_size(urm_ner_df)


def _build_emb_norm_m_dict(emb_path: Path, articles: pl.DataFrame):
    emb_name_dict = {'Ekstra_Bladet_contrastive_vector': 'contrastive_vector',
                 'FacebookAI_xlm_roberta_base': 'xlm_roberta_base',
                 'Ekstra_Bladet_image_embeddings': 'image_embeddings',
                 'google_bert_base_multilingual_cased': 'bert_base_multilingual_cased'}
    
    norm_m_dict = {}
    article_emb_mapping = articles.select('article_id').unique().with_row_index()
    for dir, file_name in emb_name_dict.items():
        logging.info(f'Processing {file_name} embedding matrix...')
        emb_df = pl.read_parquet(emb_path / dir / f'{file_name}.parquet')
        emb_df.columns = ['article_id', 'embedding']
        logging.info(f'Building normalized embeddings matrix for {file_name}...')
        m = build_normalized_embeddings_matrix(emb_df, article_emb_mapping)
        norm_m_dict[file_name] = m
    return norm_m_dict

def _get_embdeddings_agg(emb_scores_path: Path, emb_path: Path, history: pl.DataFrame, behaviors: pl.DataFrame, articles: pl.DataFrame, weight_col = 'scroll_percentage_fixed_mmnorm_l1_w'):
    
    logging.info('Collecting embedding aggregations...')
    if emb_scores_path:
        emb_scores_df = pl.read_parquet(emb_scores_path)
    else:
        history_w = build_history_w(history, articles)
        history_w = history_w.select('user_id', weight_col)
        article_emb_mapping = articles.select('article_id').unique().with_row_index()

        history_m = history\
            .select('user_id', pl.col('article_id_fixed').list.eval(
                        pl.element().replace(article_emb_mapping['article_id'], article_emb_mapping['index'], default=None)))\
            .with_row_index('user_index')

        user_history_map = history_m.select('user_id', 'user_index')
        history_m = history_m['article_id_fixed'].to_numpy()
        emb_scores_df = behaviors.select('impression_id', 'user_id', pl.col('article_ids_inview').alias('article'))\
            .join(user_history_map, on='user_id')\
            .with_columns(
                pl.col('article').list.eval(pl.element().replace(article_emb_mapping['article_id'], article_emb_mapping['index'], default=None)).name.suffix('_index'),
            ).drop('impression_time_fixed', 'scroll_percentage_fixed', 'read_time_fixed')\
            .sort('user_id')
        
        norm_m_dict = _build_emb_norm_m_dict(emb_path, articles)
        logging.info('Building embeddings scores...')
        emb_scores_df = build_embeddings_scores(emb_scores_df, history_m=history_m, m_dict=norm_m_dict)
        emb_scores_df = reduce_polars_df_memory_size(emb_scores_df)

        logging.info('Weightening embeddings scores...')
        emb_scores_df = emb_scores_df.join(history_w, on='user_id', how='left')
        weights_cols = [col for col in emb_scores_df.columns if col.endswith('_l1_w')]
        scores_cols = [col for col in emb_scores_df.columns if col.endswith('_scores')]
        logging.info(f'Weights cols: {weights_cols}')
        logging.info(f'Scores cols: {scores_cols}')
        emb_scores_df = weight_scores(emb_scores_df, scores_cols=scores_cols, weights_cols=weights_cols)
        emb_scores_df = reduce_polars_df_memory_size(emb_scores_df)

        logging.info('Aggregating embeddings scores...')
        agg_cols = [col for col in emb_scores_df.columns if '_weighted_' in col]
        logging.info(f'Agg cols: {agg_cols}')
        emb_scores_df = build_embeddings_agg_scores(emb_scores_df, agg_cols=agg_cols, last_k=[])
        emb_scores_df = reduce_polars_df_memory_size(emb_scores_df)

        
    logging.info('Normalizing embeddings scores...')
    emb_features = [col for col in emb_scores_df.columns if 'weighted' in col and 'min' not in col] #remove min aggs
    emb_scores_df = emb_scores_df.select('impression_id', 'user_id', 'article', *emb_features)\
        .explode(pl.all().exclude(['impression_id', 'user_id'])).with_columns(
            *[(pl.col(c) / pl.col(c).max().over('impression_id')
           ).alias(f'{c}_l_inf_impression') for c in emb_features],
        ).drop(emb_features)
        
    return reduce_polars_df_memory_size(emb_scores_df)



def build_features(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                   test: bool = False, sample: bool = True, npratio: int = 2,
                   tf_idf_vectorizer: TfidfVectorizer = None, previous_version=None, **kwargs) -> pl.DataFrame:
    
    emb_scores_path = Path(kwargs['emb_scores_path']) if kwargs['emb_scores_path'] else None
    urm_ner_path = Path(kwargs['urm_ner_path']) if kwargs['urm_ner_path'] else None
    emb_path = Path(kwargs['emb_path']) if kwargs['emb_path'] else None

    emb_scores_df = _get_embdeddings_agg(emb_scores_path, emb_path, history, behaviors, articles)
    urm_ner_df = _get_urm_ner(urm_ner_path, history, behaviors, articles)

    # Load old version
    if previous_version is None:
        df_features, vectorizer, unique_entities = _old_build_features(behaviors, history, articles,
                                                                                test=test, sample=sample, npratio=npratio,
                                                                                tf_idf_vectorizer=tf_idf_vectorizer, previous_version=previous_version,
                                                                                **kwargs)
    else:
        df_features = pl.read_parquet(previous_version)
        articles, vectorizer, unique_entities = _preprocess_articles(articles)

    
        
    # slice_features = sliced_df.something()
    # JOIN
    df_features = df_features.join(
        urm_ner_df, on=['impression_id', 'user_id', 'article'], how='left')
    df_features = df_features.join(
        emb_scores_df, on=['impression_id', 'user_id', 'article'], how='left')

    # Post Processing
    del emb_scores_df, urm_ner_df
    gc.collect()
    return df_features, vectorizer, unique_entities
