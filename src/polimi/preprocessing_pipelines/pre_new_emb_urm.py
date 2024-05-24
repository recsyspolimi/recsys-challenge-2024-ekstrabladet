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
from polimi.utils._urm import build_urm_ner_scores
from polimi.utils._embeddings import weight_scores, build_normalized_embeddings_matrix, build_embeddings_scores, build_embeddings_agg_scores, build_history_w, build_norm_m_dict

import gc
import logging


def _get_urm_ner(urm_ner_scores_path: Path, behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame):
    if urm_ner_scores_path:
        logging.info('Collecting URM NER Scores...')
        ner_path = f'{urm_ner_scores_path}/urm_ner_scores.parquet'
        urm_ner_df = pl.read_parquet(ner_path)
    else:
        # Create URM
        URM_train = build_ner_urm(behaviors, history, articles, 'article_id_fixed')
        
        logging.info('Train NER score algorithm...')
        recs = train_ner_score_algo(URM_train, rec_dir=None, save_algo=False)
        
        logging.info('Building URM NER scores...')
        urm_ner_df = build_urm_ner_scores(behaviors, history, articles, recs)
        urm_ner_df = reduce_polars_df_memory_size(urm_ner_df)

    
    logging.info('Normalizing NER scores...')
    ner_features = [col for col in urm_ner_df.columns if '_ner_scores' in col]
    urm_ner_df = urm_ner_df.select('impression_id', 'user_id', 'article', *ner_features)\
        .explode(pl.all().exclude(['impression_id', 'user_id'])).with_columns(
            *get_norm_expression(ner_features, over='impression_id', norm_type='infinity')
        ).drop(ner_features)
        
    return reduce_polars_df_memory_size(urm_ner_df)


def _get_embdeddings_agg(emb_scores_path: Path, behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame, dataset_path: Path, weight_col = 'scroll_percentage_fixed_mmnorm_l1_w'):
    logging.info('Collecting embedding aggregations...')
    if emb_scores_path:
        emb_scores_df = pl.read_parquet(emb_scores_path)
    else:
        history_w = build_history_w(history, articles)
        history_w = history_w.select('user_id', weight_col)
        
        logging.info('Building embeddings scores...')
        norm_m_dict = build_norm_m_dict(articles, dataset_path, logging=logging.info)
        emb_scores_df = build_embeddings_scores(behaviors, history, articles, norm_m_dict=norm_m_dict)
        
        logging.info('Weightening embeddings scores...')
        emb_scores_df = emb_scores_df.join(history_w, on='user_id', how='left')
        weights_cols = [col for col in emb_scores_df.columns if col.endswith('_l1_w')]
        scores_cols = [col for col in emb_scores_df.columns if col.endswith('_scores')]
        emb_scores_df = weight_scores(emb_scores_df, scores_cols=scores_cols, weights_cols=weights_cols)
        emb_scores_df = reduce_polars_df_memory_size(emb_scores_df)

        
        logging.info('Aggregating embeddings scores...')
        agg_cols = [col for col in emb_scores_df.columns if '_scores' in col]
        logging.info(f'Agg cols: {agg_cols}')
        emb_scores_df = build_embeddings_agg_scores(emb_scores_df, agg_cols=agg_cols, last_k=[])    
        emb_scores_df = reduce_polars_df_memory_size(emb_scores_df)
    
    
    logging.info('Normalizing embeddings scores...')
    emb_features = [col for col in emb_scores_df.columns if 'weighted' in col and 'min' not in col] #remove min aggs
    emb_scores_df = emb_scores_df.select('impression_id', 'user_id', 'article', *emb_features)\
        .explode(pl.all().exclude(['impression_id', 'user_id'])).with_columns(
            *get_norm_expression(emb_features, over='impression_id', norm_type='infinity')
        ).drop(emb_features)
        
    return reduce_polars_df_memory_size(emb_scores_df)


def build_features(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                   test: bool = False, sample: bool = True, npratio: int = 2,
                   tf_idf_vectorizer: TfidfVectorizer = None, previous_version=None, **kwargs) -> pl.DataFrame:
    
    
        
    urm_ner_scores_path = Path(kwargs.get('urm_ner_scores_path')) if kwargs.get('urm_ner_scores_path') else None
    emb_scores_path = Path(kwargs.get('emb_scores_path')) if kwargs.get('emb_scores_path') else None
    dataset_path = Path(kwargs.get('dataset_path')) if kwargs.get('dataset_path') else None

    urm_ner_df = _get_urm_ner(urm_ner_scores_path, behaviors, history, articles)
    emb_scores_df = _get_embdeddings_agg(emb_scores_path, behaviors, history, articles, dataset_path)

    # Load old version
    if previous_version is None:
        df_features, vectorizer, unique_entities = _old_build_features(behaviors, history, articles,
                                                                                test=test, sample=sample, npratio=npratio,
                                                                                tf_idf_vectorizer=tf_idf_vectorizer, previous_version=previous_version,
                                                                                **kwargs)
    else:
        df_features = pl.read_parquet(previous_version)
        articles, vectorizer, unique_entities = _preprocess_articles(articles)

    
        
    # JOIN
    df_features = df_features.join(
        urm_ner_df, on=['impression_id', 'user_id', 'article'], how='left')
    df_features = df_features.join(
        emb_scores_df, on=['impression_id', 'user_id', 'article'], how='left')

    # Post Processing
    del emb_scores_df, urm_ner_df
    gc.collect()
    return df_features, vectorizer, unique_entities
