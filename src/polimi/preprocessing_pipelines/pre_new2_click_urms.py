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
import gc
import logging


def _get_urm_ner(urm_ner_path):
    
    print('Collecting URM for NER...')
    
    ner_path = f'{urm_ner_path}/urm_ner_scores.parquet'
    urm_ner_df = pl.read_parquet(ner_path)
    ner_features = [col for col in urm_ner_df.columns if '_scores' in col]

    urm_ner_df = urm_ner_df.explode(pl.all().exclude(['impression_id', 'user_id'])).with_columns(
        *[(pl.col(c) / pl.col(c).max().over('impression_id')
           ).alias(f'{c}_l_inf_impression') for c in ner_features],
    ).drop(ner_features)

    return reduce_polars_df_memory_size(urm_ner_df)


def _get_embdeddings_agg(emb_scores_path):
    
    print('Collecting embedding aggregations...')
    
    embedding_scores_path = f'{emb_scores_path}/agg_embeddings_scores_scroll_percentage_fixed_mmnorm_l1_w.parquet'
    emb_scores_df = pl.read_parquet(embedding_scores_path)
    emb_features = [
        col for col in emb_scores_df.columns if 'weighted_scroll_percentage_fixed' in col and 'min' not in col]
    emb_scores_df = emb_scores_df.select('impression_id', 'user_id', 'article', *emb_features).explode(pl.all().exclude(['impression_id', 'user_id'])).with_columns(
        *[(pl.col(c) / pl.col(c).max().over('impression_id')
           ).alias(f'{c}_l_inf_impression') for c in emb_features],
    ).drop(emb_features)
    return reduce_polars_df_memory_size(emb_scores_df)


def _get_emotions_embeddings(emotion_emb_path, history):
    
    print('Collecting emotions embeddings...')
    
    emotion_emb = pl.read_parquet(emotion_emb_path).with_columns(
        pl.col("emotion_scores").list.to_struct()
        .struct.rename_fields(['emotion_0', 'emotion_1', 'emotion_2', 'emotion_3', 'emotion_4', 'emotion_5']))\
        .unnest("emotion_scores")

    embedded_history = pl.concat(
        rows.select(['user_id', 'article_id_fixed']).explode('article_id_fixed').join(
            emotion_emb, left_on='article_id_fixed', right_on='article_id', how='left')
        .group_by('user_id').agg(
            [pl.col(f'emotion_{i}').mean().cast(pl.Float32).alias(f'user_emotion{i}') for i in range(6)])
        for rows in tqdm(history.iter_slices(20000), total=history.shape[0] // 20000))
    return reduce_polars_df_memory_size(emotion_emb), reduce_polars_df_memory_size(embedded_history)


def _get_click_predictors(click_predictors_path, df_features):
    
    print('Collecting click predictors...')

    emb = pl.read_parquet(click_predictors_path)
    emb_col = emb.drop(['user_id', 'article']).columns
    normalized_emb = df_features.select(['impression_id','user_id', 'article']).join(emb, on=['user_id', 'article'], how='left')\
        .with_columns(
        *[(pl.col(col) / pl.col(col).max().over('user_id')) for col in emb_col])
    return normalized_emb


# classifier score small 7863 -> 7854

def build_features_iterator(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                            test: bool = False, sample: bool = True, npratio: int = 2,
                            tf_idf_vectorizer: TfidfVectorizer = None, n_batches: int = 10, previous_version=None,
                            **kwargs):

    emb_scores_path = kwargs['emb_scores_path']
    urm_ner_path = kwargs['urm_ner_path']
    emotion_emb_path = kwargs['emotion_emb_path']
    click_predictors_path = kwargs['click_predictors_path']
    recsys_features_path = kwargs['rec_sys_path']

    # Load old version
    if previous_version is None:
        df_features, vectorizer, unique_entities = _old_build_features_iterator(behaviors, history, articles,
                                                                                test=test, sample=sample, npratio=npratio,
                                                                                tf_idf_vectorizer=tf_idf_vectorizer, n_batches=n_batches, previous_version=previous_version,
                                                                                **kwargs)
    else:
        df_features = pl.read_parquet(previous_version)
        articles, vectorizer, unique_entities = _preprocess_articles(articles)

    
    urm_ner_df = _get_urm_ner(urm_ner_path)
    emb_scores_df = _get_embdeddings_agg(emb_scores_path)

    # Emotion Embeddings
    emotion_emb, embedded_history = _get_emotions_embeddings(
        emotion_emb_path, history)

    # RecSysy
    print('Collecting RECSYS features...')
    recsys_features = pl.read_parquet(recsys_features_path)

    # Click Predictors
    click_predictors = _get_click_predictors(
        click_predictors_path, df_features)

    df_features = None
    i = 0
    for sliced_df in behaviors.iter_slices(behaviors.shape[0] // n_batches):
        logging.info(f'Preprocessing slice {i}')

        # slice_features = sliced_df.something()
        # JOIN
        sliced_df = sliced_df.join(
            urm_ner_df, on=['impression_id', 'user_id', 'article'], how='left')
        sliced_df = sliced_df.join(
            emb_scores_df, on=['impression_id', 'user_id', 'article'], how='left')
        sliced_df = sliced_df.join(
            embedded_history, on='user_id', how='left')
        sliced_df = sliced_df.join(
            emotion_emb, left_on='article', right_on='article_id', how='left')
        sliced_df = sliced_df.join(click_predictors, on=[
                                   'user_id', 'article', 'impression_id'], how='left')
        sliced_df = sliced_df.join(
            recsys_features, on=['impression_id', 'user_id', 'article'], how='left')

        if df_features is None:
            df_features = inflate_polars_df(sliced_df)
        else:
            df_features = df_features.vstack(inflate_polars_df(sliced_df))

    # Post Processing
    del emb_scores_df, urm_ner_df, embedded_history, emotion_emb
    gc.collect()

    return df_features, vectorizer, unique_entities


def build_features_iterator_test(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                                 test: bool = False, sample: bool = True, npratio: int = 2,
                                 tf_idf_vectorizer: TfidfVectorizer = None, n_batches: int = 100, previous_version=None,
                                 **kwargs):
    raise "Not implemented for version new_click_urms"


def build_features(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                   test: bool = False, sample: bool = True, npratio: int = 2,
                   tf_idf_vectorizer: TfidfVectorizer = None, previous_version=None, **kwargs) -> pl.DataFrame:
    emb_scores_path = kwargs['emb_scores_path']
    urm_ner_path = kwargs['urm_ner_path']
    emotion_emb_path = kwargs['emotion_emb_path']
    click_predictors_path = kwargs['click_predictors_path']
    recsys_features_path = kwargs['rec_sys_path']

    # Load old version
    if previous_version is None:
        df_features, vectorizer, unique_entities = _old_build_features_iterator(behaviors, history, articles,
                                                                                test=test, sample=sample, npratio=npratio,
                                                                                tf_idf_vectorizer=tf_idf_vectorizer, n_batches=n_batches, previous_version=previous_version,
                                                                                **kwargs)
    else:
        df_features = pl.read_parquet(previous_version)
        articles, vectorizer, unique_entities = _preprocess_articles(articles)

    urm_ner_df = _get_urm_ner(urm_ner_path)
    emb_scores_df = _get_embdeddings_agg(emb_scores_path)

    # Emotion Embeddings
    emotion_emb, embedded_history = _get_emotions_embeddings(
        emotion_emb_path, history)

    # RecSysy
    print('Collecting RECSYS features...')
    recsys_features = pl.read_parquet(recsys_features_path)

    # Click Predictors
    click_predictors = _get_click_predictors(
        click_predictors_path, df_features)

    # slice_features = sliced_df.something()
    # JOIN
    df_features = df_features.join(
        urm_ner_df, on=['impression_id', 'user_id', 'article'], how='left')
    df_features = df_features.join(
        emb_scores_df, on=['impression_id', 'user_id', 'article'], how='left')
    df_features = df_features.join(
        embedded_history, on='user_id', how='left')
    df_features = df_features.join(
        emotion_emb, left_on='article', right_on='article_id', how='left')
    df_features = df_features.join(click_predictors, on=[
        'user_id', 'article', 'impression_id'], how='left')
    df_features = df_features.join(
        recsys_features, on=['impression_id', 'user_id', 'article'], how='left')

    # Post Processing
    del emb_scores_df, urm_ner_df, embedded_history, emotion_emb
    gc.collect()
    return df_features, vectorizer, unique_entities
