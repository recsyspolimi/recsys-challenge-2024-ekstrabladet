from tqdm import tqdm
from rich.progress import Progress
from scipy import stats
import scipy.sparse as sps
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing_extensions import Tuple, List, Dict
import logging
from datetime import datetime, time
from RecSys_Course_AT_PoliMi.Evaluation.Evaluator import EvaluatorHoldout
from RecSys_Course_AT_PoliMi.Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from polimi.utils._custom import ALGORITHMS
try:
    import polars as pl
except ImportError:
    print("polars not available")

from polimi.utils._polars import *
from polimi.utils._custom import *
from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
)
from polimi.utils._embeddings import fast_distance

"""
Utils for catboost.
"""


def build_features_iterator(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                            test: bool = False, sample: bool = True, npratio: int = 2,
                            tf_idf_vectorizer: TfidfVectorizer = None, n_batches: int = 10):
    '''
    Generator function to build the features from blocks of the behaviors. It returns an iterable of slices of the 
    dataframe with the features. See build_features for a description of the features.
    Use this function instead of build_features when there are memory issue.

    Args:
        behaviors: the dataframe containing the impressions
        history: the dataframe containing the users history
        articles: the dataframe containing the articles features
        test: if true consider the behaviors as a test split, so it will not attempt to build the target column
        sample: if true, behaviors will be sampled using wu strategy, done only if test=False 
            (otherwise there is no information about negative articles in the inview list)
        npratio: the number of negative samples for wu sampling (useful only if sample=True)
        tf_idf_vectorizer: an optional already fitted tf_idf_vectorizer, if None it will be fitted on the provided articles topics
        n_batches: the number of blocks

    Returns:
        (pl.DataFrame, TfidfVectorizer, List[str]): the dataframe with all the features, the fitted tf-idf vectorizer and the
            unique entities (useful to cast the categorical columns when eventually transforming the dataframe to polars)
    '''
    behaviors, history, articles, vectorizer, unique_entities, cols_explode, rename_cols = _preprocessing(
        behaviors, history, articles, test, sample, npratio
    )

    for sliced_df in behaviors.iter_slices(behaviors.shape[0] // n_batches):
        slice_features = sliced_df.pipe(_build_features_behaviors, history=history, articles=articles,
                                        cols_explode=cols_explode, rename_columns=rename_cols, unique_entities=unique_entities)
        yield slice_features, vectorizer, unique_entities


def build_features(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                   test: bool = False, sample: bool = True, npratio: int = 2,
                   tf_idf_vectorizer: TfidfVectorizer = None) -> pl.DataFrame:
    '''
    Builds the training/evaluation features dataframe. Each row of the resulting dataframe will be an article
    in the article_ids_inview list (that will be exploded), if sampling is performed only some negative articles
    are present (based on the npratio argument). If test is False, a binary target column is built, that is 1 if
    the viewed article was clicked, or false otherwise.
    The function builds the following features for each (article, impression) pairs:
    - weekday and hour of the day
    - booleans saying if a certain entity_group is present or not in the target article
    - the number of different categories of the articles seen by the user
    - a boolean saying that the target article is of the favourite category of the user
    - the percentage of articles seen by the user with the same category of the target article
    - some the jaccard similarities (min, max, std, mean) of the target article with the seen articles in the history
    - number of articles seen in the history by the associated user
    - statistics of the user read time (median, max, sum), scroll percentage (median, max) in the history
    - mode of the user impression hours and impression days in the history
    - percentage of seen articles (in the history) with sentiment label Positive, Negative and Neutral
    - percentage of seen articles (in the history) that are strongly Positive, Negative and Neutral (i.e. with score > 0.8)
    - most frequent category in the user history of seen articles
    - percentage of seen articles with type different from article_default by the user
    - percentage of articles by the user in the history that contains each given entity
    - the cosine similarity between the article topics and the topics in the history of the user

    Args:
        behaviors: the dataframe containing the impressions
        history: the dataframe containing the users history
        articles: the dataframe containing the articles features
        test: if true consider the behaviors as a test split, so it will not attempt to build the target column
        sample: if true, behaviors will be sampled using wu strategy, done only if test=False 
            (otherwise there is no information about negative articles in the inview list)
        npratio: the number of negative samples for wu sampling (useful only if sample=True)
        tf_idf_vectorizer: an optional already fitted tf_idf_vectorizer, if None it will be fitted on the provided articles topics

    Returns:
        (pl.DataFrame, TfidfVectorizer, List[str]): the dataframe with all the features, the fitted tf-idf vectorizer and the
            unique entities (useful to cast the categorical columns when eventually transforming the dataframe to polars)
    '''

    behaviors, history, articles, vectorizer, unique_entities, cols_explode, rename_cols = _preprocessing(
        behaviors, history, articles, test, sample, npratio
    )

    df_features = behaviors.pipe(_build_features_behaviors, history=history, articles=articles,
                                 cols_explode=cols_explode, rename_columns=rename_cols, unique_entities=unique_entities)

    return df_features, vectorizer, unique_entities


def _build_features_behaviors(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                              cols_explode: List[str], rename_columns: Dict[str, str], unique_entities: List[str],
                              unique_categories: dict):
    select_columns = ['impression_id', 'article_ids_inview', 'impression_time', 'device_type', 'read_time',
                      'scroll_percentage', 'user_id', 'is_sso_user', 'gender', 'age', 'is_subscriber', 'session_id']
    if 'labels' in behaviors.columns:
        select_columns += ['labels']

    return behaviors.select(select_columns) \
        .with_columns(pl.col('gender').fill_null(2)) \
        .explode(cols_explode) \
        .rename(rename_columns) \
        .with_columns(pl.col('article').cast(pl.Int32)) \
        .pipe(add_trendiness_feature, articles=articles, period='3d') \
        .unique(['impression_id', 'article', 'user_id']) \
        .with_columns(
            pl.col('impression_time').dt.weekday().alias('weekday'),
            pl.col('impression_time').dt.hour().alias('hour'),
            pl.col('article').cast(pl.Int32),
    ).join(articles.select(['article_id', 'premium', 'published_time', 'category',
                            'sentiment_score', 'sentiment_label', 'entity_groups',
                            'num_images', 'title_len', 'subtitle_len', 'body_len', 'num_topics']),
           left_on='article', right_on='article_id', how='left') \
        .with_columns(
            (pl.col('impression_time') - pl.col('published_time')
             ).dt.total_days().alias('article_delay_days'),
            (pl.col('impression_time') - pl.col('published_time')
             ).dt.total_hours().alias('article_delay_hours')
    ).with_columns(
            pl.col('entity_groups').list.contains(
                entity).alias(f'Entity_{entity}_Present')
            for entity in unique_entities
    ).drop('entity_groups') \
        .pipe(add_session_features, history=history, behaviors=behaviors, articles=articles) \
        .pipe(add_category_popularity, articles=articles) \
        .pipe(_join_history, history=history, articles=articles, unique_categories=unique_categories)


def _preprocessing(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                   test: bool = False, sample: bool = True, npratio: int = 2,):
    if not test and sample:
        behaviors = behaviors.pipe(
            sampling_strategy_wu2019, npratio=npratio, shuffle=False, with_replacement=True, seed=123
        )

    if not test:
        behaviors = behaviors.pipe(
            create_binary_labels_column, shuffle=True, seed=123)
        columns_to_explode = ['article_ids_inview', 'labels']
        renaming_columns = {
            'article_ids_inview': 'article', 'labels': 'target'}
    else:
        columns_to_explode = 'article_ids_inview'
        renaming_columns = {'article_ids_inview': 'article'}

    articles, tf_idf_vectorizer, unique_entities = _preprocess_articles(
        articles)
    history = _build_history_features(
        history, articles, unique_entities, tf_idf_vectorizer)
    return behaviors, history, articles, tf_idf_vectorizer, unique_entities, columns_to_explode, renaming_columns


def _preprocess_articles(articles: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
    '''
    Preprocess the articles dataframe, extracting the number of images and the length of title, subtitle and body,
    and adding the topics tf_idf vector

    Args:
        articles: the articles dataframe

    Returns:
        pl.DataFrame: the preprocessed articles
        TfidfVectorizer: the fitted tf_idf vectorizer
        List[str]: the unique entities contained in the dataframe
    '''
    unique_entities = articles.select('entity_groups').explode(
        'entity_groups')['entity_groups'].unique().to_list()
    unique_entities = [e for e in unique_entities if e is not None]

    vectorizer = TfidfVectorizer()
    articles = articles.with_columns(
        pl.col('image_ids').list.len().alias('num_images'),
        pl.col('title').str.split(by=' ').list.len().alias('title_len'),
        pl.col('subtitle').str.split(by=' ').list.len().alias('subtitle_len'),
        pl.col('body').str.split(by=' ').list.len().alias('body_len'),
        pl.col('topics').list.len().alias('num_topics'),
        # useful for saving memory when joining with the history dataframe
        pl.when(pl.col('sentiment_label') == 'Negative').then(-1) \
        .otherwise(
            pl.when(pl.col('sentiment_label') ==
                    'Positive').then(1).otherwise(0)
        ).cast(pl.Int8).alias('sentiment_label_int'),
        (pl.col('article_type') == 'article_default').cast(
            pl.UInt8).alias('is_article_default'),
        # very important for tf-idf, otherwise multiple tokens for topics with spaces are built
        pl.col('topics').list.eval(
            pl.element().str.split(by=' ').list.join('_')),
        pl.Series(
            vectorizer.fit_transform(
                articles.with_columns(pl.col('topics').list.join(separator=' '))[
                    'topics'].to_list()
            ).toarray().astype(np.float32)
        ).alias('topics_idf')
    )
    return articles, vectorizer, unique_entities


def _add_topics_tf_idf_columns(df, topics_col, vectorizer, col_name=None):
    if col_name is None:
        col_name = f'{topics_col}_tf_idf'

    return df.with_columns(
        pl.Series(
            vectorizer.transform(
                df.with_columns(pl.col(topics_col).list.join(
                    separator=' '))[topics_col].to_list()
            ).toarray().astype(np.float32)
        ).alias(f'{topics_col}_tf_idf')
    )


def _build_history_features(history: pl.DataFrame, articles: pl.DataFrame, unique_entities: List[str],
                            vectorizer: TfidfVectorizer, strong_thr: float = 0.8) -> pl.DataFrame:
    '''
    Builds all the features of the users history. These features are:
    - number of articles seen
    - statistics of the user read time (median, max, sum), scroll percentage (median, max), impression
      hours (mode), impression day (mode)
    - percentage of articles with sentiment label Positive, Negative and Neutral
    - most frequent category in the user history of seen articles
    - percentage of seen articles with type different from article_default
    - percentage of articles in the history that contains each given entity
    It also adds the topics tf-idf vector.

    Args:
        history: the (raw) users history dataframe
        articles: the preprocessed articles dataframe
        unique_entities: a list containing all the possible/considered unique entity groups of the articles
        vectorizer: a fitted tf-idf to build the topics tf-idf of the history
        strong_thr: a threshold (between 0 and 1) on the sentiment score for considering a sentiment label as strong

    Returns:
        pl.DataFrame: the preprocessed history dataframe
    '''
    history = pl.concat(
        rows.with_columns(
            pl.col('article_id_fixed').list.len().alias('NumArticlesHistory'))
        .explode(['article_id_fixed', 'impression_time_fixed', 'read_time_fixed', 'scroll_percentage_fixed'])
        .sort(by=['user_id', 'impression_time_fixed'])
        .with_columns(
            pl.col('impression_time_fixed').dt.weekday().alias('weekday'),
            pl.col('impression_time_fixed').dt.hour().alias('hour'),
        ).join(articles.select(['article_id', 'category', 'is_article_default', 'sentiment_label_int',
                                'sentiment_score', 'entity_groups', 'topics']),
               left_on='article_id_fixed', right_on='article_id', how='left')
        .with_columns(
            (pl.col('sentiment_label_int') == 0).alias('is_neutral'),
            (pl.col('sentiment_label_int') == 1).alias('is_positive'),
            (pl.col('sentiment_label_int') == -1).alias('is_negative'),
            ((pl.col('sentiment_label_int') == 0) & (
                pl.col('sentiment_score') > strong_thr)).alias('strong_neutral'),
            ((pl.col('sentiment_label_int') == 1) & (
                pl.col('sentiment_score') > strong_thr)).alias('strong_positive'),
            ((pl.col('sentiment_label_int') == -1) &
             (pl.col('sentiment_score') > strong_thr)).alias('strong_negative'),
            pl.col('entity_groups').list.unique(),
        ).group_by('user_id').agg(
            pl.col('article_id_fixed'),
            pl.col('impression_time_fixed'),
            pl.col('category'),
            pl.col('NumArticlesHistory').first(),
            pl.col('read_time_fixed').median().alias('MedianReadTime'),
            pl.col('read_time_fixed').max().alias('MaxReadTime'),
            pl.col('read_time_fixed').sum().alias('TotalReadTime'),
            pl.col('scroll_percentage_fixed').median().alias(
                'MedianScrollPercentage'),
            pl.col('scroll_percentage_fixed').max().alias(
                'MaxScrollPercentage'),
            (pl.col('is_neutral').sum() /
             pl.col('NumArticlesHistory').first()).alias('NeutralPct'),
            (pl.col('is_positive').sum() /
             pl.col('NumArticlesHistory').first()).alias('PositivePct'),
            (pl.col('is_negative').sum() /
             pl.col('NumArticlesHistory').first()).alias('NegativePct'),
            (pl.col('strong_neutral').sum() /
             pl.col('NumArticlesHistory').first()).alias('PctStrongNeutral'),
            (pl.col('strong_positive').sum() /
             pl.col('NumArticlesHistory').first()).alias('PctStrongPositive'),
            (pl.col('strong_negative').sum() /
             pl.col('NumArticlesHistory').first()).alias('PctStrongNegative'),
            (1 - (pl.col('is_article_default').sum() /
                  pl.col('NumArticlesHistory').first())).alias('PctNotDefaultArticles'),
            pl.col('category').mode().alias('MostFrequentCategory'),
            pl.col('weekday').mode().alias('MostFrequentWeekday'),
            pl.col('hour').mode().alias('MostFrequentHour'),
            pl.col('entity_groups').flatten(),
            pl.col('topics').flatten().alias('topics_flatten')
        ).pipe(_add_topics_tf_idf_columns, topics_col='topics_flatten', vectorizer=vectorizer)
        .drop('topics_flatten').with_columns(
            pl.col('MostFrequentCategory').list.first(),
            pl.col('MostFrequentWeekday').list.first(),
            pl.col('MostFrequentHour').list.first(),
        ).with_columns(
            (pl.col('entity_groups').list.count_matches(entity) /
             pl.col('NumArticlesHistory')).alias(f'{entity}Pct')
            for entity in unique_entities
        ).drop('entity_groups')
        for rows in tqdm(history.iter_slices(1000), total=history.shape[0] // 1000)
    )
    return reduce_polars_df_memory_size(history)


def _join_history(df_features: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame, unique_categories: dict):
    '''
    Join the dataframe with the current features with the history dataframe, also adding the jaccard similarity features, the 
    entity features, the tf-idf cosine and the category features.

    Args:
        df_features: the dataframe with the partial features on which the history will be joined
        history: the history dataframe
        articles: the dataframe with the articles

    Returns:
        pl.DataFrame: a dataframe with the added features
    '''

    prev_train_columns = [
        c for c in df_features.columns if c not in ['impression_id', 'article']]
    df_features = pl.concat(
        rows.join(history.select(
            ['user_id', 'article_id_fixed']), on='user_id', how='left')
        .join(articles.select(['article_id', 'topics', 'entity_groups', 'topics_idf']), left_on='article', right_on='article_id', how='left')
        .explode(['article_id_fixed'])
        .join(articles.select(['article_id', 'topics', 'entity_groups']), left_on='article_id_fixed', right_on='article_id', how='left')
        .rename({'topics_right': 'topics_history', 'entity_groups_right': 'entity_groups_history'})
        .with_columns(
            (pl.col("topics").list.set_intersection(pl.col("topics_history")).list.len().truediv(
                pl.col("topics").list.set_union(
                    pl.col("topics_history")).list.len()
            )).alias("JS"),
            pl.col('entity_groups').list.set_intersection(
                pl.col('entity_groups_history')).list.len().alias('common_entities'),
        ).drop(['entity_groups_history', 'entity_groups', 'topics', 'topics_history'])
        .group_by(['impression_id', 'article']).agg(
            pl.col(prev_train_columns).first(),
            pl.col('topics_idf').first(),
            pl.col('common_entities').mean().alias('MeanCommonEntities'),
            pl.col('common_entities').max().alias('MaxCommonEntities'),
            pl.col("JS").mean().alias("mean_JS"),
            pl.col("JS").min().alias("min_JS"),
            pl.col("JS").max().alias("max_JS"),
            pl.col("JS").std().alias("std_JS"),
        ).join(history.drop(['article_id_fixed', 'impression_time_fixed']), on='user_id', how='left')
        .with_columns(
            pl.struct(['topics_idf', 'topics_flatten_tf_idf']).map_elements(
                lambda x: fast_distance(x['topics_idf'], x['topics_flatten_tf_idf']), return_dtype=pl.Float64
            ).cast(pl.Float32).alias('topics_cosine'),
            (pl.col('category') == pl.col('MostFrequentCategory')).alias(
                'IsFavouriteCategory'),
            pl.col('category_right').list.n_unique().alias(
                'NumberDifferentCategories'),
            list_pct_matches_with_col(
                'category_right', 'category').alias('PctCategoryMatches'),
        )
        .with_columns(
            [list_pct_matches_with_constant('category_right', c).alias(f'Category_{c_str}_Pct')
             for c, c_str in unique_categories.items()])
        .drop(['topics_idf', 'topics_flatten', 'topics_flatten_tf_idf', 'category_right'])
        for rows in tqdm(df_features.iter_slices(20000), total=df_features.shape[0] // 20000)
    )
    return reduce_polars_df_memory_size(df_features)


def add_trendiness_feature(df_features: pl.DataFrame, articles: pl.DataFrame, period: str = "3d"):
    """
    Adds a new feature "trendiness_score" to each impression.
    The trendiness score for an article is computed as the sum, for each topic of the article, of the times that topic has happeared in some article
    published in the previous <period> before the impression (normalized with the number of total publications for that topic).

    Args:
        df_features: The dataset to be enriched with the new feature. It MUST contain the "impression_time"
        articles: The articles dataset with the topics of the articles and their publication time.
        period: The window size for the computation of the scores, in string encoding 
            (ex. 1ns, 1us, 1s, 1m, 1h, 1d, 1w,.. or combinations like "3d12h4m25s". See doc for polars.DataFrame.rolling for more details)

    Returns:
        pl.DataFrame: The training dataset enriched with a new column, containing the trendiness_score.
    """

    topics = articles.select("topics").explode("topics").unique()
    topics = [topic for topic in topics["topics"] if topic is not None]
    # min_impression_time = df_features.select(pl.col("impression_time")).min().item()

    # topics_total_publications= articles.filter(pl.col("published_time")< min_impression_time ).select("topics") \
    # .explode("topics").group_by("topics").len()

    topics_popularity = articles.select(["published_time", "topics"]).with_columns(
        pl.col("published_time").dt.date().alias("published_date")
    ).drop("published_time").group_by("published_date").agg(
        pl.col("topics").flatten()
    ).sort("published_date").set_sorted("published_date").upsample(time_column="published_date", every="1d") \
        .rolling(index_column="published_date", period=period).agg(
        [pl.col("topics").list.count_matches(topic).sum().alias(
            f"{topic}_matches") for topic in topics]
    )

    return df_features.with_columns(
        pl.col("impression_time").dt.date().alias("impression_date")
    ).join(other=articles.select(["article_id", "topics"]), left_on="article", right_on="article_id", how="left") \
        .with_columns(
        [pl.col("topics").list.contains(topic).cast(
            pl.Int8).alias(f"{topic}_present") for topic in topics]
    ).join(other=topics_popularity, left_on=pl.col("impression_date"), right_on=(pl.col("published_date")+pl.duration(days=1)), how="left") \
        .with_columns(
        [pl.col(f"{topic}_present").mul(pl.col(f"{topic}_matches")).alias(
            f"trendiness_score_{topic}") for topic in topics]
    ).with_columns(
        pl.sum_horizontal([pl.col(f"trendiness_score_{topic}") for topic in topics]).alias(
            "trendiness_score"),
    ).drop(
        [f"trendiness_score_{topic}" for topic in topics]
    ).drop(
        [f"{topic}_matches" for topic in topics]
    ).drop(
        [f"{topic}_present" for topic in topics]
    ).drop(["topics", "impression_date"])


def add_category_popularity(df_features: pl.DataFrame, articles: pl.DataFrame) -> pl.DataFrame:
    '''
    Add a feature to the dataframe df_features, named yesterday_category_daily_pct, that contains the percentage of articles
    published in the previous day with the same category.

    Args:
        df_features: the dataframe where the feature will be added
        articles: the articles dataframe

    Returns:
        pl.DataFrame: the dataframe with the added feature
    '''
    articles_date_popularity = articles.select(['published_time', 'article_id']) \
        .group_by(pl.col('published_time').dt.date().alias('published_date')) \
        .agg(pl.col('article_id').count().alias('daily_articles')) \

    published_category_popularity = articles.select(['published_time', 'article_id', 'category']) \
        .group_by([pl.col('published_time').dt.date().alias('published_date'), 'category']) \
        .agg(pl.col('article_id').count().alias('category_daily_articles')) \
        .join(articles_date_popularity, on='published_date', how='left') \
        .with_columns((pl.col('category_daily_articles') / pl.col('daily_articles')).alias('category_daily_pct')) \
        .drop(['category_daily_articles', 'daily_articles'])

    df_features = df_features.with_columns(
        pl.col('impression_time').dt.date().alias('drop_me')) \
        .join(published_category_popularity, how='left', right_on=['published_date', 'category'],
              left_on=[pl.col('drop_me')- pl.duration(days=1), 'category']) \
        .rename({'category_daily_pct': 'yesterday_category_daily_pct'}) \
        .drop('drop_me') \
        .with_columns(pl.col('yesterday_category_daily_pct').fill_null(0))
    return reduce_polars_df_memory_size(df_features)


def add_session_features(df_features: pl.DataFrame, history: pl.DataFrame, behaviors: pl.DataFrame, articles: pl.DataFrame):
    '''
    Add the following session features to df_features:
    - last_session_nclicks: number of clicks in the last session
    - last_session_duration: duration in minutes of the last session
    - mean_prev_sessions_duration: mean duration of the previous sessions
    - last_session_time_hour_diff: hours since the last session appeared
    - is_new_article: True if the article was published after the last session
    - is_already_seen_article: True if the article appeared on the inview list in the last session
    - is_last_session_most_seen_category: True if the article has the same category of the most viewed 
      one by the user in the last session

    Args:
        df_features: the dataframe where the features will be added
        history: the history dataframe
        behaviors: the behaviors dataframe
        articles: the articles dataframe

    Returns:
        pl.DataFrame: the dataframe containing the new features
    '''
    last_history_df = history.with_columns(
        pl.col('impression_time_fixed').list.max().alias(
            'last_history_impression_time'),
        pl.col('article_id_fixed').list.tail(1).alias('last_history_article'),
    ).select(['user_id', 'last_history_impression_time', 'last_history_article'])

    last_session_time_df = behaviors.select(['session_id', 'user_id', 'impression_time', 'article_ids_inview']) \
        .group_by('session_id').agg(
            pl.col('user_id').first(),
            pl.col('impression_time').max().alias('session_time'),
            pl.col('article_ids_inview').flatten().alias('all_seen_articles'),
            (pl.col('impression_time').max() - pl.col('impression_time').min()
             ).dt.total_minutes().alias('session_duration'),
    ).with_columns(
            pl.col(['session_time', 'session_duration']).shift(
                1).over('user_id').name.prefix('last_'),
            pl.col('all_seen_articles').list.unique().shift(1).over('user_id'),
            pl.col('session_duration').rolling_mean(100, min_periods=1).over(
                'user_id').alias('mean_prev_sessions_duration'),
    ).with_columns(pl.col(['last_session_duration']).fill_null(0)) \
        .join(last_history_df, on='user_id', how='left') \
        .with_columns(
            pl.col('last_session_time').fill_null(
                pl.col('last_history_impression_time')),
            pl.col('all_seen_articles').fill_null(
                pl.col('last_history_article')),
    ).select(['session_id', 'last_session_time', 'last_session_duration',
              'all_seen_articles', 'mean_prev_sessions_duration'])

    gc.collect()

    df_features = df_features.join(last_session_time_df, on='session_id', how='left').with_columns(
        (pl.col('impression_time') - pl.col('last_session_time')
         ).dt.total_hours().alias('last_session_time_hour_diff'),
        ((pl.col('last_session_time') - pl.col('published_time')
          ).dt.total_hours() > 0).alias('is_new_article'),
        pl.col('all_seen_articles').list.contains(
            pl.col('article')).alias('is_already_seen_article'),
    ).drop(['published_time', 'session_id', 'all_seen_articles', 'last_session_time'])
    return reduce_polars_df_memory_size(df_features)


def _preprocessing_mean_delay_features(articles, history):
    topic_mean_delays = pl.concat(
        rows.select(["impression_time_fixed", "article_id_fixed"]).explode(
            ["impression_time_fixed", "article_id_fixed"])
        .join(other=articles.select(["article_id", "topics", "published_time"]), left_on="article_id_fixed", right_on="article_id", how="left")
        .drop("article_id_fixed").with_columns(
            (pl.col('impression_time_fixed') - pl.col('published_time')
             ).dt.total_days().alias('article_delay_days'),
            (pl.col('impression_time_fixed') - pl.col('published_time')
             ).dt.total_hours().alias('article_delay_hours')
        ).explode("topics").group_by("topics").agg(
            pl.col("article_delay_days").sum().alias("topic_sum_delay_days"),
            pl.col("article_delay_hours").sum().alias("topic_sum_delay_hours"),
            pl.col("article_delay_days").count().alias("len")
        )
        for rows in tqdm(history.iter_slices(1000), total=history.shape[0] // 1000)
    ).group_by("topics").agg(
        pl.col("topic_sum_delay_days").sum().truediv(
            pl.col("len").sum()).alias("topic_mean_delay_days"),
        pl.col("topic_sum_delay_hours").sum().truediv(
            pl.col("len").sum()).alias("topic_mean_delay_hours")
    ).drop(["topic_sum_delay_days", "topic_sum_delay_hours", "len"])

    user_mean_delays = history.select(["user_id", "impression_time_fixed", "article_id_fixed"]).explode(["impression_time_fixed", "article_id_fixed"]) \
        .join(other=articles.select(["article_id", "published_time"]), left_on="article_id_fixed", right_on="article_id", how="left") \
        .drop("article_id_fixed").with_columns(
        (pl.col('impression_time_fixed') - pl.col('published_time')
         ).dt.total_days().alias('article_delay_days'),
        (pl.col('impression_time_fixed') - pl.col('published_time')
         ).dt.total_hours().alias('article_delay_hours')
    ).group_by("user_id").agg(
        pl.col("article_delay_days").mean().alias("user_mean_delay_days"),
        pl.col("article_delay_hours").mean().alias("user_mean_delay_hours")
    )

    return topic_mean_delays, user_mean_delays


def add_mean_delays_features(df_features: pl.DataFrame, articles: pl.DataFrame, topic_mean_delays: pl.DataFrame, user_mean_delays: pl.DataFrame) -> pl.DataFrame:
    """
    Adds features concerning the delay in the dataframe.
    - mean_topics_mean_delay_days/mean_topics_mean_delay_hours: For each candidate article, the mean over all its topics 
    of the mean delays (days/hours) of the articles containing that topic, considering past interactions taken from the history.
    - user_mean_delay_days/user_mean_delay_hours: The mean delays (days/hours) of the user in his past interactions.


    Args:
        df_features: The dataset the new features will be added to.
        articles: The dataframe containing the articles.
        history: The dataframe containing the past interactions.
    Returns:
        pl.DataFrame: df_features enriched with the new features.
    """

    return df_features.join(other=articles.select(["article_id", "topics"]), left_on="article", right_on="article_id", how="left").explode("topics") \
        .join(other=topic_mean_delays, on="topics", how="left").group_by(["impression_id", "article", "user_id"]).agg(
        pl.exclude(["topic_mean_delay_days",
                   "topic_mean_delay_hours", "topics"]).first(),
        pl.col("topic_mean_delay_days").mean(),
        pl.col("topic_mean_delay_hours").mean()
    ).rename({"topic_mean_delay_days": "mean_topics_mean_delay_days", "topic_mean_delay_hours": "mean_topics_mean_delay_hours"}) \
        .join(other=user_mean_delays, on="user_id", how="left")


def _preprocessing_history_trendiness_scores(history, articles):

    history_trendiness_scores = pl.concat(
        rows.select(["user_id", "impression_time_fixed", "article_id_fixed"]).explode(
            ["impression_time_fixed", "article_id_fixed"])
        .rename({"impression_time_fixed": "impression_time", "article_id_fixed": "article"}).pipe(
            add_trendiness_feature, articles
        )
        for rows in tqdm(history.iter_slices(1000), total=history.shape[0] // 1000)
    )

    users_mean_trendiness_scores = history_trendiness_scores.select(["user_id", "trendiness_score"]).group_by("user_id").agg(
        pl.col("trendiness_score").mean().alias("mean_user_trendiness_score")
    )

    topics_mean_trendiness_scores = pl.concat(
        rows.select("article", "trendiness_score")
        .join(other=articles.select(["article_id", "topics"]), left_on="article", right_on="article_id", how="left")
        .explode("topics").group_by("topics").agg(
            pl.col("trendiness_score").sum().alias(
                "sum_topic_trendiness_score"),
            pl.col("trendiness_score").count().alias("len")
        )
        for rows in tqdm(history_trendiness_scores.iter_slices(1000), total=history_trendiness_scores.shape[0] // 1000)
    ).group_by("topics").agg(
        pl.col("sum_topic_trendiness_score").sum().truediv(
            pl.col("len").sum()).alias("mean_topic_trendiness_score")
    ).drop(["sum_topic_trendiness_score", "len"])

    return users_mean_trendiness_scores, topics_mean_trendiness_scores


def add_history_trendiness_scores_feature(df_features: pl.DataFrame, articles: pl.DataFrame, users_mean_trendiness_scores: pl.DataFrame, topics_mean_trendiness_scores: pl.DataFrame, topics) -> pl.DataFrame:
    """
    Adds 2 features concerning the trendiness, computed on the history, to the features dataframe.
    - mean_user_trendiness_score: For each user, the mean trendiness_score of the impressions in his history.
    - mean_topics_trendiness_score: For each article, the mean over all its topics, of the mean trendiness_scores of the interactions in the history
    with articles containin that topic.

    Args:
        df_features: The dataframe containing the features, to be enriched with the new features.
        history: The history dataframe.
        articles: The articles dataframe.
    Returns:
        pl.DataFrame: df_feature with the 2 features added. 
    """

    return df_features.join(other=users_mean_trendiness_scores, on="user_id", how="left") \
        .join(other=articles.select(["article_id", "topics"]), left_on="article", right_on="article_id", how="left") \
        .with_columns(
        [pl.col("topics").list.contains(topic).cast(
            pl.Int8).alias(f"{topic}_present") for topic in topics]
    ).with_columns(
        [pl.col(f"{topic}_present").mul(topics_mean_trendiness_scores.filter(pl.col("topics") == topic).select("mean_topic_trendiness_score"))
         .alias(f"mean_topic_{topic}_trendiness_score") for topic in topics]
    ).with_columns(
        pl.sum_horizontal([pl.col(f"mean_topic_{topic}_trendiness_score") for topic in topics]).truediv(
            pl.col("topics").list.len())
        .alias("mean_topics_trendiness_score")
    ).drop(
        [f"{topic}_present" for topic in topics]
    ).drop(
        [f"mean_topic_{topic}_trendiness_score" for topic in topics]
    ).drop("topics")


def _create_URM(history):
    """ 
    Helper function to create an URM starting from the dataset.
    We will create an URM with as rows the users, as items the items and 
    with a 0 if there is no interaction, else 1 if there is a click.
    Args:
        - history_train: history dataframe
    Returns:
        - scipy.sparse.coo_matrix: the created urm
        - df: item_mapping
        - df: user mapping
    """

    user_id_mapping = history.sort('user_id').with_row_index() \
        .select(['index', 'user_id']).rename({'index': 'user_index', 'user_id': 'UserID'})

    item_id_mapping = history.select('article_id_fixed').explode('article_id_fixed').unique(['article_id_fixed']).rename({'article_id_fixed': 'ItemID'})\
        .unique(['ItemID'])\
        .sort('ItemID').with_row_index() \
        .select(['index', 'ItemID']) \
        .rename({'index': 'item_index'})

    urm_all_interactions = history.select('user_id', 'article_id_fixed').explode(['article_id_fixed']).rename({'article_id_fixed': 'ItemID', 'user_id': 'UserID'})\
        .unique(['ItemID', 'UserID'])\
        .join(user_id_mapping, on='UserID')\
        .join(item_id_mapping, on='ItemID')\
        .select(['UserID', 'user_index', 'ItemID', 'item_index'])\
        .unique(['user_index', 'item_index'])

    URM_all = sps.csr_matrix((np.ones(urm_all_interactions.shape[0]),
                              (urm_all_interactions['user_index'].to_numpy(), urm_all_interactions['item_index'].to_numpy())),
                             shape=(user_id_mapping.shape[0], item_id_mapping.shape[0]))

    return URM_all, item_id_mapping, user_id_mapping


def _train_recsys_algorithms(URM_train, models_to_train, URM_val=None, evaluate=False):
    """
        Function used to fit recsys models specified in models_to_train,
        with URM_train.
        All the hyperparameters and the models are specified in

        ALGORITHMS in polimi.utils._custom.py

        Args:
            - URM_train: the Urm used to train the models
            - models_to_train: list of models to train

        Return:
            - dictionary with key the names of the models, and values the trained istances

    """
    trained_algorithms = {}
    for model in models_to_train:
        if evaluate and URM_val != None:
            rec_instance = ALGORITHMS[model][0](URM_train)
            print("Training {} ...".format(model))
            rec_instance.fit()
            print("Evaluating {}".format(model))
            evaluator = EvaluatorHoldout(
                URM_val, cutoff_list=[10], exclude_seen=False)
            result_df, _ = evaluator.evaluateRecommender(rec_instance)
            print(result_df.loc[10]["MAP"])
            print("Retraining on all the URM...")
            rec_instance = ALGORITHMS[model][0](URM_train+URM_val)
            print("Training {} ...".format(model))
            rec_instance.fit()
            trained_algorithms[model] = rec_instance
        else:
            rec_instance = ALGORITHMS[model][0](URM_train)
            print("Training {} ...".format(model))
            rec_instance.fit()
            trained_algorithms[model] = rec_instance

    return trained_algorithms


def get_recommender_scores(user_items_df, recommenders):
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


def add_other_rec_features(ds, history, algorithms, evaluate=False):
    """
    For each impression (user_id, article_id) add a feature that is the prediction computed by the models specified in algorithms
    trained on the URMs created on history.

    Args:
        - ds: the dataframe to enrich with one feature for each algorithm
        - history : the history dataframe
        - algorithms: list of models to train and to compute the predictions
        - evaluate: boolean to also evaluate each model trained

    Returns:
        pl.DataFrame: the enriched dataframe
    """

    URM_all, item_mapping, user_mapping = _create_URM(history)

    if evaluate:
        URM_train, URM_val = split_train_in_two_percentage_global_sample(
            URM_all, train_percentage=0.80)
        trained_algorithms = _train_recsys_algorithms(
            URM_train=URM_train, URM_val=URM_val, models_to_train=algorithms, evaluate=evaluate)
    else:
        trained_algorithms = _train_recsys_algorithms(
            URM_train=URM_all, models_to_train=algorithms)

    ds = ds.join(item_mapping, left_on='article', right_on='ItemID')\
        .join(user_mapping, left_on='user_id', right_on='UserID')\
        .sort(['user_index', 'item_index'])\
        .group_by('user_index').map_groups(lambda df: df.pipe(get_recommender_scores, recommenders=trained_algorithms))

    return ds


def _preprocessing_window_features(history: pl.DataFrame, articles: pl.DataFrame,):

    # windows = [[5,8],[7,10],[9,12],[11,15],[14,18],[17,21],[20,23],[22,5]]
    # windows = [[5,10],[9,13],[12,15],[14,18],[17,20],[19,23],[22,6]]
    windows = [[5, 13], [12, 19], [18, 23], [22, 6]]
    # windows = [[5, 13], [12, 23], [22, 6]]

    user_windows = history.select(["user_id", "impression_time_fixed", "article_id_fixed"]).explode(['impression_time_fixed', 'article_id_fixed']) \
        .rename({'impression_time_fixed': 'impression_time'}) \
        .drop('article_id_fixed').with_columns(
        [
            pl.when(window[0] < window[1]).then(pl.col('impression_time').dt.time().is_between(time(window[0]), time(window[1]), closed='left')).otherwise(
                pl.col('impression_time').dt.time().ge(time(window[0])).or_(
                    pl.col('impression_time').dt.time().lt(time(window[1])))
            ).cast(pl.Int8).alias(f'is_inside_window_{index}') for index, window in enumerate(windows)]
    ).group_by('user_id').agg(
        [pl.col(f'is_inside_window_{index}').sum().alias(
            f'window_{index}_history_length') for index, window in enumerate(windows)]
    )

    user_topics_windows = pl.concat(
        rows.select(["user_id", "impression_time_fixed", "article_id_fixed"]).explode(
            ['impression_time_fixed', 'article_id_fixed'])
        .join(other=articles.select(['article_id', 'topics']), left_on='article_id_fixed', right_on='article_id', how='left')
        .rename({'impression_time_fixed': 'impression_time'})
        .with_columns(
            [
                pl.when(window[0] < window[1]).then(pl.col('impression_time').dt.time().is_between(time(window[0]), time(window[1]), closed='left')).otherwise(
                    pl.col('impression_time').dt.time().ge(time(window[0])).or_(
                        pl.col('impression_time').dt.time().lt(time(window[1])))
                ).cast(pl.Int8).alias(f'is_inside_window_{index}') for index, window in enumerate(windows)]
        ).drop(['article_id_fixed', 'impression_time']).explode('topics')
        .group_by(['user_id', 'topics']).agg(
            [pl.col(f'is_inside_window_{index}').sum().alias(
                f'window_{index}_score') for index, window in enumerate(windows)]
        )
        for rows in tqdm(history.iter_slices(1000), total=history.shape[0] // 1000)
    )

    user_category_windows = history.select(["user_id", "impression_time_fixed", "article_id_fixed"]) \
        .explode(['impression_time_fixed', 'article_id_fixed']) \
        .join(other=articles.select(['article_id', 'category']), left_on='article_id_fixed', right_on='article_id', how='left') \
        .rename({'impression_time_fixed': 'impression_time'}) \
        .with_columns(
            [
                pl.when(window[0] < window[1]).then(pl.col('impression_time').dt.time().is_between(time(window[0]), time(window[1]), closed='left')).otherwise(
                    pl.col('impression_time').dt.time().ge(time(window[0])).or_(
                        pl.col('impression_time').dt.time().lt(time(window[1])))
                ).cast(pl.Int8).alias(f'is_inside_window_{index}') for index, window in enumerate(windows)]
    ).drop(['article_id_fixed', 'impression_time']) \
        .group_by(['user_id', 'category']).agg(
            [pl.col(f'is_inside_window_{index}').sum().alias(
                f'window_{index}_score') for index, window in enumerate(windows)]
    )

    return windows, user_windows, user_topics_windows, user_category_windows


def add_window_features(df_features: pl.DataFrame, articles: pl.DataFrame, user_windows: pl.DataFrame, user_category_windows: pl.DataFrame, user_topics_windows: pl.DataFrame, windows) -> pl.DataFrame:
    """
    Given each impression with its timestamp, it assigns to it:
     - Its time_window, thanks to the is_inside_window_{index} features
     - The length of the user history divided for each time window, thanks to window_{index}_history_length features
     - How many articles with at least one of the topics of the candidate article, the user has clicked in that time_window,
        thanks to window_topics_score feature
    - How many articles with the same category as the candidate article, the user has clicked in that time_window, 
        thanks to window_category_score feature

    Args:
        df_features: The features dataframe to be enriched with the time window features
        history: The history dataframe
        articles: The articles dataframe.
    Returns:
        pl.DataFrame: The df_features with the new features.
    """

    return df_features.join(other=user_windows, on='user_id', how='left').with_columns(
        [
            pl.when(window[0] < window[1]).then(pl.col('impression_time').dt.time().is_between(time(window[0]), time(window[1]), closed='left')).otherwise(
                pl.col('impression_time').dt.time().ge(time(window[0])).or_(
                    pl.col('impression_time').dt.time().lt(time(window[1])))
            ).cast(pl.Int8).alias(f'is_inside_window_{index}') for index, window in enumerate(windows)]
    ).join(other=articles.select(['article_id', 'topics']), left_on='article', right_on='article_id', how='left') \
        .join(other=user_category_windows, on=['user_id', 'category'], how='left') \
        .with_columns(
            pl.max_horizontal([pl.col(f'window_{index}_score').mul(pl.col(
                f'is_inside_window_{index}')) for index, window in enumerate(windows)]).alias('score')
    ).group_by(['impression_id', 'article', 'user_id']).agg(
            pl.exclude(["impression_id", "article",
                       "user_id", "score"]).first(),
            pl.col('score').sum().alias('window_category_score')
    ).drop([f'window_{index}_score'for index, window in enumerate(windows)]) \
        .explode('topics') \
        .join(other=user_topics_windows, on=['user_id', 'topics'], how='left') \
        .with_columns(
            pl.max_horizontal([pl.col(f'window_{index}_score').mul(pl.col(
                f'is_inside_window_{index}')) for index, window in enumerate(windows)]).alias('score')
    ).drop([f'window_{index}_score'for index, window in enumerate(windows)]) \
        .drop('topics') \
        .group_by(['impression_id', 'article', 'user_id']).agg(
            pl.exclude(["impression_id", "article",
                       "user_id", "score"]).first(),
            pl.col('score').sum().alias('window_topics_score')
    )


def add_trendiness_feature_categories(df_features: pl.DataFrame, articles: pl.DataFrame, period: str = "3d") -> pl.DataFrame:
    """
    Adds a new feature "trendiness_score_category" to each impression.
    The trendiness score by category for an article is computed as the number of the times that category has happeared in some article
    published in the previous <period> before the impression.

    Args:
        df_features: The dataset to be enriched with the new feature. It MUST contain the "impression_time" and the "category"
        articles: The articles dataset with the topics of the articles and their publication time.
        period: The window size for the computation of the scores, in string encoding 
            (ex. 1ns, 1us, 1s, 1m, 1h, 1d, 1w,.. or combinations like "3d12h4m25s". See doc for polars.DataFrame.rolling for more details)

    Returns:
        pl.DataFrame: The training dataset enriched with a new column, containing the trendiness_score by category.
    """

    categories = articles.select("category").unique()
    categories = [category for category in categories["category"]
                  if category is not None]

    categories_popularity = articles.select(["published_time", "category"]).with_columns(
        pl.col("published_time").dt.date().alias("published_date")
    ).drop("published_time").group_by("published_date").agg(
        pl.col("category").alias("categories")
    ).sort("published_date").set_sorted("published_date").upsample(time_column="published_date", every="1d") \
        .rolling(index_column="published_date", period=period).agg(
        [pl.col("categories").list.count_matches(category).sum().alias(
            f'{category}_matches') for category in categories]
    )

    return df_features.with_columns(
        pl.col("impression_time").dt.date().alias("impression_date")
    ).with_columns(
        [pl.col("category").eq(category).cast(pl.Int8).alias(
            f"{category}_present") for category in categories]
    ).join(other=categories_popularity, left_on=pl.col("impression_date"), right_on=(pl.col("published_date")+pl.duration(days=1)), how="left") \
        .with_columns(
        [pl.col(f"{category}_present").mul(pl.col(f"{category}_matches")).alias(
            f"trendiness_score_{category}") for category in categories]
    ).with_columns(
        pl.sum_horizontal([pl.col(f"trendiness_score_{category}") for category in categories]).alias(
            "trendiness_score_category"),
    ).drop(
        [f"trendiness_score_{category}" for category in categories]
    ).drop(
        [f"{category}_matches" for category in categories]
    ).drop(
        [f"{category}_present" for category in categories]
    ).drop("impression_date")


def _build_last_n_topics_tf_idf(history, articles, vectorizer, last_n=5) -> pl.DataFrame:
    '''
    Builds the last n topics tf-idf for each user in the history dataframe

    Args:
        history: the history dataframe
        articles: the articles dataframe
        vectorizer: the vectorizer used to compute the tf-idf
        last_n: the number of last articles to consider

    Returns:
        pl.DataFrame: the dataframe containing the last n topics tf-idf
    '''
    return pl.concat(
        rows.select(['user_id', 'article_id_fixed']).with_columns(
            pl.col('article_id_fixed').list.slice(-last_n,
                                                  last_n).alias('last_five_elements')
        ).drop('article_id_fixed')
        .explode('last_five_elements')
        .join(articles.select(['article_id', 'topics']), left_on='last_five_elements', right_on='article_id', how='left')
        .group_by('user_id').agg(
            pl.col('topics').flatten().alias(
                'topics_flatten')
        )
        .pipe(_add_topics_tf_idf_columns, topics_col='topics_flatten', vectorizer=vectorizer, col_name=f'last_{last_n}_tf_idf').drop('topics_flatten')
        .join(history, on='user_id')
        for rows in tqdm(history.iter_slices(1000), total=history.shape[0] // 1000))


def _last_n_topics_tf_idf_distance(df, articles, last_n_tf_idf, last_n=5):
    '''
    Computes the distance between the last n topics tf-idf and the topics tf-idf of the articles

    Args:
        df: the dataframe to be enriched with the new feature
        articles: the articles dataframe
        last_n_tf_idf: the dataframe containing the last n topics tf-idf
        last_n: the number of last articles to consider (used to naming the new feature)

    Returns:
        pl.DataFrame: the dataframe containing distance between the last n topics tf-idf and the topics tf-idf of the articles
    '''
    return pl.concat(
        rows.select(['impression_id', 'user_id', 'article'])
        .join(articles.select(['article_id', 'topics_idf']), left_on='article', right_on='article_id')
        .join(last_n_tf_idf.select(['user_id', f'last_{last_n}_tf_idf']), on='user_id')
        .with_columns(
            pl.struct(['topics_idf', f'last_{last_n}_tf_idf']).map_elements(
                lambda x: fast_distance(x['topics_idf'], x[f'last_{last_n}_tf_idf']), return_dtype=pl.Float64).cast(pl.Float32).alias(f'last_{last_n}_topics_cosine')
        )
        .drop(['topics_idf', f'last_{last_n}_tf_idf'])
        for rows in tqdm.tqdm(df.iter_slices(1000), total=df.shape[0] // 1000))\
        .unique()


def build_last_n_topics_distances(history, articles, df, vectorizer, last_n) -> pl.DataFrame:
    '''
    Builds the last n topics similarities for each user in the history dataframe

    Args:
        history: the history dataframe
        articles: the articles dataframe
        df: the dataframe to be enriched with the new feature
        vectorizer: the vectorizer used to compute the tf-idf
        last_n: array containing the number of last articles to consider

    Returns:
        pl.DataFrame: the dataframe containing the last n topics similarities
    '''
    last_n_topics_cosine = df.select(['impression_id', 'user_id', 'article'])

    for n in last_n:
        last_n_tf_idf = _build_last_n_topics_tf_idf(
            history, articles, vectorizer, last_n=n)
        last_n_topics_cosine = _last_n_topics_tf_idf_distance(last_n_topics_cosine, articles, last_n_tf_idf, last_n=n)\
            .join(last_n_topics_cosine, on=['impression_id', 'user_id', 'article'], how='left')
        del last_n_tf_idf

    return df.join(last_n_topics_cosine, on=['impression_id', 'user_id', 'article'], how='left')


"""
OLD IMPLEMENTATION:
def add_article_endorsement_feature(df_features: pl.DataFrame, period: str = "10h") -> pl.DataFrame:
    
    Adds a feature which is a count of how many times that article has been proposed to a user in the last <period> hours.
    Args:
        df_features: The dataframe to be enriched with the new feature.
        period: The window size for the computation of the scores, in string encoding 
            (ex. 1ns, 1us, 1s, 1m, 1h, 1d, 1w,.. or combinations like "3d12h4m25s". See doc for polars.DataFrame.rolling for more details)
    Returns:
        pl.DataFrame: The dataframe with the new feature added

    
    endorsement = df_features.select(['impression_time', 'article']) \
        .sort('impression_time') \
        .set_sorted('impression_time') \
        .with_columns(
        pl.lit(1).alias(f'endorsement_{period}')
    ).rolling(index_column='impression_time', period=period, by='article') \
        .agg(
        pl.col(f'endorsement_{period}').count()
    ).unique()

    return df_features.join(other=endorsement, on=['impression_time', 'article'], how='left')

"""


def _preprocessing_article_endorsement_feature(behaviors, period, batch_dim=10000):
    # slice the behaviors dataframe in windows based on the article_id_inview
    exploded_behaviors = behaviors.select(
        ['impression_time', 'article_ids_inview']).explode("article_ids_inview")

    # get min and max id of the articles
    min_article_id = exploded_behaviors.select(
        "article_ids_inview").min().item()
    max_article_id = exploded_behaviors.select(
        "article_ids_inview").max().item()
    diff = max_article_id - min_article_id

    n_batch = diff // batch_dim
    if diff % batch_dim != 0:  # If there are remaining items
        n_batch += 1 
        
    # return pl.concat(
    #     exploded_behaviors.filter(
    #         pl.col("article_ids_inview").ge(min_article_id + i * batch_dim)
    #         .and_(pl.col("article_ids_inview").lt(min_article_id + (i + 1) * batch_dim))) \
    #     .sort("impression_time").set_sorted("impression_time")
    #     .with_columns(
    #         pl.lit(1).alias(f'endorsement_{period}'))
    #     .rename({'article_ids_inview': 'article'})
    #     .rolling(index_column="impression_time", period=period, by='article').agg(
    #         pl.col(f'endorsement_{period}').count()
    #     ).unique(["article","impression_time"])
    #     for i in tqdm(range(n_batch))
    # )
    
    return pl.concat(
        exploded_behaviors.filter(
            pl.col("article_ids_inview").ge(min_article_id + i * batch_dim)
            .and_(pl.col("article_ids_inview").lt(min_article_id + (i + 1) * batch_dim)))\
        .with_columns(pl.col('impression_time').dt.round('1m').alias('impression_time_rounded'))\
        .group_by(['impression_time_rounded','article_ids_inview']).len()\
        .rename({'impression_time_rounded': 'impression_time', 'len':f'endorsement_{period}'}) \
        .sort("impression_time").set_sorted("impression_time") \
        .rename({'article_ids_inview': 'article'})
        .rolling(index_column="impression_time", period=period, group_by='article').agg(
            pl.col(f'endorsement_{period}').sum()
        ).unique(["article","impression_time"]) \
        for i in tqdm(range(n_batch))                    
    )
    

    # df1 = behaviors.select(["impression_time", "article_ids_inview"]) \
    #     .sort("impression_time").set_sorted("impression_time") \
    #     .rolling(index_column="impression_time", period="10h").agg(
    #     pl.col("article_ids_inview").flatten().value_counts()
    # ).unique("impression_time")

    # return pl.concat(
    #     rows.explode("article_ids_inview").unnest("article_ids_inview")
    #     .rename({"article_ids_inview": "article", "count": f"endorsement_{period}"})
    #     for rows in tqdm.tqdm(df1.iter_slices(10), total=df1.shape[0] // 10)
    # )
    
    
def _preprocessing_normalize_endorsement(articles_endorsement_raw: pl.DataFrame, endorsement_col='endorsement_10h'):
    return articles_endorsement_raw.sort(by='impression_time').with_columns(
        (
            pl.col(endorsement_col) / 
            pl.col(endorsement_col).sum().over('impression_time')
        ).alias(f'normalized_{endorsement_col}'),
        (
            pl.col(endorsement_col) - 
            pl.col(endorsement_col).rolling_mean(10, min_periods=1).over('article')
        ).alias(f'{endorsement_col}_diff_rolling'),
        (
            pl.col(endorsement_col).rolling_mean(5, min_periods=1).over('article') - 
            pl.col(endorsement_col).rolling_mean(10, min_periods=1).over('article')
        ).alias(f'{endorsement_col}_macd'),
        (
            pl.col(endorsement_col) / 
            pl.col(endorsement_col).quantile(0.8).over('impression_time')
        ).alias(f'{endorsement_col}_quantile_norm')
    ).with_columns(
        (
            pl.col(f'normalized_{endorsement_col}') / 
            pl.col(f'normalized_{endorsement_col}').rolling_max(10, min_periods=1).over('impression_time')
        ).alias(f'normalized_{endorsement_col}_rolling_max_ratio'),
    )


def add_article_endorsement_feature(df_features, articles_endorsement):
    # old return df_features.join(other=articles_endorsement, on=["article", "impression_time"], how="left")
    articles_endorsement = articles_endorsement.with_columns(pl.col('article').cast(pl.Int32))
    return df_features.with_columns(pl.col('impression_time').dt.round('1m').alias('rounded_impression_time'))\
                         .join(articles_endorsement.rename({'impression_time' : 'rounded_impression_time'}), on=['article','rounded_impression_time'], how='left')\
                         .drop('rounded_impression_time')

def _preprocessing_normalize_endorsement_by_article_and_user(articles_endorsement_articleuser_raw:pl.DataFrame,endorsement_col='endorsement_20h_articleuser' ):
    return articles_endorsement_articleuser_raw.sort(by=['user_id','impression_time']).with_columns(
        (
            pl.col(endorsement_col) / 
            pl.col(endorsement_col).sum().over(['user_id','impression_time'])
        ).alias(f'normalized_{endorsement_col}'),
        (
            pl.col(endorsement_col) - 
            pl.col(endorsement_col).rolling_mean(10, min_periods=1).over(['article','user_id'])
        ).alias(f'{endorsement_col}_diff_rolling'),
        (
            pl.col(endorsement_col).rolling_mean(5, min_periods=1).over(['article','user_id']) - 
            pl.col(endorsement_col).rolling_mean(10, min_periods=1).over(['article','user_id'])
        ).alias(f'{endorsement_col}_macd'),
        (
            pl.col(endorsement_col) / 
            pl.col(endorsement_col).quantile(0.8).over(['user_id','impression_time'])
        ).alias(f'{endorsement_col}_quantile_norm')
    ).with_columns(
        (
            pl.col(endorsement_col) / 
            pl.col(endorsement_col).rolling_max(10, min_periods=1).over(['user_id','impression_time'])
        ).alias(f'normalized_{endorsement_col}_rolling_max_ratio'),
    )
    

def get_unique_categories(articles: pl.DataFrame):
    '''
    The function returns a dictionary containing the entities of the articles

    Args:
        articles: the articles dataframe

    Returns:
        dict: the dictionary containing the entities of the articles
    '''
    unique_categories_df = articles.select(
        ['category', 'category_str']).unique('category').drop_nulls('category')
    unique_categories = {
        row['category']: row['category_str'] for row in unique_categories_df.iter_rows(named=True)
    }
    del unique_categories_df

    return unique_categories

def subsample_dataset(original_datset_path: str, dataset_path : str, new_path: str, npratio: int = 2):
    starting_dataset =  pl.read_parquet(original_datset_path).select(['impression_id','user_id','article_ids_inview','article_ids_clicked'])
    dataset = pl.read_parquet(dataset_path)
    
    behaviors = pl.concat(
        rows.pipe(
            sampling_strategy_wu2019, npratio=npratio, shuffle=False, with_replacement=True, seed=123
        ).explode('article_ids_inview').drop(columns = 'article_ids_clicked').rename({'article_ids_inview' : 'article'})\
        .with_columns(pl.col('user_id').cast(pl.UInt32),
                      pl.col('article').cast(pl.Int32))\
        
         for rows in tqdm(starting_dataset.iter_slices(1000), total=starting_dataset.shape[0] // 1000)
    )
        
    behaviors.join(dataset, on = ['impression_id','user_id','article'], how = 'left').write_parquet(new_path)
    
    
def subsample_fold(train_behaviors_path: str, val_behaviors_path: str, dataset_path : str, new_path: str, npratio: int = 2):
    behaviors_train =  pl.read_parquet(train_behaviors_path).select(['impression_id','user_id','article_ids_inview','article_ids_clicked'])
    behaviors_val =  pl.read_parquet(val_behaviors_path).select(['impression_id','user_id','article_ids_inview','article_ids_clicked'])
    dataset = pl.read_parquet(dataset_path)
    
    behaviors_train = pl.concat(
        rows.pipe(
            sampling_strategy_wu2019, npratio=npratio, shuffle=False, with_replacement=True, seed=123
        ).explode('article_ids_inview').drop(columns = 'article_ids_clicked').rename({'article_ids_inview' : 'article'})\
        .with_columns(pl.concat_str([pl.col('user_id').cast(pl.String), pl.lit('1')], separator='_').alias('user_id'),
                      pl.concat_str([pl.col('impression_id').cast(pl.String), pl.lit('1')], separator='_').alias('impression_id'),
                      pl.col('article').cast(pl.Int32))\
        
         for rows in tqdm(behaviors_train.iter_slices(1000), total=behaviors_train.shape[0] // 1000)
    )
    
    behaviors_val = pl.concat(
        rows.pipe(
            sampling_strategy_wu2019, npratio=npratio, shuffle=False, with_replacement=True, seed=123
        ).explode('article_ids_inview').drop(columns = 'article_ids_clicked').rename({'article_ids_inview' : 'article'})\
        .with_columns(pl.concat_str([pl.col('user_id').cast(pl.String), pl.lit('2')], separator='_').alias('user_id'),
                      pl.concat_str([pl.col('impression_id').cast(pl.String), pl.lit('2')], separator='_').alias('impression_id'),
                      pl.col('article').cast(pl.Int32))\
        
         for rows in tqdm(behaviors_val.iter_slices(1000), total=behaviors_val.shape[0] // 1000)
    )
       
    # doing inner join since some data might be on the fold test set 
    fold_part_1 = behaviors_train.join(dataset, on = ['impression_id','user_id','article'], how = 'inner')
    fold_part_2 = behaviors_val.join(dataset, on = ['impression_id','user_id','article'], how = 'inner')
    print(fold_part_1.head())
    print(fold_part_2.head())
    assert dataset.unique('impression_id').shape[0] == fold_part_1.unique('impression_id').shape[0] + fold_part_2.unique('impression_id').shape[0]
    if 'impression_time' in dataset.columns:
        # for sure impression_time should not have nulls if the join is correct
        assert fold_part_1.select(pl.col('impression_time').is_null().sum()).item(0,0) == 0
        assert fold_part_2.select(pl.col('impression_time').is_null().sum()).item(0,0) == 0
    pl.concat([fold_part_1, fold_part_2], how='diagonal_relaxed').write_parquet(new_path)
    

def _preprocessing_article_endorsement_feature_by_article_and_user(behaviors, period, batch_dim=10000):
    # slice the behaviors dataframe in windows based on the article_id_inview
    exploded_behaviors = behaviors.select(
        ['impression_time', 'article_ids_inview','user_id']).explode("article_ids_inview")

    # get min and max id of the articles
    min_article_id = exploded_behaviors.select(
        "article_ids_inview").min().item()
    max_article_id = exploded_behaviors.select(
        "article_ids_inview").max().item()
    diff = max_article_id - min_article_id

    n_batch = diff // batch_dim
    if diff % batch_dim != 0:  # If there are remaining items
        n_batch += 1 
        
    
    return pl.concat(
        exploded_behaviors.filter(
            pl.col("article_ids_inview").ge(min_article_id + i * batch_dim)
            .and_(pl.col("article_ids_inview").lt(min_article_id + (i + 1) * batch_dim)))\
        .with_columns(pl.col('impression_time').dt.round('1m').alias('impression_time_rounded'))\
        .group_by(['impression_time_rounded','article_ids_inview','user_id']).len()\
        .rename({'impression_time_rounded': 'impression_time', 'len':f'endorsement_{period}_articleuser'}) \
        .sort("impression_time").set_sorted("impression_time") \
        .rename({'article_ids_inview': 'article'})
        .rolling(index_column="impression_time", period=period, group_by=['article','user_id']).agg(
            pl.col(f'endorsement_{period}_articleuser').sum()
        ).unique(["article","impression_time","user_id"]) \
        for i in tqdm(range(n_batch))                    
    )
    


def add_article_endorsement_feature_by_article_and_user(df_features, articles_endorsement_articleuser):
    
    articles_endorsement_articleuser = articles_endorsement_articleuser.with_columns(pl.col('article').cast(pl.Int32))
    return df_features.with_columns(pl.col('impression_time').dt.round('1m').alias('rounded_impression_time'))\
                         .join(articles_endorsement_articleuser.rename({'impression_time' : 'rounded_impression_time'}), on=['user_id','article','rounded_impression_time'], how='left')\
                         .drop('rounded_impression_time')

def add_article_endorsement_feature_leak(df_features:pl.DataFrame, articles_endorsement:pl.DataFrame,period:str='10h'):
    # old return df_features.join(other=articles_endorsement, on=["article", "impression_time"], how="left")
    articles_endorsement = articles_endorsement.with_columns(pl.col('article').cast(pl.Int32)).rename({f'endorsement_{period}':f'endorsement_{period}_leak'}) \
                            .with_columns(
                                pl.col('impression_time') - pl.duration(hours=int(period[:-1]))
                            ).sort('impression_time').set_sorted('impression_time')
    
    return df_features.with_columns(pl.col('impression_time').dt.round('1m').alias('rounded_impression_time'))\
                        .sort('rounded_impression_time').set_sorted('rounded_impression_time') \
                         .join_asof(articles_endorsement.rename({'impression_time' : 'rounded_impression_time'}),by='article', left_on='rounded_impression_time',right_on='rounded_impression_time', strategy='nearest')\
                         .drop('rounded_impression_time') \
                        .with_columns(
                            pl.col(f'endorsement_{period}_leak').fill_null(0)
                        )

def add_trendiness_feature_leak(df_features: pl.DataFrame, articles: pl.DataFrame, period: str = "3d"):

    topics = articles.select("topics").explode("topics").unique()
    topics = [topic for topic in topics["topics"] if topic is not None]
    # min_impression_time = df_features.select(pl.col("impression_time")).min().item()

    # topics_total_publications= articles.filter(pl.col("published_time")< min_impression_time ).select("topics") \
    # .explode("topics").group_by("topics").len()

    topics_popularity = articles.select(["published_time", "topics"]).with_columns(
        pl.col("published_time").dt.date().alias("published_date")
    ).drop("published_time").group_by("published_date").agg(
        pl.col("topics").flatten()
    ).sort("published_date").set_sorted("published_date").upsample(time_column="published_date", every="1d") \
        .rolling(index_column="published_date", period=period).agg(
        [pl.col("topics").list.count_matches(topic).sum().alias(
            f"{topic}_matches") for topic in topics]
    )

    return df_features.with_columns(
        pl.col("impression_time").dt.date().alias("impression_date")
    ).join(other=articles.select(["article_id", "topics"]), left_on="article", right_on="article_id", how="left") \
        .with_columns(
        [pl.col("topics").list.contains(topic).cast(
            pl.Int8).alias(f"{topic}_present") for topic in topics]
    ).sort('impression_date').set_sorted('impression_date') \
    .join_asof(topics_popularity.with_columns(pl.col('published_date') + pl.duration(days=1) - pl.duration(days=int(period[:-1]))) \
               .sort('published_date').set_sorted('published_date'),
               left_on="impression_date", right_on="published_date", strategy="nearest") \
        .with_columns(
        [pl.col(f"{topic}_present").mul(pl.col(f"{topic}_matches")).alias(
            f"trendiness_score_{topic}") for topic in topics]
    ).with_columns(
        pl.sum_horizontal([pl.col(f"trendiness_score_{topic}") for topic in topics]).alias(
            "trendiness_score_leak"),
    ).drop(
        [f"trendiness_score_{topic}" for topic in topics]
    ).drop(
        [f"{topic}_matches" for topic in topics]
    ).drop(
        [f"{topic}_present" for topic in topics]
    ).drop(["topics", "impression_date","published_date"])

def _preprocessing_leak_counts(history, behaviors):
    history_counts=history.select(['article_id_fixed']).explode('article_id_fixed').group_by('article_id_fixed').len() \
    .rename({'article_id_fixed':'article','len':'clicked_count'})

    behaviors_counts=behaviors.select('article_ids_inview').explode('article_ids_inview').group_by('article_ids_inview').len() \
    .rename({'article_ids_inview':'article','len':'inview_count'})

    return history_counts, behaviors_counts