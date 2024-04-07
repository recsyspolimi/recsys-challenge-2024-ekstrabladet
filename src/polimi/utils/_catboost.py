from tqdm import tqdm
from rich.progress import Progress
from scipy import stats
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing_extensions import Tuple, List, Dict
import logging

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
                              cols_explode: List[str], rename_columns: Dict[str, str], unique_entities: List[str]):  
    return behaviors.select(['impression_id', 'article_ids_inview', 'impression_time', 'labels', 
                                    'device_type', 'read_time', 'scroll_percentage', 'user_id', 'is_sso_user', 'gender',
                                    'age', 'is_subscriber', 'session_id']) \
        .with_columns(pl.col('gender').fill_null(2)) \
        .explode(cols_explode) \
        .rename(rename_columns) \
        .with_columns(pl.col('article').cast(pl.Int32)) \
        .pipe(add_trendiness_feature, articles=articles, period='3d') \
        .unique(['impression_id', 'article']) \
        .with_columns(
            pl.col('impression_time').dt.weekday().alias('weekday'),
            pl.col('impression_time').dt.hour().alias('hour'),
            pl.col('article').cast(pl.Int32),
        ).join(articles.select(['article_id', 'premium', 'published_time', 'category',
                                'sentiment_score', 'sentiment_label', 'entity_groups',
                                'num_images', 'title_len', 'subtitle_len', 'body_len']),
               left_on='article', right_on='article_id', how='left') \
        .with_columns(
            (pl.col('impression_time') - pl.col('published_time')).dt.total_days().alias('article_delay_days'),
            (pl.col('impression_time') - pl.col('published_time')).dt.total_hours().alias('article_delay_hours')
        ).with_columns(
            pl.col('entity_groups').list.contains(entity).alias(f'Entity_{entity}_Present')
            for entity in unique_entities
        ).drop('entity_groups') \
        .pipe(add_session_features, history=history, behaviors=behaviors, articles=articles) \
        .pipe(add_category_popularity, articles=articles) \
        .pipe(_join_history, history=history, articles=articles)


def _preprocessing(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame, 
                   test: bool = False, sample: bool = True, npratio: int = 2,):
    if not test and sample:
        behaviors = behaviors.pipe(
            sampling_strategy_wu2019, npratio=npratio, shuffle=False, with_replacement=True, seed=123
        )
        
    if not test:
        behaviors = behaviors.pipe(create_binary_labels_column, shuffle=True, seed=123) 
        columns_to_explode = ['article_ids_inview', 'labels']
        renaming_columns = {'article_ids_inview': 'article', 'labels': 'target'}
    else:
        columns_to_explode = 'article_ids_inview'
        renaming_columns = {'article_ids_inview': 'article'}
        
    articles, tf_idf_vectorizer, unique_entities = _preprocess_articles(articles)
    history = _build_history_features(history, articles, unique_entities, tf_idf_vectorizer)
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
    unique_entities = articles.select('entity_groups').explode('entity_groups')['entity_groups'].unique().to_list()
    unique_entities = [e for e in unique_entities if e is not None]
    
    vectorizer = TfidfVectorizer()
    articles = articles.with_columns(
        pl.col('image_ids').list.len().alias('num_images'),
        pl.col('title').str.split(by=' ').list.len().alias('title_len'),
        pl.col('subtitle').str.split(by=' ').list.len().alias('subtitle_len'),
        pl.col('body').str.split(by=' ').list.len().alias('body_len'),
        # useful for saving memory when joining with the history dataframe
        pl.when(pl.col('sentiment_label') == 'Negative').then(-1) \
            .otherwise(
                pl.when(pl.col('sentiment_label') == 'Positive').then(1).otherwise(0)
            ).cast(pl.Int8).alias('sentiment_label_int'),
        (pl.col('article_type') == 'article_default').cast(pl.UInt8).alias('is_article_default'),
        # very important for tf-idf, otherwise multiple tokens for topics with spaces are built
        pl.col('topics').list.eval(pl.element().str.split(by=' ').list.join('_')),
        pl.Series(
            vectorizer.fit_transform(
                articles.with_columns(pl.col('topics').list.join(separator=' '))['topics'].to_list()
            ).toarray().astype(np.float32)
        ).alias('topics_idf')
    )
    return articles, vectorizer, unique_entities


def _add_topics_tf_idf_columns(df, topics_col, vectorizer):
    return df.with_columns(
        pl.Series(
            vectorizer.transform(
                df.with_columns(pl.col(topics_col).list.join(separator=' '))[topics_col].to_list()
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
        rows.with_columns(pl.col('article_id_fixed').list.len().alias('NumArticlesHistory')) \
            .explode(['article_id_fixed', 'impression_time_fixed', 'read_time_fixed', 'scroll_percentage_fixed']) \
            .sort(by=['user_id', 'impression_time_fixed']) \
            .with_columns(
                pl.col('impression_time_fixed').dt.weekday().alias('weekday'),
                pl.col('impression_time_fixed').dt.hour().alias('hour'),
            ).join(articles.select(['article_id', 'category', 'is_article_default', 'sentiment_label_int', 
                                    'sentiment_score', 'entity_groups', 'topics']), 
                left_on='article_id_fixed', right_on='article_id', how='left') \
            .with_columns(
                (pl.col('sentiment_label_int') == 0).alias('is_neutral'),
                (pl.col('sentiment_label_int') == 1).alias('is_positive'),
                (pl.col('sentiment_label_int') == -1).alias('is_negative'),
                ((pl.col('sentiment_label_int') == 0) & (pl.col('sentiment_score') > strong_thr)).alias('strong_neutral'),
                ((pl.col('sentiment_label_int') == 1) & (pl.col('sentiment_score') > strong_thr)).alias('strong_positive'),
                ((pl.col('sentiment_label_int') == -1) & (pl.col('sentiment_score') > strong_thr)).alias('strong_negative'),
                pl.col('entity_groups').list.unique(),
            ).group_by('user_id').agg(
                pl.col('article_id_fixed'),
                pl.col('impression_time_fixed'),
                pl.col('category'),
                pl.col('NumArticlesHistory').first(),
                pl.col('read_time_fixed').median().alias('MedianReadTime'),
                pl.col('read_time_fixed').max().alias('MaxReadTime'),
                pl.col('read_time_fixed').sum().alias('TotalReadTime'),
                pl.col('scroll_percentage_fixed').median().alias('MedianScrollPercentage'),
                pl.col('scroll_percentage_fixed').max().alias('MaxScrollPercentage'),
                (pl.col('is_neutral').sum() / pl.col('NumArticlesHistory').first()).alias('NeutralPct'),
                (pl.col('is_positive').sum() / pl.col('NumArticlesHistory').first()).alias('PositivePct'),
                (pl.col('is_negative').sum() / pl.col('NumArticlesHistory').first()).alias('NegativePct'),
                (pl.col('strong_neutral').sum() / pl.col('NumArticlesHistory').first()).alias('PctStrongNeutral'),
                (pl.col('strong_positive').sum() / pl.col('NumArticlesHistory').first()).alias('PctStrongPositive'),
                (pl.col('strong_negative').sum() / pl.col('NumArticlesHistory').first()).alias('PctStrongNegative'),
                (1 - (pl.col('is_article_default').sum() / pl.col('NumArticlesHistory').first())).alias('PctNotDefaultArticles'),
                pl.col('category').mode().alias('MostFrequentCategory'),
                pl.col('weekday').mode().alias('MostFrequentWeekday'),
                pl.col('hour').mode().alias('MostFrequentHour'),
                pl.col('entity_groups').flatten(),
                pl.col('topics').flatten().alias('topics_flatten')
            ).pipe(_add_topics_tf_idf_columns, topics_col='topics_flatten', vectorizer=vectorizer) \
            .drop('topics_flatten').with_columns(
                pl.col('MostFrequentCategory').list.first(),
                pl.col('MostFrequentWeekday').list.first(),
                pl.col('MostFrequentHour').list.first(),
            ).with_columns(
                (pl.col('entity_groups').list.count_matches(entity) / pl.col('NumArticlesHistory')).alias(f'{entity}Pct')
                for entity in unique_entities
            ).drop('entity_groups')
        for rows in tqdm(history.iter_slices(1000), total=history.shape[0] // 1000)
    )
    return reduce_polars_df_memory_size(history)


def _join_history(df_features: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame):
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
    
    prev_train_columns = [c for c in df_features.columns if c not in ['impression_id', 'article']]

    df_features = pl.concat(
        rows.join(history.select(['user_id', 'article_id_fixed']), on='user_id', how='left') \
            .join(articles.select(['article_id', 'topics', 'entity_groups', 'topics_idf']), left_on='article', right_on='article_id', how='left') \
            .explode(['article_id_fixed']) \
            .join(articles.select(['article_id', 'topics', 'entity_groups']), left_on='article_id_fixed', right_on='article_id', how='left') \
            .rename({'topics_right': 'topics_history', 'entity_groups_right': 'entity_groups_history'}) \
            .with_columns(
                (pl.col("topics").list.set_intersection(pl.col("topics_history")).list.len().truediv(
                    pl.col("topics").list.set_union(pl.col("topics_history")).list.len()
                )).alias("JS"),
                pl.col('entity_groups').list.set_intersection(pl.col('entity_groups_history')).list.len().alias('common_entities'),
            ).drop(['entity_groups_history', 'entity_groups', 'topics', 'topics_history']) \
            .group_by(['impression_id', 'article']).agg(
                pl.col(prev_train_columns).first(),
                pl.col('topics_idf').first(),
                pl.col('common_entities').mean().alias('MeanCommonEntities'),
                pl.col('common_entities').max().alias('MaxCommonEntities'),
                pl.col("JS").mean().alias("mean_JS"),
                pl.col("JS").min().alias("min_JS"),
                pl.col("JS").max().alias("max_JS"),
                pl.col("JS").std().alias("std_JS"),
            ).join(history.drop(['article_id_fixed', 'impression_time_fixed']), on='user_id', how='left') \
            .with_columns(
                pl.struct(['topics_idf', 'topics_flatten_tf_idf']).map_elements(
                    lambda x: cosine_similarity(x['topics_idf'], x['topics_flatten_tf_idf']), return_dtype=pl.Float64
                ).cast(pl.Float32).alias('topics_cosine'),
                (pl.col('category') == pl.col('MostFrequentCategory')).alias('IsFavouriteCategory'),
                pl.col('category_right').list.n_unique().alias('NumberDifferentCategories'),
                list_pct_matches_with_col('category_right', 'category').alias('PctCategoryMatches'),
            ).drop(['topics_idf', 'topics_flatten', 'topics_flatten_tf_idf', 'category_right'])
        for rows in tqdm(df_features.iter_slices(10000), total=df_features.shape[0] // 10000)
    )
    return reduce_polars_df_memory_size(df_features)


def add_trendiness_feature(df_features: pl.DataFrame ,articles: pl.DataFrame ,period:str="3d"):
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
        pl.DataFrame: The training dataset enriched with a new column, containing the trendiness_score. Can be 
    """
    
        
    topics=articles.select("topics").explode("topics").unique()
    topics=[topic for topic in topics["topics"] if topic is not None]
    #min_impression_time = df_features.select(pl.col("impression_time")).min().item()
    
    #topics_total_publications= articles.filter(pl.col("published_time")< min_impression_time ).select("topics") \
    #.explode("topics").group_by("topics").len()
    
    topics_popularity = articles.select(["published_time","topics"]).with_columns(
        pl.col("published_time").dt.date().alias("published_date")
    ).drop("published_time").group_by("published_date").agg(
        pl.col("topics").flatten()
    ).sort("published_date").set_sorted("published_date").upsample(time_column="published_date",every="1d") \
    .rolling(index_column="published_date",period=period).agg(
        [pl.col("topics").list.count_matches(topic).sum().alias(f"{topic}_matches") for topic in topics]
    )
    
    
    
    return df_features.with_columns(
        pl.col("impression_time").dt.date().alias("impression_date")
    ).join(other= articles.select(["article_id","topics"]),left_on="article",right_on="article_id",how="left" ) \
    .with_columns(
        [pl.col("topics").list.contains(topic).cast(pl.Int8).alias(f"{topic}_present") for topic in topics]
    ).join(other=topics_popularity,left_on=pl.col("impression_date"),right_on=(pl.col("published_date")+pl.duration(days=1)),how="left") \
    .with_columns(
        [pl.col(f"{topic}_present").mul(pl.col(f"{topic}_matches")).alias(f"trendiness_score_{topic}") for topic in topics]
    ).with_columns(
        pl.mean_horizontal( [pl.col(f"trendiness_score_{topic}") for topic in topics] ).alias("trendiness_score"),
    ).drop(
        [f"trendiness_score_{topic}" for topic in topics]
    ).drop(
        [f"{topic}_matches" for topic in topics]
    ).drop(
        [f"{topic}_present" for topic in topics]
    ).drop(["topics","impression_date"])
    
    
    

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

    df_features = df_features.join(published_category_popularity, how='left', right_on=['published_date', 'category'],
                                   left_on=[pl.col('impression_time').dt.date() - pl.duration(days=1), 'category']) \
        .rename({'category_daily_pct': 'yesterday_category_daily_pct'}).drop(['impression_time']) \
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
        pl.col('impression_time_fixed').list.max().alias('last_history_impression_time'),
        pl.col('article_id_fixed').list.tail(1).alias('last_history_article'),
    ).select(['user_id', 'last_history_impression_time', 'last_history_article'])

    last_session_time_df = behaviors.select(['session_id', 'user_id', 'impression_time', 'article_ids_inview', 'article_ids_clicked']) \
        .explode('article_ids_clicked') \
        .with_columns(pl.col('article_ids_clicked').cast(pl.Int32)) \
        .join(articles.select(['article_id', 'category']), left_on='article_ids_clicked', right_on='article_id', how='left') \
        .group_by('session_id').agg(
            pl.col('user_id').first(), 
            pl.col('impression_time').max().alias('session_time'), 
            pl.col('article_ids_inview').flatten().alias('all_seen_articles'),
            (pl.col('impression_time').max() - pl.col('impression_time').min()).dt.total_minutes().alias('session_duration'),
            pl.col('article_ids_clicked').count().alias('session_nclicks'),
            # pl.col('category').alias('all_categories'),
            pl.col('category').mode().alias('most_freq_category'),
        ).sort(['user_id', 'session_time']).with_columns(
            pl.col('most_freq_category').list.first(),
        ).with_columns(
            pl.col(['session_time', 'session_nclicks', 'session_duration', 'most_freq_category']) \
                .shift(1).over('user_id').name.prefix('last_'),
            pl.col('all_seen_articles').list.unique().shift(1).over('user_id'),
            pl.col('session_duration').rolling_mean(100, min_periods=1).over('user_id').alias('mean_prev_sessions_duration'),
        ).with_columns(pl.col(['last_session_nclicks', 'last_session_duration']).fill_null(0)) \
        .join(last_history_df, on='user_id', how='left') \
        .with_columns(
            pl.col('last_session_time').fill_null(pl.col('last_history_impression_time')),
            pl.col('all_seen_articles').fill_null(pl.col('last_history_article')),
        ).select(['session_id', 'last_session_time', 'last_session_nclicks', 'last_most_freq_category',
                'last_session_duration', 'all_seen_articles', 'mean_prev_sessions_duration'])
        
    gc.collect()
        
    df_features = df_features.join(last_session_time_df, on='session_id', how='left').with_columns(
        (pl.col('impression_time') - pl.col('last_session_time')).dt.total_hours().alias('last_session_time_hour_diff'),
        ((pl.col('last_session_time') - pl.col('published_time')).dt.total_hours() > 0).alias('is_new_article'),
        pl.col('all_seen_articles').list.contains(pl.col('article')).alias('is_already_seen_article'),
        (pl.col('category') == pl.col('last_most_freq_category')).fill_null(False).alias('is_last_session_most_seen_category'),
    ).drop(['published_time', 'session_id', 'all_seen_articles', 'last_session_time', 'last_most_freq_category'])
    return reduce_polars_df_memory_size(df_features)


def add_mean_delays_features(df_features:pl.DataFrame,articles:pl.DataFrame,history:pl.DataFrame)->pl.DataFrame:
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
    topic_mean_delays = history.select(["impression_time_fixed","article_id_fixed"]).explode(["impression_time_fixed","article_id_fixed"]) \
    .join(other=articles.select(["article_id","topics","published_time"]),left_on="article_id_fixed",right_on="article_id",how="left") \
    .drop("article_id_fixed").with_columns(
        (pl.col('impression_time_fixed') - pl.col('published_time')).dt.total_days().alias('article_delay_days'),
        (pl.col('impression_time_fixed') - pl.col('published_time')).dt.total_hours().alias('article_delay_hours')
    ).explode("topics").group_by("topics").agg(
        pl.col("article_delay_days").mean().alias("topic_mean_delay_days"),
        pl.col("article_delay_hours").mean().alias("topic_mean_delay_hours")
    )
    
    user_mean_delays = history.select(["user_id","impression_time_fixed","article_id_fixed"]).explode(["impression_time_fixed","article_id_fixed"]) \
    .join(other=articles.select(["article_id","published_time"]),left_on="article_id_fixed",right_on="article_id",how="left") \
    .drop("article_id_fixed").with_columns(
        (pl.col('impression_time_fixed') - pl.col('published_time')).dt.total_days().alias('article_delay_days'),
        (pl.col('impression_time_fixed') - pl.col('published_time')).dt.total_hours().alias('article_delay_hours')
    ).group_by("user_id").agg(
        pl.col("article_delay_days").mean().alias("user_mean_delay_days"),
        pl.col("article_delay_hours").mean().alias("user_mean_delay_hours")
    )
    
    return df_features.join(other=articles.select(["article_id","topics"]),left_on="article",right_on="article_id",how="left").explode("topics") \
    .join(other=topic_mean_delays, on="topics",how="left").group_by(["impression_id","article","user_id"]).agg(
        pl.exclude(["topic_mean_delay_days","topic_mean_delay_hours","topics"]).first(),
        pl.col("topic_mean_delay_days").mean(),
        pl.col("topic_mean_delay_hours").mean()
    ).rename({"topic_mean_delay_days":"mean_topics_mean_delay_days","topic_mean_delay_hours":"mean_topics_mean_delay_hours"}) \
    .join(other=user_mean_delays,on="user_id",how="left")


def add_history_trendiness_scores_feature(df_features:pl.DataFrame,history:pl.DataFrame,articles:pl.DataFrame)-> pl.DataFrame:
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
    
    topics=articles.select("topics").explode("topics").unique()
    topics=[topic for topic in topics["topics"] if topic is not None]
    
    history_trendiness_scores= history.select(["user_id","impression_time_fixed","article_id_fixed"]).explode(["impression_time_fixed","article_id_fixed"]) \
    .rename({"impression_time_fixed":"impression_time","article_id_fixed":"article"}).pipe(
        add_trendiness_feature,articles
    )
    
    users_mean_trendiness_scores = history_trendiness_scores.select(["user_id","trendiness_score"]).group_by("user_id").agg(
        pl.col("trendiness_score").mean().alias("mean_user_trendiness_score")
    )
    
    topics_mean_trendiness_scores= history_trendiness_scores.select("article","trendiness_score") \
    .join(other=articles.select(["article_id","topics"]),left_on="article",right_on="article_id",how="left") \
    .explode("topics").group_by("topics").agg(
        pl.col("trendiness_score").mean().alias("mean_topic_trendiness_score")
    )
    
    return df_features.join(other=users_mean_trendiness_scores, on="user_id",how="left") \
    .join(other=articles.select(["article_id","topics"]),left_on="article",right_on="article_id",how="left") \
    .with_columns(
        [pl.col("topics").list.contains(topic).cast(pl.Int8).alias(f"{topic}_present") for topic in topics]
    ).with_columns(
        [pl.col(f"{topic}_present").mul(topics_mean_trendiness_scores.filter(pl.col("topics")==topic).select("mean_topic_trendiness_score")) \
         .alias(f"mean_topic_{topic}_trendiness_score") for topic in topics]
    ).with_columns(
        pl.sum_horizontal( [pl.col(f"mean_topic_{topic}_trendiness_score") for topic in topics] ).truediv(pl.col("topics").list.len())
        .alias("mean_topics_trendiness_score")
    ).drop(
        [f"{topic}_present" for topic in topics]
    ).drop(
        [f"mean_topic_{topic}_trendiness_score" for topic in topics]
    ).drop("topics")