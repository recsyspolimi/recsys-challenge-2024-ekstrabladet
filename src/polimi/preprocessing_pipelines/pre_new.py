import os
import polars as pl
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from polimi.utils._catboost import (
    _preprocessing,
    _build_features_behaviors,
    _preprocessing_history_trendiness_scores,
    add_history_trendiness_scores_feature,
    _preprocessing_mean_delay_features,
    add_mean_delays_features,
    _preprocessing_window_features,
    add_window_features,
    add_trendiness_feature_categories,
    _preprocessing_article_endorsement_feature,
    add_article_endorsement_feature,
    get_unique_categories,
    _preprocessing_normalize_endorsement,
    add_trendiness_feature,
    _preprocessing_article_endorsement_feature_by_article_and_user,
    _preprocessing_normalize_endorsement_by_article_and_user,
    add_article_endorsement_feature_by_article_and_user,
    _preprocessing_leak_counts,
    add_article_endorsement_feature_leak,
    add_trendiness_feature_leak

)
from polimi.utils._topic_model import _compute_topic_model, add_topic_model_features
from polimi.utils._polars import reduce_polars_df_memory_size, inflate_polars_df
from polimi.utils._norm_and_stats import *
'''
New features:
    - endorsement normalizations
    - many trendiness periods + normalizations
    - feature normalizations over the impression/artice/user
    - statistics over the impression
'''
CATEGORICAL_COLUMNS = ['device_type', 'is_sso_user', 'gender', 'is_subscriber', 'weekday',
                       'premium', 'category', 'sentiment_label', 'is_new_article', 'is_already_seen_article',
                       'MostFrequentCategory', 'MostFrequentWeekday', 'IsFavouriteCategory', 'article_type', 'postcode']



NORMALIZE_OVER_IMPRESSION_ID = [
    'trendiness_score_3d','trendiness_score_5d', 'endorsement_10h', 
    'total_pageviews/inviews', 'mean_JS','mean_topic_model_cosine', 'topics_cosine',
    'article_delay_hours', 'total_pageviews','total_inviews', 'trendiness_score_category',
    'std_JS', 'total_read_time', 'endorsement_10h_leak',"trendiness_score_3d_leak",
    "clicked_count","inview_count",
]
RANK_IMPRESSION_DESCENDING = [
    'trendiness_score_3d','trendiness_score_5d', 'endorsement_10h', 
    'total_pageviews/inviews', 'mean_JS', 'mean_topic_model_cosine', 'topics_cosine',
    'total_pageviews', 'total_read_time','total_inviews', 'trendiness_score_category',
    'endorsement_10h_leak',"trendiness_score_3d_leak","clicked_count","inview_count"
]
RANK_IMPRESSION_ASCENDING = [
    'article_delay_hours', 'std_JS', 'mean_topics_mean_delay_hours'
]
NORMALIZE_OVER_USER_ID = [
    'mean_JS', 'std_JS', 'mean_topic_model_cosine', 'topics_cosine',
]
NORMALIZE_OVER_ARTICLE = [
    'article_delay_hours', 'mean_JS', 'std_JS', 'mean_topic_model_cosine', 'topics_cosine',
]
LIST_DIVERSITY = [
    'category', 'sentiment_label', 'article_type'
]
NORMALIZE_OVER_ARTICLE_AND_USER_ID= ['endorsement_20h_articleuser']



def build_features_iterator(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                            test: bool = False, sample: bool = True, npratio: int = 2,
                            tf_idf_vectorizer: TfidfVectorizer = None, n_batches: int = 10, previous_version=None, 
                            **kwargs):
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
    old_behaviors = behaviors
    articles = articles.with_columns((pl.col('total_pageviews') / pl.col('total_inviews')).alias('total_pageviews/inviews'))
    
    print('Preprocessing article endorsement feature...')
    articles_endorsement = _preprocessing_article_endorsement_feature(
        behaviors=behaviors, period="10h")
    
    print('Preprocessing article endorsement by article and user feature...')
    articles_endorsement_articleuser = _preprocessing_article_endorsement_feature_by_article_and_user(
    behaviors=behaviors, period="20h")

    print('Preprocessing leak count features')
    history_counts,behaviors_counts = _preprocessing_leak_counts(history=history,behaviors=behaviors)
    
    if previous_version is None:
        print('Computing topic model...')
        articles, topic_model_columns, n_components = _compute_topic_model(
            articles)

        topics = articles.select("topics").explode("topics").unique()
        topics = [topic for topic in topics["topics"] if topic is not None]

        print('Preprocessing history trendiness scores...')
        users_mean_trendiness_scores, topics_mean_trendiness_scores = _preprocessing_history_trendiness_scores(
            history=history, articles=articles)

        print('Preprocessing mean delays...')
        topic_mean_delays, user_mean_delays = _preprocessing_mean_delay_features(
            articles=articles, history=history)

        print('Preprocessing window features...')
        windows, user_windows, user_topics_windows, user_category_windows = _preprocessing_window_features(
            history=history, articles=articles)

        unique_categories = get_unique_categories(articles)

    print('Reading old features...')
    if previous_version is not None:
        behaviors = pl.read_parquet(previous_version)
        
    articles_endorsement_norm = _preprocessing_normalize_endorsement(articles_endorsement, 'endorsement_10h')
    articles_endorsement_norm = articles_endorsement_norm.drop('endorsment_10h')

    articles_endorsement_articleuser_norm = _preprocessing_normalize_endorsement_by_article_and_user(articles_endorsement_articleuser,'endorsement_20h_articleuser')

    print('Building features...')
    df_features = None
    i = 0
    for sliced_df in behaviors.iter_slices(behaviors.shape[0] // n_batches):
        print(f'Preprocessing slice {i}')
        i += 1
        if previous_version is None:
            slice_features = sliced_df.pipe(_build_features_behaviors, history=history, articles=articles,
                                            cols_explode=cols_explode, rename_columns=rename_cols, unique_entities=unique_entities,
                                            unique_categories=unique_categories)
            slice_features = _build_v127_features(df_features=slice_features, history=history, articles=articles, users_mean_trendiness_scores=users_mean_trendiness_scores,
                                                  topics_mean_trendiness_scores=topics_mean_trendiness_scores, topics=topics, topic_mean_delays=topic_mean_delays,
                                                  user_mean_delays=user_mean_delays, windows=windows, user_windows=user_windows,
                                                  user_category_windows=user_category_windows, user_topics_windows=user_topics_windows, articles_endorsement=articles_endorsement,
                                                  topic_model_columns=topic_model_columns, n_components=n_components)
        else:
            slice_features = sliced_df
            
        slice_features = _build_new_features(slice_features, old_behaviors, articles, articles_endorsement_norm,articles_endorsement_articleuser_norm,history_counts,behaviors_counts)
        if df_features is None:
            df_features = inflate_polars_df(slice_features)
        else:
            df_features = df_features.vstack(inflate_polars_df(slice_features))

    df_features = _build_normalizations(df_features)
    return df_features, vectorizer, unique_entities


def build_features_iterator_test(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                                 test: bool = False, sample: bool = True, npratio: int = 2,
                                 tf_idf_vectorizer: TfidfVectorizer = None, n_batches: int = 100, previous_version=None,
                                 **kwargs):
    
    behaviors, history, articles, vectorizer, unique_entities, cols_explode, rename_cols = _preprocessing(
        behaviors, history, articles, test, sample, npratio
    )
    old_behaviors = behaviors
    articles = articles.with_columns((pl.col('total_pageviews') / pl.col('total_inviews')).alias('total_pageviews/inviews'))
    
    print('Preprocessing article endorsement feature...')
    articles_endorsement = _preprocessing_article_endorsement_feature(
        behaviors=behaviors.filter(pl.col('impression_time')!= 0), period="10h")
    
    print('Preprocessing article endorsement by article and user feature...')
    articles_endorsement_articleuser = _preprocessing_article_endorsement_feature_by_article_and_user(
    behaviors=behaviors.filter(pl.col('impression_time')!= 0), period="20h")

    print('Preprocessing leak count features')
    history_counts,behaviors_counts = _preprocessing_leak_counts(history=history,behaviors=behaviors.filter(pl.col('impression_time')!= 0))

    if previous_version is None:
        print('Computing topic model...')
        articles, topic_model_columns, n_components = _compute_topic_model(
            articles)

        topics = articles.select("topics").explode("topics").unique()
        topics = [topic for topic in topics["topics"] if topic is not None]

        print('Preprocessing history trendiness scores...')
        users_mean_trendiness_scores, topics_mean_trendiness_scores = _preprocessing_history_trendiness_scores(
            history=history, articles=articles)

        print('Preprocessing mean delays...')
        topic_mean_delays, user_mean_delays = _preprocessing_mean_delay_features(
            articles=articles, history=history)

        print('Preprocessing window features...')
        windows, user_windows, user_topics_windows, user_category_windows = _preprocessing_window_features(
            history=history, articles=articles)
        
        unique_categories = get_unique_categories(articles)

    print('Building features...')
    articles_endorsement_norm = _preprocessing_normalize_endorsement(articles_endorsement, 'endorsement_10h')
    articles_endorsement_norm = articles_endorsement_norm.drop('endorsment_10h')

    articles_endorsement_articleuser_norm = _preprocessing_normalize_endorsement_by_article_and_user(articles_endorsement_articleuser,'endorsement_20h_articleuser')

    df_features = None
    iterator = range(0,101) if previous_version is not None else behaviors.iter_slices(behaviors.shape[0] // n_batches)
    i = 0
    for slice in iterator:
        print(f'Preprocessing slice {i}')
        i += 1
        if previous_version is not None:
            slice_features = pl.read_parquet(os.path.join(previous_version, f'Sliced_ds/test_slice_{slice}.parquet'))
        else:
            slice_features = slice.pipe(_build_features_behaviors, history=history, articles=articles,
                                        cols_explode=cols_explode, rename_columns=rename_cols, unique_entities=unique_entities,
                                        unique_categories=unique_categories)
            
            slice_features = _build_v127_features(df_features=slice_features, history=history, articles=articles, users_mean_trendiness_scores=users_mean_trendiness_scores,
                                                  topics_mean_trendiness_scores=topics_mean_trendiness_scores, topics=topics, topic_mean_delays=topic_mean_delays,
                                                  user_mean_delays=user_mean_delays, windows=windows, user_windows=user_windows,
                                                  user_category_windows=user_category_windows, user_topics_windows=user_topics_windows, articles_endorsement=articles_endorsement,
                                                  topic_model_columns=topic_model_columns, n_components=n_components)

        slice_features = _build_new_features(slice_features, old_behaviors, articles, articles_endorsement_norm,articles_endorsement_articleuser_norm,history_counts,behaviors_counts)
        if df_features is None:
            df_features = inflate_polars_df(slice_features)
        else:
            df_features = df_features.vstack(inflate_polars_df(slice_features))

    df_features = _build_normalizations(df_features)
    return df_features, vectorizer, unique_entities


def build_features(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                   test: bool = False, sample: bool = True, npratio: int = 2,
                   tf_idf_vectorizer: TfidfVectorizer = None, previous_version=None, **kwargs) -> pl.DataFrame:

    behaviors, history, articles, vectorizer, unique_entities, cols_explode, rename_cols = _preprocessing(
        behaviors, history, articles, test, sample, npratio
    )
    articles = articles.with_columns((pl.col('total_pageviews') / pl.col('total_inviews')).alias('total_pageviews/inviews'))
    
    articles_endorsement = _preprocessing_article_endorsement_feature(
        behaviors=behaviors, period="10h")
    
    articles_endorsement_articleuser = _preprocessing_article_endorsement_feature_by_article_and_user(
    behaviors=behaviors, period="20h")

    print('Preprocessing leak count features')
    history_counts,behaviors_counts = _preprocessing_leak_counts(history=history,behaviors=behaviors)

    if previous_version is None:
        articles, topic_model_columns, n_components = _compute_topic_model(
            articles)

        topics = articles.select("topics").explode("topics").unique()
        topics = [topic for topic in topics["topics"] if topic is not None]

        users_mean_trendiness_scores, topics_mean_trendiness_scores = _preprocessing_history_trendiness_scores(
            history=history, articles=articles)

        topic_mean_delays, user_mean_delays = _preprocessing_mean_delay_features(
            articles=articles, history=history)

        windows, user_windows, user_topics_windows, user_category_windows = _preprocessing_window_features(
            history=history, articles=articles)

        unique_categories = get_unique_categories(articles)

        df_features = behaviors.pipe(_build_features_behaviors, history=history, articles=articles,
                                    cols_explode=cols_explode, rename_columns=rename_cols, unique_entities=unique_entities,
                                    unique_categories=unique_categories)

        df_features = _build_v127_features(df_features=df_features, history=history, articles=articles, users_mean_trendiness_scores=users_mean_trendiness_scores,
                                           topics_mean_trendiness_scores=topics_mean_trendiness_scores, topics=topics, topic_mean_delays=topic_mean_delays,
                                           user_mean_delays=user_mean_delays, windows=windows, user_windows=user_windows,
                                           user_category_windows=user_category_windows, user_topics_windows=user_topics_windows, articles_endorsement=articles_endorsement,
                                           topic_model_columns=topic_model_columns, n_components=n_components)
    else:
        df_features = pl.read_parquet(previous_version)
    
    articles_endorsement = _preprocessing_normalize_endorsement(articles_endorsement, 'endorsement_10h')
    # dropping column endorsment_10h since it is already in the previous version
    articles_endorsement = articles_endorsement.drop('endorsment_10h')

    articles_endorsement_articleuser_norm = _preprocessing_normalize_endorsement_by_article_and_user(articles_endorsement_articleuser,'endorsement_20h_articleuser')


    
    df_features = _build_new_features(df_features, articles, articles_endorsement,articles_endorsement_articleuser_norm,history_counts,behaviors_counts)
    df_features = _build_normalizations(df_features)
    return reduce_polars_df_memory_size(df_features), vectorizer, unique_entities


def _build_v127_features(df_features: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame, users_mean_trendiness_scores: pl.DataFrame,
                        topics_mean_trendiness_scores: pl.DataFrame, topics: pl.DataFrame, topic_mean_delays: pl.DataFrame,
                        user_mean_delays: pl.DataFrame, windows: pl.DataFrame, user_windows: pl.DataFrame,
                        user_category_windows: pl.DataFrame, user_topics_windows: pl.DataFrame, articles_endorsement: pl.DataFrame,
                        topic_model_columns: list[str], n_components: int) -> pl.DataFrame:
    df_features = pl.concat(
        rows.pipe(add_history_trendiness_scores_feature, articles=articles, users_mean_trendiness_scores=users_mean_trendiness_scores,
                  topics_mean_trendiness_scores=topics_mean_trendiness_scores, topics=topics)
        .pipe(add_mean_delays_features, articles=articles,
              topic_mean_delays=topic_mean_delays, user_mean_delays=user_mean_delays)
        .pipe(add_window_features, articles=articles, user_windows=user_windows,
              user_category_windows=user_category_windows, user_topics_windows=user_topics_windows, windows=windows)
        .pipe(add_trendiness_feature_categories, articles=articles)
        .pipe(add_article_endorsement_feature, articles_endorsement=articles_endorsement)
        for rows in tqdm(df_features.iter_slices(20000), total=df_features.shape[0] // 20000))

    df_features = reduce_polars_df_memory_size(df_features)

    df_features = df_features.pipe(add_topic_model_features, history=history,
                                   articles=articles, topic_model_columns=topic_model_columns, n_components=n_components)

    return reduce_polars_df_memory_size(df_features)


def _build_new_features(df_features: pl.DataFrame, behaviors: pl.DataFrame, articles: pl.DataFrame, articles_endorsement: pl.DataFrame,
                        articles_endorsement_articleuser:pl.DataFrame,history_counts:pl.DataFrame, behaviors_counts: pl.DataFrame):
    '''
    articles_endorsement must be without the col endorsment_10, only the normalizations are added in this version
    trendiness_score must be already present in the features dataframe (it will be renamed to trendiness_score_3d)
    '''
    df_features = pl.concat(
        rows.pipe(add_article_endorsement_feature, articles_endorsement=articles_endorsement)
            .pipe(add_article_endorsement_feature_by_article_and_user, articles_endorsement_articleuser = articles_endorsement_articleuser)
            .pipe(add_article_endorsement_feature_leak, articles_endorsement = articles_endorsement)
            .rename({
                "normalized_endorsement_10h_right":"normalized_endorsement_10h_leak",
                "endorsement_10h_diff_rolling_right":"endorsement_10h_leak_diff_rolling",
                "endorsement_10h_macd_right":"endorsement_10h_leak_macd",
                "endorsement_10h_quantile_norm_right":"endorsement_10h_leak_quantile_norm",
                "normalized_endorsement_10h_rolling_max_ratio_right":"normalized_endorsement_10h_leak_rolling_max_ratio"
            })
            .rename({'trendiness_score': 'trendiness_score_3d'})
            .pipe(add_trendiness_feature, articles=articles, period='1d')
            .rename({'trendiness_score': 'trendiness_score_1d'})
            .pipe(add_trendiness_feature, articles=articles, period='5d')
            .rename({'trendiness_score': 'trendiness_score_5d'})
            .pipe(add_trendiness_feature_leak, articles=articles, period='3d')
            .rename({'trendiness_score_leak':'trendiness_score_3d_leak'})
            .with_columns(
                (
                    pl.col('trendiness_score_1d') / 
                    pl.col('trendiness_score_3d')
                ).alias('trendiness_score_1d/3d'),
                (
                    pl.col('trendiness_score_1d') / 
                    pl.col('trendiness_score_5d')
                ).alias('trendiness_score_1d/5d'),
                (
                    pl.col('trendiness_score_3d') / 
                    pl.col('trendiness_score_3d').max().over(pl.col('impression_time').dt.date())
                ).alias('normalized_trendiness_score_overall'),
            ).join(behaviors.select(['impression_id', 'user_id', 'postcode']), on=['impression_id', 'user_id'], how='left')
            .join(articles.select(['article_id', 'total_pageviews', 'total_inviews', 'total_read_time', 
                                   'total_pageviews/inviews', 'article_type']).with_columns(pl.col('article_id').cast(pl.Int32)),
                  left_on='article', right_on='article_id', how='left')
            .join(history_counts, on='article',how='left')
            .join(behaviors_counts , on='article', how='left')
        for rows in tqdm(df_features.iter_slices(20000), total=df_features.shape[0] // 20000))

    return reduce_polars_df_memory_size(df_features)


def _build_normalizations(df_features: pl.DataFrame):
    expressions = sum([
        get_norm_expression(NORMALIZE_OVER_IMPRESSION_ID, over='impression_id', norm_type='infinity', suffix_name='_impression'),
        get_list_rank_expression(RANK_IMPRESSION_ASCENDING, over='impression_id', suffix_name='_impression', descending=False),
        get_list_rank_expression(RANK_IMPRESSION_DESCENDING, over='impression_id', suffix_name='_impression', descending=True),
        get_norm_expression(NORMALIZE_OVER_USER_ID, over='user_id', norm_type='infinity', suffix_name='_user'),
        get_norm_expression(NORMALIZE_OVER_ARTICLE, over='article', norm_type='infinity', suffix_name='_article'),
        get_group_stats_expression(NORMALIZE_OVER_IMPRESSION_ID, over='impression_id', stat_type='std', suffix_name='_impression'),
        get_group_stats_expression(NORMALIZE_OVER_IMPRESSION_ID, over='impression_id', stat_type='skew', suffix_name='_impression'),
        get_group_stats_expression(NORMALIZE_OVER_IMPRESSION_ID, over='impression_id', stat_type='kurtosis', suffix_name='_impression'),
        get_group_stats_expression(NORMALIZE_OVER_IMPRESSION_ID, over='impression_id', stat_type='entropy', suffix_name='_impression'),
        get_diff_norm_expression(NORMALIZE_OVER_IMPRESSION_ID, over='impression_id', diff_type='median', suffix_name='_impression'),
        get_list_diversity_expression(LIST_DIVERSITY, over='impression_id', suffix_name='_impression'),
        get_norm_expression(NORMALIZE_OVER_ARTICLE_AND_USER_ID, over=['article','user_id'], norm_type='infinity', suffix_name='_articleuser'),
    ], [])
    return reduce_polars_df_memory_size(df_features.with_columns(expressions))