import os
import polars as pl
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from polimi.utils._catboost import _preprocessing
from polimi.utils._catboost import _build_features_behaviors
from polimi.utils._catboost import _preprocessing_history_trendiness_scores
from polimi.utils._catboost import add_history_trendiness_scores_feature
from polimi.utils._catboost import _preprocessing_mean_delay_features
from polimi.utils._catboost import add_mean_delays_features
from polimi.utils._catboost import _preprocessing_window_features
from polimi.utils._catboost import add_window_features
from polimi.utils._catboost import add_trendiness_feature_categories
from polimi.utils._catboost import _preprocessing_article_endorsement_feature
from polimi.utils._catboost import add_article_endorsement_feature
from polimi.utils._topic_model import _compute_topic_model, add_topic_model_features
from polimi.utils._polars import reduce_polars_df_memory_size
from polimi.utils._catboost import get_unique_categories

'''
New features:
    - trendiness scores for history
    - mean read delay
    - topic model related features
'''
CATEGORICAL_COLUMNS = ['device_type', 'is_sso_user', 'gender', 'is_subscriber', 'weekday',
                       'premium', 'category', 'sentiment_label', 'is_new_article', 'is_already_seen_article',
                       'MostFrequentCategory', 'MostFrequentWeekday', 'IsFavouriteCategory']


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

    print('Preprocessing article endorsement feature...')
    articles_endorsement = _preprocessing_article_endorsement_feature(
        behaviors=behaviors, period="10h")

    unique_categories = get_unique_categories(articles)

    print('Reading old features...')
    if previous_version is not None:
        behaviors = pl.read_parquet(previous_version)

    print('Building features...')
    for sliced_df in behaviors.iter_slices(behaviors.shape[0] // n_batches):
        if previous_version is None:
            slice_features = sliced_df.pipe(_build_features_behaviors, history=history, articles=articles,
                                            cols_explode=cols_explode, rename_columns=rename_cols, unique_entities=unique_entities,
                                            unique_categories=unique_categories)
        else:
            slice_features = sliced_df

        slice_features = _build_new_features(df_features=slice_features, history=history, articles=articles, users_mean_trendiness_scores=users_mean_trendiness_scores,
                                             topics_mean_trendiness_scores=topics_mean_trendiness_scores, topics=topics, topic_mean_delays=topic_mean_delays,
                                             user_mean_delays=user_mean_delays, windows=windows, user_windows=user_windows,
                                             user_category_windows=user_category_windows, user_topics_windows=user_topics_windows, articles_endorsement=articles_endorsement,
                                             topic_model_columns=topic_model_columns, n_components=n_components)

        yield slice_features, vectorizer, unique_entities


def build_features_iterator_test(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                                 test: bool = False, sample: bool = True, npratio: int = 2,
                                 tf_idf_vectorizer: TfidfVectorizer = None, n_batches: int = 10, previous_version=None,
                                 **kwargs):
    
    behaviors, history, articles, vectorizer, unique_entities, cols_explode, rename_cols = _preprocessing(
        behaviors, history, articles, test, sample, npratio
    )

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

    print('Preprocessing article endorsement feature...')
    articles_endorsement = _preprocessing_article_endorsement_feature(
        behaviors=behaviors.filter(pl.col('impression_time')!= 0), period="10h")

    print('Building features...')
    for slice in range(0,101):
        slice_features = pl.read_parquet(os.path.join(previous_version, f'Sliced_ds/test_slice_{slice}.parquet'))

        slice_features = _build_new_features(df_features=slice_features, history=history, articles=articles, users_mean_trendiness_scores=users_mean_trendiness_scores,
                                             topics_mean_trendiness_scores=topics_mean_trendiness_scores, topics=topics, topic_mean_delays=topic_mean_delays,
                                             user_mean_delays=user_mean_delays, windows=windows, user_windows=user_windows,
                                             user_category_windows=user_category_windows, user_topics_windows=user_topics_windows, articles_endorsement=articles_endorsement,
                                             topic_model_columns=topic_model_columns, n_components=n_components)

        yield slice_features, vectorizer, unique_entities


def build_features(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                   test: bool = False, sample: bool = True, npratio: int = 2,
                   tf_idf_vectorizer: TfidfVectorizer = None, previous_version=None, **kwargs) -> pl.DataFrame:
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

    articles_endorsement = _preprocessing_article_endorsement_feature(
        behaviors=behaviors, period="10h")

    unique_categories = get_unique_categories(articles)

    df_features = behaviors.pipe(_build_features_behaviors, history=history, articles=articles,
                                 cols_explode=cols_explode, rename_columns=rename_cols, unique_entities=unique_entities,
                                 unique_categories=unique_categories)

    df_features = _build_new_features(df_features=df_features, history=history, articles=articles, users_mean_trendiness_scores=users_mean_trendiness_scores,
                                      topics_mean_trendiness_scores=topics_mean_trendiness_scores, topics=topics, topic_mean_delays=topic_mean_delays,
                                      user_mean_delays=user_mean_delays, windows=windows, user_windows=user_windows,
                                      user_category_windows=user_category_windows, user_topics_windows=user_topics_windows, articles_endorsement=articles_endorsement,
                                      topic_model_columns=topic_model_columns, n_components=n_components)

    return df_features, vectorizer, unique_entities


def _build_new_features(df_features: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame, users_mean_trendiness_scores: pl.DataFrame,
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
