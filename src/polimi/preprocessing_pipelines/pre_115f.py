import polars as pl
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from polimi.utils._catboost import _preprocessing
from polimi.utils._catboost import _build_features_behaviors
from polimi.utils._catboost import _preprocessing_history_trendiness_scores
from polimi.utils._catboost import add_history_trendiness_scores_feature
from polimi.utils._catboost import _preprocessing_mean_delay_features
from polimi.utils._catboost import add_mean_delays_features
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
                            tf_idf_vectorizer: TfidfVectorizer = None, n_batches: int = 10, previous_version = None):
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
    print('Building Topic Model Representation')
    articles, topic_model_columns, n_components = _compute_topic_model(articles)
    print('Topic Model Representation done')

    print("Building topics list")
    topics = articles.select("topics").explode("topics").unique()
    topics = [topic for topic in topics["topics"] if topic is not None]
    print("Topics list built")

    print("History trendiness scores preprocessing")
    users_mean_trendiness_scores, topics_mean_trendiness_scores = _preprocessing_history_trendiness_scores(history=history, articles=articles)
    print("History trendiness scores preprocessing done")

    print("Mean delays preprocessing")
    topic_mean_delays, user_mean_delays = _preprocessing_mean_delay_features(articles=articles,history=history)
    print("Mean delays preprocessing done")
    
    unique_categories = get_unique_categories(articles)
    
    for sliced_df in behaviors.iter_slices(behaviors.shape[0] // n_batches):
        slice_features = sliced_df.pipe(_build_features_behaviors, history=history, articles=articles,
                                        cols_explode=cols_explode, rename_columns=rename_cols, unique_entities=unique_entities,
                                        unique_categories=unique_categories)
        
        print('Adding history trendiness features')
        slice_features = pl.concat( 
                        rows.pipe(add_history_trendiness_scores_feature, articles=articles, users_mean_trendiness_scores= users_mean_trendiness_scores, topics_mean_trendiness_scores= topics_mean_trendiness_scores, topics = topics)
                        for rows in tqdm(slice_features.iter_slices(20000), total=slice_features.shape[0] // 20000))
        slice_features = reduce_polars_df_memory_size(slice_features)
        
        print('Adding mean delays features')
        slice_features = pl.concat( 
                        rows.pipe(add_mean_delays_features, articles=articles, topic_mean_delays=topic_mean_delays, user_mean_delays=user_mean_delays)
                        for rows in tqdm(slice_features.iter_slices(20000), total=slice_features.shape[0] // 20000))
        slice_features = reduce_polars_df_memory_size(slice_features)
        
        print('Adding topic model features')
        slice_features = slice_features.pipe(add_topic_model_features, history=history, articles=articles,topic_model_columns=topic_model_columns,n_components=n_components)
        slice_features = reduce_polars_df_memory_size(slice_features)
        
        yield slice_features, vectorizer, unique_entities


def build_features(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                   test: bool = False, sample: bool = True, npratio: int = 2,
                   tf_idf_vectorizer: TfidfVectorizer = None, previous_version = None) -> pl.DataFrame:
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

    print('Building Topic Model Representation')
    articles, topic_model_columns, n_components = _compute_topic_model(articles)
    print('Topic Model Representation done')

    print("Building topics list")
    topics = articles.select("topics").explode("topics").unique()
    topics = [topic for topic in topics["topics"] if topic is not None]
    print("Topics list built")

    print("History trendiness scores preprocessing")
    users_mean_trendiness_scores, topics_mean_trendiness_scores = _preprocessing_history_trendiness_scores(history=history, articles=articles)
    print("History trendiness scores preprocessing done")

    print("Mean delays preprocessing")
    topic_mean_delays, user_mean_delays = _preprocessing_mean_delay_features(articles=articles,history=history)
    print("Mean delays preprocessing done")
    
    unique_categories = get_unique_categories(articles)
    
    behaviors, history, articles, vectorizer, unique_entities, cols_explode, rename_cols = _preprocessing(
        behaviors, history, articles, test, sample, npratio
    )
    
    articles, topic_model_columns, n_components = _compute_topic_model(articles)
    
    df_features = behaviors.pipe(_build_features_behaviors, history=history, articles=articles,
                                 cols_explode=cols_explode, rename_columns=rename_cols, unique_entities=unique_entities,
                                 unique_categories=unique_categories)
    
    df_features = pl.concat( 
                        rows.pipe(add_history_trendiness_scores_feature, articles=articles, users_mean_trendiness_scores= users_mean_trendiness_scores, topics_mean_trendiness_scores= topics_mean_trendiness_scores, topics = topics)
                        for rows in tqdm(df_features.iter_slices(20000), total=df_features.shape[0] // 20000))
    df_features = reduce_polars_df_memory_size(df_features)
        
    print('Adding mean delays features')
    df_features = pl.concat( 
                        rows.pipe(add_mean_delays_features, articles=articles, topic_mean_delays=topic_mean_delays, user_mean_delays=user_mean_delays)
                        for rows in tqdm(df_features.iter_slices(20000), total=df_features.shape[0] // 20000))
    df_features = reduce_polars_df_memory_size(df_features)
    
    df_features = df_features.pipe(add_topic_model_features, history=history, articles=articles,topic_model_columns=topic_model_columns,n_components=n_components)
    df_features = reduce_polars_df_memory_size(df_features)
    
    return df_features, vectorizer, unique_entities
