import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from polimi.utils._catboost import _preprocessing
from polimi.utils._catboost import _build_features_behaviors
from polimi.utils._catboost import get_unique_categories


'''
Baseline preprocessing

!! ATTENTION !!
This pipeline version calls the version that implements 94 features and drops the unrequired columns.
There's no reason to use this implementation since it has less features but requires the same computational time of
the version that implements 94 features.
'''

NEW_VERSION_FEATURES = ['PctCategoryMatches', 'Category_underholdning_Pct', 'Category_bibliotek_Pct', 'Category_migration_catalog_Pct', 'Category_ferie_Pct', 'Category_krimi_Pct',
                        'Category_side9_Pct', 'Category_tilavis_Pct', 'Category_penge_Pct', 'Category_abonnement_Pct', 'Category_dagsorden_Pct', 'Category_plus_Pct', 'Category_musik_Pct', 
                        'Category_podcast_Pct', 'Category_webmaster-test-sektion_Pct', 'Category_rssfeed_Pct','Category_auto_Pct', 'Category_horoskoper_Pct', 'Category_haandvaerkeren_Pct', 
                        'Category_forbrug_Pct', 'Category_vin_Pct', 'Category_services_Pct', 'Category_opinionen_Pct', 'Category_nyheder_Pct', 'Category_biler_Pct', 'Category_incoming_Pct',
                        'Category_sport_Pct', 'Category_om_ekstra_bladet_Pct', 'Category_sex_og_samliv_Pct', 'Category_video_Pct', 'Category_nationen_Pct', 'Category_webtv_Pct', 'Category_eblive_Pct']


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

    unique_categories = get_unique_categories(articles)

    for sliced_df in behaviors.iter_slices(behaviors.shape[0] // n_batches):
        slice_features = sliced_df.pipe(_build_features_behaviors, history=history, articles=articles,
                                        cols_explode=cols_explode, rename_columns=rename_cols,
                                        unique_entities=unique_entities, unique_categories=unique_categories)\
            .drop(NEW_VERSION_FEATURES)
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

    behaviors, history, articles, vectorizer, unique_entities, cols_explode, rename_cols = _preprocessing(
        behaviors, history, articles, test, sample, npratio
    )

    unique_categories = get_unique_categories(articles)

    df_features = behaviors.pipe(_build_features_behaviors, history=history, articles=articles,
                                 cols_explode=cols_explode, rename_columns=rename_cols, unique_entities=unique_entities,
                                 unique_categories=unique_categories)\
        .drop(NEW_VERSION_FEATURES)
    return df_features, vectorizer, unique_entities


def strip_new_features(df: pl.DataFrame) -> pl.DataFrame:
    '''
    Strip the new features from the dataframe, i.e. the features that are not present in the 68 features version.

    Args:
        df: the dataframe containing the features
    '''
    columns = df.columns
    for col in columns:
        if col in NEW_VERSION_FEATURES:
            df = df.drop(columns=col)
    return df
