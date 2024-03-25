from tqdm import tqdm
from rich.progress import Progress
from scipy import stats

try:
    import polars as pl
except ImportError:
    print("polars not available")

from _polars import *
from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
)
"""
Utils for catboost.
""" 


def build_features(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame, 
                   test: bool = False, sample: bool = True, npratio: int = 2) -> pl.DataFrame:
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
    - most frequent category in the user history of seen articles
    - percentage of seen articles with type different from article_default by the user
    - percentage of articles by the user in the history that contains each given entity
    
    Args:
        behaviors: the dataframe containing the impressions
        history: the dataframe containing the users history
        articles: the dataframe containing the articles features
        test: if true consider the behaviors as a test split, so it will not attempt to build the target column
        sample: if true, behaviors will be sampled using wu strategy, done only if test=False 
            (otherwise there is no information about negative articles in the inview list)
        npratio: the number of negative samples for wu sampling (useful only if sample=True)
        
    Returns:
        pl.DataFrame: the dataframe with all the features.
    '''
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
        
    unique_entities = articles.select('entity_groups').explode('entity_groups')['entity_groups'].unique().to_list()
    unique_entities = [e for e in unique_entities if e is not None]
    
    history = _build_history_features(history, articles, unique_entities)
        
    df_features = behaviors.select(['impression_id', 'article_ids_inview', 'article_id', 'impression_time', 'labels', 
                                    'device_type', 'read_time', 'scroll_percentage', 'user_id', 'is_sso_user', 'gender',
                                    'age', 'is_subscriber']) \
        .explode(columns_to_explode) \
        .rename(renaming_columns) \
        .unique(['impression_id', 'article']) \
        .with_columns(
            pl.col('impression_time').dt.weekday().alias('weekday'),
            pl.col('impression_time').dt.hour().alias('hour'),
            pl.col('article').cast(pl.Int32),
        ).join(articles.select(['article_id', 'premium', 'published_time', 'category', 
                                'sentiment_score', 'sentiment_label', 'entity_groups']),
            left_on='article', right_on='article_id', how='left') \
        .with_columns(
            (pl.col('impression_time') - pl.col('published_time')).dt.total_days().alias('article_delay_days'),
            (pl.col('impression_time') - pl.col('published_time')).dt.total_hours().alias('article_delay_hours')
        ).drop(['impression_time', 'published_time', 'article_id']) \
        .with_columns(
            pl.col('entity_groups').list.contains(entity).alias(f'Entity_{entity}_Present')
            for entity in unique_entities
        ).join(history.drop(['entity_groups_detailed']), on='user_id', how='left').with_columns(
            (pl.col('category') == pl.col('MostFrequentCategory')).alias('IsFavouriteCategory'),
            pl.col('category_right').list.n_unique().alias('NumberDifferentCategories'),
            list_pct_matches_with_col('category_right', 'category').alias('PctCategoryMatches'),
        ).drop('category_right')
    
    df_features = add_features_JS_history_topics(df_features, history)
    df_features = _add_entity_features(df_features, history) 
    return df_features


def add_features_JS_history_topics(train_ds, articles, history):
    """
    Returns train_ds enriched with features computed using the user's history.
    For each impression (user_id, article_id) considers the user's history (composed of "n" articles) and computes "n" Jaccard Similarity values, between the set of
    topics of the article of the impression and the "n" sets of topics of the articles in the user's history.
    Then, these "n" values get aggregated using mean, min, max, std.dev.

     Args:
        train_ds: The training dataset (Can contain any feature, but it MUST contain user_id and article)
        articles: The articles dataset (MUST contain article_id and topics)
        history: The history dataset (MUST contain user_id and article_id_fixed)

    Returns:
        pl.DataFrame: The training dataset with added features
    """
    article_ds= articles.select(["article_id","topics"]).rename({"article_id":"article"})
    history_ds = history.select(["user_id","article_id_fixed"])
    
    df = pl.concat(
    (
    rows.select(["impression_id","user_id","article"]) #Select only useful columns
        .join(article_ds,on="article",how="left") #Add topics of the inview_article
        .join(other = history_ds, on = "user_id",how="left") #Add history of the user
        .explode("article_id_fixed") #explode the user's history
        #For each article of the user's history, add its topics
        .join(other = article_ds.rename({"article":"article_id_fixed","topics":"topics_history"}), on="article_id_fixed",how="left")
        #add the JS between the topics of the article_inview and the topics of the article in the history
        .with_columns(
        (pl.col("topics").list.set_intersection(pl.col("topics_history")).list.len().truediv(
            pl.col("topics").list.set_union(pl.col("topics_history")).list.len()
        )).alias("JS")
        ).group_by(["impression_id","article"]).agg([ #grouping on all the "n" articles in the user's history, compute aggregations of the "n" JS values
            pl.col("JS").mean().alias("mean_JS"),
            pl.col("JS").min().alias("min_JS"),
            pl.col("JS").max().alias("max_JS"),
            pl.col("JS").std().alias("std_JS")]
        )
    for rows in tqdm(train_ds.iter_slices(100), total = train_ds.shape[0] // 100) #Process train_ds in chunks of rows
    )

    )
    return train_ds.join(other = df, on=["impression_id","article"], how="left")
        
        
def _build_history_features(history: pl.DataFrame, articles: pl.DataFrame, unique_entities: List[str]) -> pl.DataFrame:
    '''
    Builds all the features of the users history. These features are:
    - number of articles seen
    - statistics of the user read time (median, max, sum), scroll percentage (median, max), impression
      hours (mode), impression day (mode)
    - percentage of articles with sentiment label Positive, Negative and Neutral
    - most frequent category in the user history of seen articles
    - percentage of seen articles with type different from article_default
    - percentage of articles in the history that contains each given entity
    
    Args:
        history: the (raw) users history dataframe
        articles: the (raw) articles dataframe
        unique_entities: a list containing all the possible/considered unique entity groups of the articles
        
    Returns:
        pl.DataFrame: the preprocessed history dataframe
    '''
    history = _preprocess_history_df(history, articles)
    
    history = history.with_columns(
        pl.col('read_time_fixed').list.len().alias('NumArticlesHistory'),
        pl.col('read_time_fixed').list.median().alias('MedianReadTime'),
        pl.col('read_time_fixed').list.max().alias('MaxReadTime'),
        pl.col('read_time_fixed').list.sum().alias('TotalReadTime'),
        pl.col('scroll_percentage_fixed').list.median().alias('MedianScrollPercentage'),
        pl.col('scroll_percentage_fixed').list.max().alias('MaxScrollPercentage'),
        pl.col('impression_time_fixed').list.eval(pl.element().dt.weekday()).alias('weekdays'),
        pl.col('impression_time_fixed').list.eval(pl.element().dt.hour()).alias('hours'),
    ).with_columns(
        pl.col('weekdays').map_elements(lambda x: stats.mode(x)[0], return_dtype=pl.Int64).cast(pl.Int8).alias('MostFrequentWeekday'),
        pl.col('hours').map_elements(lambda x: stats.mode(x)[0], return_dtype=pl.Int64).cast(pl.Int8).alias('MostFrequentHour'),
        pl.col('category').map_elements(lambda x: stats.mode(x)[0], return_dtype=pl.Int64).cast(pl.Int16).alias('MostFrequentCategory'),
        (1 - (pl.col('article_type').list.count_matches('article_default') / pl.col('NumArticlesHistory'))).alias('PctNotDefaultArticles'),
        (pl.col('sentiment_label').list.count_matches('Negative') / pl.col('NumArticlesHistory')).alias('NegativePct'),
        (pl.col('sentiment_label').list.count_matches('Positive') / pl.col('NumArticlesHistory')).alias('PositivePct'),
        (pl.col('sentiment_label').list.count_matches('Neutral') / pl.col('NumArticlesHistory')).alias('NeutralPct'),
        # TODO: not sure about how sentiment scores are since there are 3 classes
        # probably the score is related to the class so this should be done differently for each class
        # pl.col('sentiment_score').list.mean().alias('MeanSentimentScore'),
        # pl.col('sentiment_score').list.max().alias('MaxSentimentScore'),
        # pl.col('sentiment_score').list.min().alias('MinSentimentScore'),
    ).drop(
        ['read_time_fixed', 'scroll_percentage_fixed', 'impression_time_fixed', 
        'weekdays', 'hours', 'sentiment_label', 'sentiment_score', 'article_type']
    ).with_columns(
        (pl.col('entity_groups').list.count_matches(entity) / pl.col('NumArticlesHistory')).alias(f'{entity}Pct')
        for entity in unique_entities
    ).drop('entity_groups')
    
    return history
    
    
def _preprocess_history_df(history: pl.DataFrame, articles: pl.DataFrame) -> pl.DataFrame:
    '''
    Retrieves the categories, the article types, the sentiment labels and scores from the seen articles by each user
    in the history dataframe, i.e. of the articles whose ids are in the article_id_fixed column.
    
    Args:
        history: the dataframe containing the history of the users
        articles: the dataframe containing the articles features
        
    Returns:
        pl.DataFrame: the history dataframe, together with the columns 'category', 'article_type', 'sentiment_label', 
        'sentiment_score', 'entity_groups', 'entity_groups_detailed'. The difference between 'entity_groups' and 
        'entity_groups_detailed' is that the first one is a list of entities (all the concatenated entities from all
        the seen articles, after applying .unique() in each single list and flattening the set of lists) and the second
        one is a list of lists (where each list contains the unique entities of in the correspoding article of article_ids_fixed)  
    '''
    columns = ['category', 'article_type', 'sentiment_label', 'sentiment_score']
    return_dtypes = [pl.Int64, pl.String, pl.String, pl.Float64]
    with Progress() as progress: 
        
        tasks = {}
        for col in columns:
            tasks[col] = progress.add_task(f"Getting {col}", total=history.shape[0])
        tasks['entity_groups'] = progress.add_task("Getting entity_groups", total=history.shape[0])
        tasks['entity_groups_detailed'] = progress.add_task("Getting detailed entity_groups", total=history.shape[0])

        history = history.with_columns(
            [pl.col('article_id_fixed').map_elements(get_single_feature_function(articles, 'article_id', col, 
                                                                                 progress, tasks[col]), 
                                                    return_dtype=pl.List(dtype)).alias(col)
            for col, dtype in zip(columns, return_dtypes)] + \
            [pl.col('article_id_fixed').map_elements(get_unique_list_exploded_feature_function(articles, 'article_id', 'entity_groups', 
                                                                                               progress, tasks['entity_groups']), 
                                                    return_dtype=pl.List(pl.String)).alias('entity_groups'),
            pl.col('article_id_fixed').map_elements(get_unique_list_feature_function(articles, 'article_id', 'entity_groups', 
                                                                                     progress, tasks['entity_groups_detailed']), 
                                                    return_dtype=pl.List(pl.List(pl.String))).alias('entity_groups_detailed')]
        ).drop('article_id_fixed')
    return history


def _add_entity_features(df_features: pl.DataFrame, history: pl.DataFrame) -> pl.DataFrame:
    '''
    Adds the entity features to the df_features dataframe, from the history dataframe (already preprocessed
    with the 'entity_groups_detailed' column). The dataframe df_features must be already preprocessed with the
    'entity_groups' of the target article in each row.
    
    Args:
        df_features: the training/validation/testing dataset with partial features and the 'entity_groups' column
        history: the users history, with already the 'entity_groups_detailed' column
    
    Returns:
        pl.DataFrame: df_features without 'entity_groups' column, and with two additional features, MeanCommonEntities and
        MeanCommonEntities, that contains the mean and the maximum common entities with the articles in the history
    '''
    entities_df = pl.concat(
        (
            rows.select(['impression_id', 'user_id', 'article', 'entity_groups']) \
                .join(history.select(['user_id', 'entity_groups_detailed']), on='user_id', how='left') \
                .explode('entity_groups_detailed')
                .with_columns(
                    pl.col('entity_groups').list.set_intersection(pl.col('entity_groups_detailed')).list.len().alias('common_entities')
                ).drop(['entity_groups_detailed', 'entity_groups']) \
                .group_by(['impression_id', 'article']).agg(
                    pl.col('common_entities').mean().alias('MeanCommonEntities'),
                    pl.col('common_entities').max().alias('MeanCommonEntities'),
                )
            for rows in tqdm.tqdm(df_features.iter_slices(100), total=df_features.shape[0] // 100)
        )
    )
    return df_features.join(entities_df, on=['impression_id', 'article'], how='left').drop(['entity_groups'])