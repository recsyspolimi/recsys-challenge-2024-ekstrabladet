from polimi.utils._polars import reduce_polars_df_memory_size
import json
import polars as pl
from pathlib import Path
import tqdm
import gc

NORMALIZE_OVER_USER_ID = [
    'total_pageviews/inviews', 'endorsement_10h', 'trendiness_score_3d',
    'total_pageviews', 'total_read_time', 'total_inviews', 'article_delay_hours'
]
COMPLETE_NORMALIZE_OVER_USER_ID = [
    'mean_JS', 'std_JS', 'mean_topic_model_cosine', 'topics_cosine',
]
NORMALIZE_OVER_ARTICLE = [
    'article_delay_hours', 'mean_JS', 'std_JS', 'mean_topic_model_cosine', 'topics_cosine',
]
CATEGORICAL_ENTROPY = [
    'category', 'sentiment_label', 'article_type', 'premium'
]

expressions = [
    [(pl.col(c) / pl.col(c).max().over(pl.col('user_id'))).alias(f'{c}_l_inf_user')
    for c in NORMALIZE_OVER_USER_ID],
    [(pl.col(c) - pl.col(c).median().over(pl.col('user_id'))).alias(f'{c}_minus_median_user')
    for c in COMPLETE_NORMALIZE_OVER_USER_ID],
    [pl.col(c).entropy().over(pl.col('user_id')).alias(f'{c}_entropy_user')
    for c in COMPLETE_NORMALIZE_OVER_USER_ID],
    [pl.col(c).entropy().over(pl.col('user_id')).alias(f'{c}_skew_user')
    for c in COMPLETE_NORMALIZE_OVER_USER_ID],
    [(pl.col(c) - pl.col(c).median().over(pl.col('article'))).alias(f'{c}_minus_median_article')
    for c in NORMALIZE_OVER_ARTICLE],
    [pl.col(c).entropy().over(pl.col('article')).alias(f'{c}_entropy_article')
    for c in NORMALIZE_OVER_ARTICLE],
    [pl.col(c).skew().over(pl.col('article')).alias(f'{c}_skew_article')
    for c in NORMALIZE_OVER_ARTICLE],
    [pl.col(c).value_counts().struct[1].entropy().over('impression_id').alias(f'{c}_distribution_entropy_impression')
    for c in CATEGORICAL_ENTROPY]
]

if __name__ == '__main__':
    
    OUTPUT_DIR_DATASET = '/home/ubuntu/tmp_dataset/train_ds.parquet'
    
    with open('/home/ubuntu/dset_complete/data_info.json') as data_info_file:
        data_info = json.load(data_info_file)
        
    data_info['categorical_columns'] += ['cwh']

    with open('/home/ubuntu/tmp_dataset/data_info.json', 'w') as data_info_file:
        json.dump(data_info, data_info_file)
    
    history = pl.read_parquet('/home/ubuntu/dataset/ebnerd_small/train/history.parquet')
    articles = pl.read_parquet('/home/ubuntu/dataset/ebnerd_small/articles.parquet')
    behaviors = pl.read_parquet('/home/ubuntu/dataset/ebnerd_small/train/behaviors.parquet')
    dataset = pl.read_parquet('/home/ubuntu/dset_complete/train_ds.parquet')
    
    history_counts_df = history.select(['user_id', 'article_id_fixed']) \
        .with_columns(pl.col('article_id_fixed').list.len().alias('history_len')).drop('article_id_fixed')
    num_unique_topics = len(articles.select(['topics']).explode('topics')['topics'].unique())
    num_unique_categories = len(articles['category'].unique())

    history_topics_proba_df = history.select(['user_id', 'article_id_fixed']) \
        .explode('article_id_fixed') \
        .join(articles.select(['article_id', 'topics']), left_on='article_id_fixed', right_on='article_id', how='left') \
        .explode('topics') \
        .pivot(values='article_id_fixed', index='user_id', columns='topics', aggregate_function='len') \
        .join(history_counts_df, on='user_id') \
        .with_columns((pl.all().exclude(['user_id', 'history_len']).fill_null(0) + 1) / (pl.col('history_len') + num_unique_topics)) \
        .drop('history_len')
        
    history_topics_proba_df = history_topics_proba_df.melt(
        id_vars='user_id', value_vars=[c for c in history_topics_proba_df.columns if c != 'user_id']
    ).rename({'variable': 'topic', 'value': 'smoothed_frequency_topic'})

    history_category_proba_df = history.select(['user_id', 'article_id_fixed']) \
        .explode('article_id_fixed') \
        .join(articles.select(['article_id', 'category']), left_on='article_id_fixed', right_on='article_id', how='left') \
        .pivot(values='article_id_fixed', index='user_id', columns='category', aggregate_function='len') \
        .join(history_counts_df, on='user_id') \
        .with_columns((pl.all().exclude(['user_id', 'history_len']).fill_null(0) + 1) / (pl.col('history_len') + num_unique_categories)) \
        .drop('history_len')
        
    history_category_proba_df = history_category_proba_df.melt(
        id_vars='user_id', value_vars=[c for c in history_category_proba_df.columns if c != 'user_id']
    ).rename({'variable': 'category', 'value': 'smoothed_frequency_category'}) \
    .with_columns(pl.col('category').cast(pl.Int16))
    
    history_cwh = history.select(['user_id', 'article_id_fixed', 'impression_time_fixed']) \
        .explode(['article_id_fixed', 'impression_time_fixed']) \
        .with_columns(
            pl.col('impression_time_fixed').dt.weekday().alias('weekday').cast(pl.String),
            (pl.col('impression_time_fixed').dt.hour().alias('hour_group') // 4).cast(pl.String),
        ).join(articles.select(['article_id', 'category']), left_on='article_id_fixed', right_on='article_id') \
        .with_columns(pl.concat_str(['category', 'weekday', 'hour_group'], separator='_').alias('cwh'))
        
    history_cwh_freq = history_cwh.select(pl.col('cwh').value_counts()).with_columns(
        pl.col('cwh').struct[0],
        pl.col('cwh').struct[1].alias('counts')
    ).with_columns(
        ((pl.col('counts') + 1) / (history_cwh.shape[0] + pl.col('cwh').count())).alias('cwh_prob')
    ).select(['cwh', 'cwh_prob'])
    
    history_cwh_freq_user = history_cwh.group_by('user_id').agg(
        pl.col('cwh').value_counts(), pl.col('cwh').count().alias('count_impressions')
    ).explode('cwh').with_columns(
        pl.col('cwh').struct[0],
        pl.col('cwh').struct[1].alias('counts')
    ).with_columns(
        ((pl.col('counts') + 1) / (pl.col('count_impressions') + pl.col('cwh').n_unique())).alias('cwh_prob_user')
    ).select(['user_id', 'cwh', 'cwh_prob_user'])

    user_probabilities = behaviors.select(['user_id', 'impression_id', 'article_ids_inview']) \
        .with_columns(pl.col('article_ids_inview').list.len().alias('inview_len')) \
        .explode('article_ids_inview').rename({'article_ids_inview': 'article'}) \
        .join(articles.select(['article_id', 'category', 'topics']), left_on='article', right_on='article_id', how='left') \
        .join(history_category_proba_df, on=['user_id', 'category'], how='left') \
        .explode('topics').join(history_topics_proba_df, left_on=['user_id', 'topics'], right_on=['user_id', 'topic'], how='left') \
        .group_by(['user_id', 'impression_id', 'article']).agg(
            pl.col('smoothed_frequency_category').first().alias('score_category'),
            pl.col('smoothed_frequency_topic').product().alias('score_topic') * (pl.col('inview_len').first()**(pl.col('topics').count() - 1)),
        ).with_columns((pl.col('score_category') * pl.col('score_topic')).alias('score_topic_category'))
        
    history = history.select(['user_id', 'article_id_fixed', 'impression_time_fixed', 'scroll_percentage_fixed', 'read_time_fixed']) \
        .explode('article_id_fixed', 'impression_time_fixed', 'scroll_percentage_fixed', 'read_time_fixed') \
        .join(articles.select(['article_id', 'category', 'article_type', 'sentiment_label', 'premium']),
            left_on='article_id_fixed', right_on='article_id', how='left') \
        .with_columns(pl.col('impression_time_fixed').dt.round('1h').alias('impression_hour')) \
        .with_columns(
            pl.col('impression_time_fixed').dt.weekday().alias('weekday'),
            pl.col('impression_time_fixed').dt.weekday().gt(5).alias('is_weekend'),
            (pl.col('impression_time_fixed').dt.hour() // 4).alias('hour_window'),
            pl.col('impression_time_fixed').dt.hour().alias('hour'),
        ).group_by('user_id').agg(
            pl.col('category').n_unique().alias('num_categories_history'),
            pl.col("category").value_counts().struct[1].entropy().alias("content_diversity"),
            pl.col("article_type").value_counts().struct[1].entropy().alias("article_type_diversity"),
            pl.col("sentiment_label").value_counts().struct[1].entropy().alias("sentiment_label_diversity"),
            pl.col("premium").value_counts().struct[1].entropy().alias("premium_diversity"),
            pl.col("hour").value_counts().struct[1].entropy().alias("hour_diversity"),
            pl.col("hour_window").value_counts().struct[1].entropy().alias("hour_window_diversity"),
            pl.col("weekday").value_counts().struct[1].entropy().alias("weekday_diversity"),
            pl.col("is_weekend").value_counts().struct[1].entropy().alias("is_weekend_diversity"),
            pl.col("is_weekend").mean().alias('weekend_pct'),
            pl.col("premium").mean().alias('premium_pct'),
        )
        
    history = reduce_polars_df_memory_size(history)
    user_probabilities = reduce_polars_df_memory_size(user_probabilities)
    history_cwh_freq_user = reduce_polars_df_memory_size(history_cwh_freq_user)
    history_cwh_freq = reduce_polars_df_memory_size(history_cwh_freq)
    
    del articles, behaviors, history_topics_proba_df, history_category_proba_df, history_counts_df
    gc.collect()

    dataset = dataset.with_columns(
        (pl.col('endorsement_10h') / pl.col('trendiness_score_1d')).alias('endorsement_over_trend'),
        (pl.col('normalized_endorsement_10h') / pl.col('normalized_trendiness_score_overall')).alias('norm_endorsement_over_trend'),
        (pl.col('user_mean_delay_hours') / pl.col('mean_topics_mean_delay_hours')).alias('user_over_topics_delay_hours')
    ).join(history, on='user_id', how='left').with_columns(
        pl.col('category').cast(pl.String).alias('category_string')
    ).join(user_probabilities, on=['user_id', 'article', 'impression_id'], how='left').drop('category_string') \

    # bayes rule: P(click(user)|cwh) = P(cwh|click(user)) * P(click(user)) / P(cwh)
    # assuming P(click(user)) prior as uniform across the inview
    dataset = dataset.with_columns(pl.concat_str(['category', 'weekday', pl.col('hour') // 4], separator='_').alias('cwh')) \
        .join(history_cwh_freq_user, on=['user_id', 'cwh'], how='left') \
        .join(history_cwh_freq, on='cwh', how='left') \
        .with_columns(pl.col('cwh_prob_user') / (pl.col('cwh_prob') * pl.col('article').count().over('impression_id')))
    
    for expressions_group in tqdm.tqdm(expressions):
        dataset = dataset.with_columns(expressions_group)
        dataset = reduce_polars_df_memory_size(dataset)
        
    dataset.write_parquet(OUTPUT_DIR_DATASET)