import nltk
import polars as pl
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import tqdm


def _compute_topic_model(articles, n_components=5):
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('danish')
    title_vectorizer = CountVectorizer(stop_words=stopwords)

    titles_bow = title_vectorizer.fit_transform(articles['title'].to_list())

    lda_model = LatentDirichletAllocation(
        n_components=n_components,
        doc_topic_prior=0.99,
        topic_word_prior=0.75,
        learning_method='online'
    )
    articles = articles.with_columns(
        pl.Series(lda_model.fit_transform(titles_bow).astype(np.float32))
        .list.to_array(5).alias('topic_model_embeddings')
    )

    topic_model_columns = [
        f'topic_model_embedding_{i}' for i in range(n_components)]

    articles = articles.with_columns(
        pl.col('topic_model_embeddings').arr.to_struct(
            fields=lambda i: f'topic_model_embedding_{i}')
    ).with_columns([
        pl.col('topic_model_embeddings').struct.field(f'topic_model_embedding_{i}') for i in range(n_components)
    ]).drop('topic_model_embeddings')

    return articles, topic_model_columns, n_components


def add_topic_model_features(df, history, articles, topic_model_columns, n_components):
    
    prev_train_columns = [c for c in df.columns if c not in ['impression_id', 'article']]

    return pl.concat(
        rows.join(history.select(
            ['user_id', 'article_id_fixed', 'impression_time_fixed']), on='user_id', how='left')
        .join(articles.select(['article_id', 'topics', 'entity_groups', 'topics_idf'] + topic_model_columns),
              left_on='article', right_on='article_id', how='left')
        .explode(['article_id_fixed', 'impression_time_fixed'])
        .join(articles.select(['article_id', 'topics', 'entity_groups'] + topic_model_columns),
              left_on='article_id_fixed', right_on='article_id', how='left')
        .rename({'topics_right': 'topics_history', 'entity_groups_right': 'entity_groups_history'})
        .rename({f'topic_model_embedding_{i}_right': f'topic_model_embedding_{i}_history' for i in range(n_components)})
        .with_columns(
            # 1/delay gives the weight for the weighted mean of the lda embeddings
            (1 / (pl.col('impression_time') - pl.col('impression_time_fixed')
                  ).dt.total_hours()).alias('history_weight'),
            (pl.col("topics").list.set_intersection(pl.col("topics_history")).list.len().truediv(
                pl.col("topics").list.set_union(
                    pl.col("topics_history")).list.len()
            )).alias("JS"),
            pl.col('entity_groups').list.set_intersection(
                pl.col('entity_groups_history')).list.len().alias('common_entities'),
        ).drop(['entity_groups_history', 'entity_groups', 'topics', 'topics_history']) \
        .with_columns(
            # summing delays to normalize them before weighted mean
            pl.col('history_weight').sum().over(
                ['impression_id', 'article']).alias('history_weight_sum'),
            *[pl.col(x).mul(pl.col(f'{x}_history')).alias(f'{x}_dot')
              for x in topic_model_columns],
            *[pl.col(x).mul(pl.col(x)) for x in topic_model_columns],
            *[pl.col(f'{x}_history').mul(pl.col(f'{x}_history')
                                         ).alias(f'{x}_history_square') for x in topic_model_columns],
        ).with_columns(
            # weights now sum to 1
            pl.col('history_weight').truediv(pl.col('history_weight_sum')),
            pl.sum_horizontal(topic_model_columns).sqrt().alias(
                'topic_model_norm'),
            pl.sum_horizontal([f'{x}_history_square' for x in topic_model_columns]).sqrt(
            ).alias('topic_model_history_norm'),
            pl.sum_horizontal([f'{x}_dot' for x in topic_model_columns]).alias(
                'topic_model_dot'),
        ).with_columns(
            *[pl.col(f'{x}_history').mul(pl.col('history_weight')).alias(f'{x}_history_weighted')
              for i, x in enumerate(topic_model_columns)],
            pl.col('topic_model_dot').truediv(pl.col('topic_model_norm').mul(
                'topic_model_history_norm')).alias('topic_model_cosine')
        ).group_by(['impression_id', 'article']).agg(
            pl.col(prev_train_columns).first(),
            pl.col("JS").mul(pl.col("history_weight")
                             ).sum().alias("weighted_mean_JS"),
            pl.col("topic_model_cosine").mean().alias(
                "mean_topic_model_cosine"),
            pl.col("topic_model_cosine").min().alias("min_topic_model_cosine"),
            pl.col("topic_model_cosine").max().alias("max_topic_model_cosine"),
            pl.col("topic_model_cosine").std().alias("std_topic_model_cosine"),
            pl.col("topic_model_cosine").mul(pl.col("history_weight")
                                             ).sum().alias("weighted_mean_topic_model_cosine"),
            *[pl.col(f'{x}_history').mean().alias(f'lda_{i}_history_mean')
              for i, x in enumerate(topic_model_columns)],
            *[pl.col(f'{x}_history_weighted').sum().alias(f'lda_{i}_history_weighted_mean') for i, x in enumerate(topic_model_columns)]
        ).drop(['topics_idf', 'topics_flatten', 'topics_flatten_tf_idf', 'category_right'])
        for rows in tqdm.tqdm(df.iter_slices(10000), total=df.shape[0] // 10000))
