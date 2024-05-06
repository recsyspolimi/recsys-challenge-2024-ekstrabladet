import polars as pl
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
from polimi.preprocessing_pipelines.pre_127 import _build_new_features


def build_features_iterator(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                            test: bool = False, sample: bool = True, npratio: int = 2,
                            tf_idf_vectorizer: TfidfVectorizer = None, n_batches: int = 10, previous_version=None,
                            missing : pl.DataFrame = None, **kwargs): 

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
    print(missing)
    
    n_batches = 10


    print('Building features...')
    for sliced_df in missing.iter_slices(missing.shape[0] // n_batches):
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
        
 
if __name__ == '__main__':
    dataset = []
    behaviors = pl.read_parquet('/home/ubuntu/dataset/ebnerd_large/validation/behaviors.parquet')
    history = pl.read_parquet('/home/ubuntu/dataset/ebnerd_large/validation/history.parquet')
    articles = pl.read_parquet('/home/ubuntu/dataset/ebnerd_large/articles.parquet')
    is_test_data = False
    sample = False
    previous_version = '/home/ubuntu/experiments/preprocessing_validation_2024-04-23_08-38-39/validation_ds.parquet'
    split_type = 'validation'
    dataset_type = 'validation'
    
    print(pl.read_parquet('/home/ubuntu/tmp/missing_validation_ds.parquet'))
    
    print('Reading old dataset...')
    for i in range(99):
        dataset.append(
           pl.read_parquet(f'/home/ubuntu/experiments/preprocessing_validation_2024-04-29_13-17-47/Sliced_ds/validation_slice_{i}.parquet')
               .select(['impression_id','article','user_id'])
        )
        
    dataset_complete = pl.concat(dataset, how='vertical_relaxed')
    print(dataset_complete)
    
    previous = pl.read_parquet(previous_version)
    precomputed = dataset_complete.select(['impression_id','article','user_id'])
    old = previous.join(precomputed, on=['impression_id','article','user_id'], how="anti")
    
    
    print('Resuming preprocessing...')
    missing = []
    i = 0
    for dataset, vectorizer, unique_entities in build_features_iterator(behaviors, history, articles, test=is_test_data, 
                                                                        sample=sample, n_batches=10 ,previous_version = previous_version,
                                                                        split_type=dataset_type,
                                                                        missing= old):
                                                                        # urm_path=urm_path):
        print(f'Batch {i}')
        i += 1
        missing.append(dataset)
        
    missing_complete = pl.concat(missing, how='vertical_relaxed')
    missing_complete.write_parquet('/home/ubuntu/tmp/missing_validation_ds.parquet')
    dataset_complete = pl.concat([dataset_complete, missing_complete], how='vertical_relaxed')
    
    dataset_complete.write_parquet('/home/ubuntu/tmp/validation_ds.parquet')
    
    shape = behaviors.select(['impression_id','article_ids_inview','user_id']).explode('article_ids_inview')\
                     .rename({'article_ids_inview': 'article'})\
                     .join(dataset_complete, on=['impression_id','article','user_id'], how='anti').shape
                   
    assert pl.read_parquet(previous_version).shape[0] ==  dataset_complete.shape[0]
    assert shape[0] == 0
