import os
from pathlib import Path
import polars as pl
from tqdm import tqdm
import scipy.sparse as sps
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
import polimi.preprocessing_pipelines.pre_127 as pre_127
from RecSys_Course_AT_PoliMi.Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender
from RecSys_Course_AT_PoliMi.Recommenders.BaseRecommender import BaseRecommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from RecSys_Course_AT_PoliMi.Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender, PureSVDItemRecommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from RecSys_Course_AT_PoliMi.Evaluation.Evaluator import EvaluatorHoldout
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from RecSys_Course_AT_PoliMi.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from polimi.utils._custom import load_sparse_csr, load_best_optuna_params,load_recommenders
from polimi.utils._urm import train_recommender, build_ner_scores_features, load_recommender, build_item_mapping, build_user_id_mapping



'''
New features:
    - ner
'''
CATEGORICAL_COLUMNS = ['device_type', 'is_sso_user', 'gender', 'is_subscriber', 'weekday',
                       'premium', 'category', 'sentiment_label', 'is_new_article', 'is_already_seen_article',
                       'MostFrequentCategory', 'MostFrequentWeekday', 'IsFavouriteCategory']
algo_dict= {
    UserKNNCFRecommender: {
        'params': {'similarity': 'tversky', 'topK': 590, 'shrink': 0, 
                   'tversky_alpha': 1.6829525098337292, 'tversky_beta': 0.13181828101203877},
        'study_name': 'UserKNNCFRecommender-ner-small-ndcg100',
        'load': False
    }

}
"""
algo_dict = {
    PureSVDItemRecommender: {
        'params': {'num_factors': 997, 'topK': 589},
        'study_name': 'PureSVDItemRecommender-ner-small-ndcg100',
        'load': False
    },
    P3alphaRecommender: {
        'params': {'topK': 486, 'normalize_similarity': True, 'alpha': 1.9993719084032937},
        'study_name': 'P3alphaRecommender-ner-small-ndcg100',
        'load': False
    },
    ItemKNNCFRecommender: {
        'params': {'similarity': 'tversky', 'topK': 222, 'shrink': 177, 
                   'tversky_alpha': 0.012267012177140928, 'tversky_beta': 1.3288117939629838},
        'study_name': 'ItemKNNCFRecommender-ner-small-ndcg100',
        'load': False
    },
    RP3betaRecommender: {
        'params': {'topK': 499, 'normalize_similarity': True, 'alpha': 1.9956096660427538, 'beta': 0.04484545361718186},
        'study_name': 'RP3betaRecommender-ner-small-ndcg100',
        'load': False
    },
    UserKNNCFRecommender: {
        'params': {'similarity': 'tversky', 'topK': 590, 'shrink': 0, 
                   'tversky_alpha': 1.6829525098337292, 'tversky_beta': 0.13181828101203877},
        'study_name': 'UserKNNCFRecommender-ner-small-ndcg100',
        'load': False
    }
}
"""

def build_features_iterator(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                            test: bool = False, sample: bool = True, npratio: int = 2,
                            tf_idf_vectorizer: TfidfVectorizer = None, n_batches: int = 10, previous_version=None,
                            urm_path: str = None, output_path: str = None, split_type: str = 'train',
                            recsys_models_path: str = None, recsys_urm_path: str = None):
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
    URM = load_sparse_csr(Path(os.path.join(urm_path, f'URM_{split_type}.npz')))
    ner_features = _build_ner_features(behaviors, history, articles, URM, Path(output_path))

    recs = []
    if recsys_urm_path and recsys_models_path:
        print('Preprocessing URM ...')
        URM_train = load_sparse_csr(Path(os.path.join(recsys_urm_path, f'URM_{split_type}.npz')))
        
        print('Preprocessing recsys models ...')
        recs = load_recommenders(URM=URM_train,file_path=recsys_models_path)

        recsys_features = build_recsys_features(history, behaviors,articles,recs)


    
    if not previous_version:
        
        for df_features, vectorizer, unique_entities in pre_127.build_features_iterator(
            behaviors, history, articles, test, sample, npratio, tf_idf_vectorizer, n_batches, previous_version
        ):
            df_features = df_features.join(ner_features, on=['impression_id', 'user_id', 'article'], how='left')
            if(len(recs)>0):
                df_features = df_features.join(recsys_features, on=['impression_id', 'user_id', 'article'], how='left')
            yield df_features, vectorizer, unique_entities
            
    else:
        # TODO: seed if the join needs to be done in slices
        # not a real iterator, but keeping compatibility with previous versions
        behaviors, history, articles, vectorizer, unique_entities, cols_explode, rename_cols = _preprocessing(
            behaviors, history, articles, test, sample, npratio
        )
        df_features = pl.read_parquet(previous_version)
        df_features = df_features.join(ner_features, on=['impression_id', 'user_id', 'article'], how='left')
        if(len(recs)>0):
            df_features = df_features.join(recsys_features, on=['impression_id', 'user_id', 'article'], how='left')
        yield df_features, vectorizer, unique_entities


def build_features_iterator_test(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                                 test: bool = False, sample: bool = True, npratio: int = 2,
                                 tf_idf_vectorizer: TfidfVectorizer = None, n_batches: int = 10, previous_version=None,
                                 urm_path: str = None, output_path: str = None, split_type: str = 'train',
                                 recsys_models_path: str = None, recsys_urm_path: str = None):
    
    URM = load_sparse_csr(Path(os.path.join(urm_path, f'URM_{split_type}.npz')))
    ner_features = _build_ner_features(behaviors, history, articles, URM, Path(output_path))

    recs = []
    if recsys_urm_path and recsys_models_path:
        print('Preprocessing URM ...')
        URM_train = load_sparse_csr(path=Path(recsys_urm_path))
        
        print('Preprocessing recsys models ...')
        recs = load_recommenders(URM=URM_train,file_path=recsys_models_path)

        recsys_features = build_recsys_features(history, behaviors,articles,recs)
    
    if not previous_version:
        
        for slice_features, vectorizer, unique_entities in pre_127.build_features_iterator_test(
            behaviors, history, articles, test, sample, npratio, tf_idf_vectorizer, n_batches, previous_version
        ):
            slice_features = slice_features.join(ner_features, on=['impression_id', 'user_id', 'article'], how='left')
            if(len(recs)>0):
                slice_features = slice_features.join(recsys_features, on=['impression_id', 'user_id', 'article'], how='left')
            yield slice_features, vectorizer, unique_entities
            
    else:
        behaviors, history, articles, vectorizer, unique_entities, cols_explode, rename_cols = _preprocessing(
            behaviors, history, articles, test, sample, npratio
        )
        for slice in tqdm(range(0,101)):
            slice_features = pl.read_parquet(os.path.join(previous_version, f'Sliced_ds/test_slice_{slice}.parquet'))
            slice_features = slice_features.join(ner_features, on=['impression_id', 'user_id', 'article'], how='left')
            if(len(recs)>0):
                slice_features = slice_features.join(recsys_features, on=['impression_id', 'user_id', 'article'], how='left')
            yield slice_features, vectorizer, unique_entities


def build_features(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                   test: bool = False, sample: bool = True, npratio: int = 2,
                   tf_idf_vectorizer: TfidfVectorizer = None, previous_version: pl.DataFrame = None,
                   urm_path: str = None, output_path: str = None, split_type: str = 'train',
                   recsys_models_path: str = None, recsys_urm_path: str = None) -> pl.DataFrame:
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
    URM = load_sparse_csr(Path(os.path.join(urm_path, f'URM_{split_type}.npz')))
    ner_features = _build_ner_features(behaviors, history, articles, URM, Path(output_path))

    recs = []
    if recsys_urm_path and recsys_models_path:
        print('Preprocessing URM ...')
        URM_train = load_sparse_csr(path=Path(recsys_urm_path))
        
        print('Preprocessing recsys models ...')
        recs = load_recommenders(URM=URM_train,file_path=recsys_models_path)

        recsys_features = build_recsys_features(history, behaviors,articles,recs)

    if not previous_version:
        df_features, vectorizer, unique_entities = pre_127.build_features(behaviors, history, articles, test, sample, npratio,
                                                                          tf_idf_vectorizer, previous_version)
    else:
        behaviors, history, articles, vectorizer, unique_entities, cols_explode, rename_cols = _preprocessing(
            behaviors, history, articles, test, sample, npratio
        )
        df_features = pl.read_parquet(previous_version)
        
    df_features = df_features.join(ner_features, on=['impression_id', 'user_id', 'article'], how='left')
    if(len(recs)>0):
        df_features = df_features.join(recsys_features, on=['impression_id', 'user_id', 'article'], how='left')
    return df_features, vectorizer, unique_entities


def _build_ner_features(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame,
                        URM: sps.csr_matrix, rec_output_dir: str = None) -> pl.DataFrame:
    recs = []
    for rec, info in algo_dict.items():
        params = info['params']
        study_name = info['study_name']
        if not params:
            print('Params are missing, loading best params...')
            params = load_best_optuna_params(study_name)
        
        if 'load' in info and info['load']:
            rec_instance = load_recommender(URM, rec, rec_output_dir, file_name=study_name)
        else:
            rec_instance = train_recommender(URM, rec, params, file_name=study_name, output_dir=rec_output_dir) #also saves the model
            
        recs.append(rec_instance)
        
    ner_features = build_ner_scores_features(history=history, behaviors=behaviors, articles=articles, recs=recs)
    return reduce_polars_df_memory_size(ner_features)




def build_recsys_features(history: pl.DataFrame, behaviors: pl.DataFrame, articles: pl.DataFrame, recs: list[BaseRecommender]):
    
    user_id_mapping = build_user_id_mapping(history)
    item_mapping = build_item_mapping(articles)
    
    recsys_scores = behaviors\
            .select('impression_id', 'article_ids_inview', 'user_id')\
            .explode('article_ids_inview')\
            .unique()\
            .rename({'article_ids_inview': 'article_id'})\
            .join(item_mapping, on='article_id')\
            .join(user_id_mapping, on='user_id')\
            .sort(['user_index', 'item_index'])\
            .rename({'article_id': 'article'})\
            .group_by('user_index').map_groups(lambda df: df.pipe(_compute_recommendations, recommenders=recs))\
            .drop('user_index')\
            .drop('item_index')
            
    recsys_scores = reduce_polars_df_memory_size(recsys_scores)
    
    return recsys_scores

    

            

def _compute_recommendations(user_items_df, recommenders):
    user_index = user_items_df['user_index'].to_list()[0]
    items = user_items_df['item_index'].to_numpy()

    scores = {}
    for rec in recommenders:
        scores[rec.RECOMMENDER_NAME] = rec._compute_item_score([user_index], items)[0, items]

    return user_items_df.with_columns(
        [
            pl.Series(model).alias(name) for name, model in scores.items()
        ]
    )