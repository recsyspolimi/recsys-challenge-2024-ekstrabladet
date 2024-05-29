import polars as pl
import pandas as pd
from tqdm import tqdm
import os
import json
import numpy as np
from catboost import CatBoostClassifier, CatBoostRanker, Pool, sum_models
from sklearn.utils import resample
from polimi.utils._inference import _inference
import gc
from ebrec.evaluation.metrics_protocols import *
import catboost
from ebrec.utils._behaviors import sampling_strategy_wu2019
import matplotlib.pyplot as plt
from fastauc.fastauc.fast_auc import CppAuc
from polimi.utils._polars import reduce_polars_df_memory_size

# TARGET 0.8037703435093432

# TARGET DEMO 0.7916280544043559

dataset_path = '/home/ubuntu/experiments/preprocessing_train_small_new'
original_datset_path = '/home/ubuntu/dataset/ebnerd_small/train/behaviors.parquet'
validation_path = '/home/ubuntu/experiments/subsample_validation_small_new'

# dataset_path = '/home/ubuntu/experiments/preprocessing_train_2024-05-18_09-34-07'
# validation_path = '/home/ubuntu/experiments/preprocessing_validation_2024-05-18_09-43-19'

catboost_params = {
    "iterations": 1000,
    "subsample": 0.5,
    "rsm": 0.7
}

EVAL = True
SAVE_FEATURES_ORDER = False
SAVE_PREDICTIONS = True
N_BATCH = 10
NPRATIO = 2

to_drop = [ 'sum_RP3betaRecommender_ner_scores_l_inf_impression', 'sum_PureSVDItemRecommender_ner_scores_l_inf_impression', 
    'sum_ItemKNNCFRecommender_ner_scores_l_inf_impression', 'sum_MatrixFactorization_BPR_Cython_ner_scores_l_inf_impression', 
    'sum_P3alphaRecommender_ner_scores_l_inf_impression', 'max_RP3betaRecommender_ner_scores_l_inf_impression', 
    'max_PureSVDItemRecommender_ner_scores_l_inf_impression', 'max_ItemKNNCFRecommender_ner_scores_l_inf_impression',
    'max_MatrixFactorization_BPR_Cython_ner_scores_l_inf_impression', 'max_P3alphaRecommender_ner_scores_l_inf_impression',
    'mean_RP3betaRecommender_ner_scores_l_inf_impression', 'mean_PureSVDItemRecommender_ner_scores_l_inf_impression', 
    'mean_ItemKNNCFRecommender_ner_scores_l_inf_impression', 'mean_MatrixFactorization_BPR_Cython_ner_scores_l_inf_impression',
    'mean_P3alphaRecommender_ner_scores_l_inf_impression'
    ]
emb = [
    'contrastive_vector_scores_weighted_scroll_percentage_fixed_mmnorm_l1_w_mean_l_inf_impression','xlm_roberta_base_scores_weighted_scroll_percentage_fixed_mmnorm_l1_w_mean_l_inf_impression', 
    'image_embeddings_scores_weighted_scroll_percentage_fixed_mmnorm_l1_w_mean_l_inf_impression',
    'bert_base_multilingual_cased_scores_weighted_scroll_percentage_fixed_mmnorm_l1_w_mean_l_inf_impression', 
    'contrastive_vector_scores_weighted_scroll_percentage_fixed_mmnorm_l1_w_max_l_inf_impression', 
    'xlm_roberta_base_scores_weighted_scroll_percentage_fixed_mmnorm_l1_w_max_l_inf_impression', 
    'image_embeddings_scores_weighted_scroll_percentage_fixed_mmnorm_l1_w_max_l_inf_impression', 
    'bert_base_multilingual_cased_scores_weighted_scroll_percentage_fixed_mmnorm_l1_w_max_l_inf_impression',
    'contrastive_vector_scores_weighted_scroll_percentage_fixed_mmnorm_l1_w_std_l_inf_impression', 
    'xlm_roberta_base_scores_weighted_scroll_percentage_fixed_mmnorm_l1_w_std_l_inf_impression', 
    'image_embeddings_scores_weighted_scroll_percentage_fixed_mmnorm_l1_w_std_l_inf_impression', 
    'bert_base_multilingual_cased_scores_weighted_scroll_percentage_fixed_mmnorm_l1_w_std_l_inf_impression',
    'contrastive_vector_scores_weighted_scroll_percentage_fixed_mmnorm_l1_w_median_l_inf_impression', 
    'xlm_roberta_base_scores_weighted_scroll_percentage_fixed_mmnorm_l1_w_median_l_inf_impression', 
    'image_embeddings_scores_weighted_scroll_percentage_fixed_mmnorm_l1_w_median_l_inf_impression', 
    'bert_base_multilingual_cased_scores_weighted_scroll_percentage_fixed_mmnorm_l1_w_median_l_inf_impression',
    'PureSVDRecommender', 'ItemKNNCFRecommender', 'RP3betaRecommender', 'SLIM_BPR_Cython', 'PureSVDItemRecommender']

click_feat = [ 'emotions_user_item_distance', 'word2vec_user_item_distance', 'contrastive_user_item_distance', 'roberta_user_item_distance',
              'distil_user_item_distance', 'bert_user_item_distance', 'kenneth_user_item_distance', 'TW_click_predictor_distilbert', 
              'SP%W_click_predictor_distilbert', 'readtime_click_predictor_distilbert', 'TW_click_predictor_bert',
              'SP%W_click_predictor_bert', 'readtime_click_predictor_bert', 'TW_click_predictor_roberta',
              'SP%W_click_predictor_roberta', 'readtime_click_predictor_roberta', 'TW_click_predictor_kenneth',
              'SP%W_click_predictor_kenneth', 'readtime_click_predictor_kenneth', 'TW_click_predictor_w2v', 'SP%W_click_predictor_w2v',
              'readtime_click_predictor_w2v', 'TW_click_predictor_emotion', 'SP%W_click_predictor_emotion',
              'readtime_click_predictor_emotion']

contrastive = ['TW_click_predictor_contrastive', 'SP%W_click_predictor_contrastive', 'readtime_click_predictor_contrastive','contrastive_user_item_distance']

emotions = ['user_emotion0', 'user_emotion1', 'user_emotion2', 'user_emotion3', 'user_emotion4', 'user_emotion5', 'emotion_0', 
            'emotion_1', 'emotion_2', 'emotion_3', 'emotion_4', 'emotion_5',]

drop_me = to_drop + click_feat + contrastive + emotions
def drop_history_max_len(ds):
    max_allowed_articels = ds.select('NumArticlesHistory').describe(percentiles = (0.95))\
        .filter(pl.col('statistic') == '95%').select('NumArticlesHistory').item()
        
    return ds.filter(pl.col('NumArticlesHistory') < max_allowed_articels)

def drop_history_min_len(ds):
    max_allowed_articels = ds.select('NumArticlesHistory').describe(percentiles = (0.95))\
        .filter(pl.col('statistic') == '95%').select('NumArticlesHistory').item()
        
    return ds.filter(pl.col('NumArticlesHistory') > max_allowed_articels)
        
def build_topic_endorsement(dataset, type):
    period ='10h'
    
    articles = pl.read_parquet('/home/ubuntu/dataset/ebnerd_small/articles.parquet')
    behaviors_train = pl.read_parquet(f'/home/ubuntu/dataset/ebnerd_small/{type}/behaviors.parquet')
    history_train = pl.read_parquet(f'/home/ubuntu/dataset/ebnerd_small/{type}/history.parquet')
    
    end_2 = dataset.select('article', 'impression_time')\
        .join(articles.select(['article_id','topics']), left_on = 'article', right_on = 'article_id')\
        .explode('topics')\
     .with_columns(pl.col('impression_time').dt.round('1m').alias('impression_time_rounded'))\
     .group_by(['impression_time_rounded','topics']).len()\
        .rename({'impression_time_rounded': 'impression_time', 'len':f'topic_endorsement_{period}'}) \
        .sort("impression_time").set_sorted("impression_time") \
        .rolling(index_column="impression_time", period=period, group_by='topics').agg(
            pl.col(f'topic_endorsement_{period}').sum()
        ).unique(subset = ['topics','impression_time'])

    aggregate_col = end_2.columns
    drop_cols = ['article', 'impression_time', 'impression_id','topics']
    aggregate_col = [c for c in aggregate_col if c not in drop_cols]

    to_join = dataset.select('article', 'impression_time')\
            .join(articles.select(['article_id','topics']), left_on = 'article', right_on = 'article_id')\
            .explode('topics')\
            .join_asof(end_2, by='topics', on='impression_time')\
            .unique(subset = ['article','topics','impression_time'])\
            .pivot(index=['article','impression_time'], columns="topics", values="topic_endorsement_10h").fill_null(0).fill_nan(0)
        
    dataset = dataset.join(to_join, on = ['article','impression_time'])
    
    return dataset

if __name__ == '__main__':
    
    train_ds = reduce_polars_df_memory_size(pl.read_parquet(os.path.join(dataset_path, 'train_ds.parquet')), verbose=False)
    # train_ds = train_ds.drop(columns = drop_me)
    
    # self_ds = reduce_polars_df_memory_size(pl.read_parquet('/home/ubuntu/experiments/test_batch_training/self_supervised_ds.parquet'), verbose=False)
    
    starting_dataset =  pl.read_parquet(original_datset_path).select(['impression_id','user_id','article_ids_inview','article_ids_clicked'])
    
    behaviors = pl.concat(
        rows.pipe(
            sampling_strategy_wu2019, npratio=NPRATIO, shuffle=False, with_replacement=True, seed=123
        ).explode('article_ids_inview').drop(columns =['article_ids_clicked']).rename({'article_ids_inview' : 'article'})\
        .with_columns(pl.col('user_id').cast(pl.UInt32),
                      pl.col('article').cast(pl.Int32))\
        
         for rows in tqdm(starting_dataset.iter_slices(1000), total=starting_dataset.shape[0] // 1000)
    )
        
    train_ds = behaviors.join(train_ds, on = ['impression_id','user_id','article'], how = 'left')
    # train_ds = drop_history_max_len(train_ds)
    print(train_ds.shape[0])
    column_oder = train_ds.columns
    # train_ds = train_ds.vstack(self_ds.select(column_oder))
    print(train_ds.shape[0])
    print(f'N features {len(column_oder)}')
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)

    print(f'Data info: {data_info}')

    print(f'Starting to train the catboost model')
    
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
    
    categorical_columns = data_info['categorical_columns']
    categorical_columns = [cat for cat in categorical_columns if cat in column_oder]  
    # train_ds = build_topic_endorsement(train_ds, 'train')
         
    if 'postcode' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in train_ds.columns:
        train_ds = train_ds.drop(['impression_time'])
        
    train_ds = train_ds.drop(['impression_id', 'article', 'user_id']).to_pandas()
    train_ds[categorical_columns] = train_ds[categorical_columns].astype('category')

    X = train_ds.drop(columns = ['target'])
    print(X.columns[0])
    y = train_ds['target']
    
    model = CatBoostClassifier(**catboost_params, cat_features=categorical_columns)
    model.fit(X, y, verbose=25)
    
        
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    fig = plt.figure(figsize=(10, 60))
    print(np.array(X.columns)[sorted_idx])
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
    plt.title('Feature Importance')
    plt.savefig('/home/ubuntu/experiments/test_batch_training/feature_importance.png', bbox_inches='tight')
    plt.close()
    
    if SAVE_FEATURES_ORDER:
        feature_importance_list =  np.array(X.columns)[sorted_idx]
        file = open('/home/ubuntu/experiments/test_batch_training/feature_importance.txt','w')
        for item in feature_importance_list:
            file.write(item+"\n")
        file.close()

            
    if EVAL:
        with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
            data_info = json.load(data_info_file)

        categorical_columns = data_info['categorical_columns']
        categorical_columns = [cat for cat in categorical_columns if cat in column_oder] 
        
        
        val_ds = pl.read_parquet(validation_path + '/validation_ds.parquet').select(column_oder)
        # val_ds = build_topic_endorsement(val_ds, 'validation')
        
        if 'postcode' in val_ds.columns:
            val_ds = val_ds.with_columns(pl.col('postcode').fill_null(5))
        if 'article_type' in val_ds.columns:
            val_ds = val_ds.with_columns(
                pl.col('article_type').fill_null('article_default'))

        if 'impression_time' in val_ds.columns:
            val_ds = val_ds.drop('impression_time')

        val_ds_pandas = val_ds.drop(
            ['impression_id', 'article', 'user_id']).to_pandas()

        val_ds_pandas[categorical_columns
                      ] = val_ds_pandas[categorical_columns].astype('category')

        X_val = val_ds_pandas.drop(columns = ['target'])
        # X_val = X_val.drop(columns = drop_features)
        # X_val = X_val.drop(columns = to_drop)
        # X_val = X_val.drop(columns = click_feat)
        y_val = val_ds_pandas['target']
        
        evaluation_ds = val_ds[['impression_id', 'article', 'target']]        
        prediction_ds = evaluation_ds.with_columns(pl.Series(model.predict_proba(X_val)[:, 1]).alias('prediction'))
        if SAVE_PREDICTIONS:
            prediction_ds.write_parquet('/home/ubuntu/experiments/test_batch_training/classifier_predictions.parquet')
        prediction_ds = prediction_ds.group_by('impression_id').agg(pl.col('target'), pl.col('prediction'))
            

        cpp_auc = CppAuc()
        result = np.mean(
            [cpp_auc.roc_auc_score(np.array(y_t).astype(bool), np.array(y_s).astype(np.float32)) 
                for y_t, y_s in zip(prediction_ds['target'].to_list(), 
                                    prediction_ds['prediction'].to_list())]
        )
        print('AUC : ')
        print(result)