import polars as pl
import pandas as pd
from tqdm import tqdm
import os
import json
import numpy as np
from ebrec.evaluation.metrics_protocols import *
from ebrec.utils._behaviors import sampling_strategy_wu2019
from polimi.utils._polars import reduce_polars_df_memory_size
from datetime import datetime
import argparse
import pandas as pd
import json
import numpy as np
from typing_extensions import List, Tuple, Dict, Type
import polars as pl
from polimi.utils._tuning_params import get_models_params
import gc
from polars import testing

TRAIN = True
output_dir = '/home/ubuntu/experiments/dataset_level_2'
paths_features = {
            'train_df' : '/mnt/ebs_volume/experiments/preprocessing_validation_new',
            'test_df': '/mnt/ebs_volume/experiments/preprocessing_test_new',
            'train_icm': '/mnt/ebs_volume/icm_features/large/recsys_validation.parquet',
            'test_icm': '/mnt/ebs_volume/icm_features/large/recsys_train_large.parquet'
    }

models = [
    {
        'model_name': 'cat_rnk_new_95',
        'model_train_path': '/mnt/ebs_volume/stacking/pred_val_large/pred_val_cat_rnk_new_95.parquet',
        'model_test_path': '/mnt/ebs_volume/stacking/pred_test/pred_test_cat_rnk_new_95.parquet'
    },
    {
        'model_name': 'catboost_new_noK',
        'model_train_path': '/mnt/ebs_volume/stacking/pred_val_large/pred_val_catboost_new_noK.parquet',
        'model_test_path': '/mnt/ebs_volume/stacking/pred_test/pred_test_catboost_new_noK.parquet'
    },
    {
        'model_name': 'mlp_new_trial_208',
        'model_train_path': '/mnt/ebs_volume/stacking/pred_val_large/pred_val_mlp_new_trial_208.parquet',
        'model_test_path': '/mnt/ebs_volume/stacking/pred_test/pred_test_mlp_new_trial_208.parquet'
    },
    {
        'model_name': 'gandalf_new_trial_130',
        'model_train_path': '/mnt/ebs_volume/stacking/pred_val_large/pred_val_gandalf_new_trial_130.parquet',
        'model_test_path': '/mnt/ebs_volume/stacking/pred_test/pred_test_gandalf_new_trial_130.parquet',
    },
    {
        'model_name': 'deep_cross_new_trial_67',
        'model_train_path': '/mnt/ebs_volume/stacking/pred_val_large/pred_val_deep_cross_new_trial_67.parquet',
        'model_test_path': '/mnt/ebs_volume/stacking/pred_test/pred_test_deep_cross_new_trial_67',
    },
    {
        'model_name': 'wide_deep_new_trial_72',
        'model_train_path': '/mnt/ebs_volume/stacking/pred_val_large/pred_val_wide_deep_new_trial_72.parquet',
        'model_test_path': '/mnt/ebs_volume/stacking/pred_test/pred_test_wide_deep_new_trial_72.parquet',
    }
]
drop_me = ['Category_auto_Pct', 'Category_bibliotek_Pct', 'Category_biler_Pct', 'Category_dagsorden_Pct', 'Category_ferie_Pct', 'Category_forbrug_Pct', 'Category_haandvaerkeren_Pct', 'Category_horoskoper_Pct', 'Category_incoming_Pct',
           'Category_krimi_Pct', 'Category_musik_Pct', 'Category_nationen_Pct', 'Category_nyheder_Pct', 'Category_om_ekstra_bladet_Pct', 'Category_opinionen_Pct', 'Category_penge_Pct', 'Category_plus_Pct',
           'Category_podcast_Pct', 'Category_services_Pct', 'Category_video_Pct', 'Category_vin_Pct', 'EVENTPct', 'Entity_EVENT_Present', 'Entity_PER_Present', 'LOCPct', 'MISCPct',
           'MaxReadTime', 'MaxScrollPercentage', 'MedianReadTime', 'MedianScrollPercentage', 'MostFrequentCategory', 'MostFrequentHour', 'MostFrequentWeekday', 'NegativePct', 'NeutralPct', 'NumArticlesHistory',
           'NumberDifferentCategories', 'ORGPct', 'PERPct', 'PRODPct', 'PctCategoryMatches', 'PctNotDefaultArticles', 'PctStrongNegative', 'PctStrongNeutral', 'PctStrongPositive', 'PositivePct', 'TotalReadTime', 'age', 'clicked_count_l_inf_impression', 'endorsement_20h_articleuser_l_inf_articleuser', 'endorsement_20h_articleuser_macd',
           'endorsement_10h_leak_macd', 'entropy_impression_mean_JS', 'entropy_impression_std_JS', 'entropy_impression_topics_cosine', 'entropy_impression_trendiness_score_3d_leak', 'entropy_impression_trendiness_score_category', 'gender', 'is_already_seen_article', 'is_inside_window_1', 'kurtosis_impression_article_delay_hours', 'kurtosis_impression_endorsement_10h', 'kurtosis_impression_endorsement_10h_leak',
           'kurtosis_impression_inview_count', 'kurtosis_impression_mean_JS', 'kurtosis_impression_mean_topic_model_cosine', 'kurtosis_impression_std_JS', 'kurtosis_impression_topics_cosine', 'kurtosis_impression_total_read_time',
           'kurtosis_impression_trendiness_score_3d', 'kurtosis_impression_trendiness_score_3d_leak', 'kurtosis_impression_trendiness_score_5d', 'kurtosis_impression_trendiness_score_category', 'last_session_duration', 'last_session_time_hour_diff', 'lda_0_history_mean', 'lda_0_history_weighted_mean', 'lda_1_history_mean', 'lda_1_history_weighted_mean', 'lda_2_history_mean', 'lda_2_history_weighted_mean', 'lda_3_history_mean', 'lda_3_history_weighted_mean', 'lda_4_history_mean', 'lda_4_history_weighted_mean', 'max_ner_item_knn_scores', 'max_ner_svd_scores', 'max_topic_model_cosine', 'mean_ner_item_knn_scores',
           'mean_ner_svd_scores', 'mean_prev_sessions_duration', 'mean_topic_model_cosine', 'mean_topic_model_cosine_l_inf_article', 'mean_topic_model_cosine_l_inf_impression', 'mean_topic_model_cosine_l_inf_user', 'mean_topic_model_cosine_minus_median_impression', 'mean_topic_model_cosine_rank_impression', 'mean_user_trendiness_score', 'min_JS',
           'min_topic_model_cosine', 'num_topics', 'postcode', 'sentiment_label_diversity_impression', 'skew_impression_inview_count', 'skew_impression_mean_topic_model_cosine', 'skew_impression_std_JS', 'skew_impression_topics_cosine', 'skew_impression_total_pageviews/inviews', 'skew_impression_trendiness_score_3d', 'skew_impression_trendiness_score_3d_leak', 'skew_impression_trendiness_score_5d', 'skew_impression_trendiness_score_category', 'std_impression_mean_JS',
           'std_impression_mean_topic_model_cosine', 'std_impression_std_JS', 'std_impression_topics_cosine', 'std_impression_trendiness_score_3d', 'std_impression_trendiness_score_3d_leak', 'std_impression_trendiness_score_5d', 'std_impression_trendiness_score_category', 'std_topic_model_cosine', 'topics_cosine_l_inf_user', 'total_ner_item_knn_scores', 'total_ner_svd_scores',
           'total_pageviews_minus_median_impression', 'total_pageviews_rank_impression', 'total_read_time_l_inf_impression', 'total_read_time_minus_median_impression', 'total_read_time_rank_impression', 'trendiness_score_1d/5d', 'weighted_mean_topic_model_cosine', 'window_0_history_length', 'window_1_history_length', 'window_2_history_length', 'window_3_history_length', 'window_topics_score'
           ]


def load_predictions(directories, model_list, test=False):
    model_name = model_list[0]
    print(f'Loading Predictions for {model_name}')
    merged_df = reduce_polars_df_memory_size(pl.read_parquet(directories[0]), verbose=0)\
        .sort(by=['impression_id', 'article']).rename({'prediction': f'prediction_{model_name}'})
    merged_df = merged_df.filter(pl.col('impression_id') != 0)
    original_shape = merged_df.shape[0]
    for df in range(1, len(model_list)):
        model_name = model_list[df]
        print(f'Loading Predictions for {model_name}')
        model_predictions = reduce_polars_df_memory_size(
            pl.read_parquet(directories[df]), verbose=0).sort(by=['impression_id', 'article'])
        if not test:
            testing.assert_frame_equal(merged_df.select(['impression_id', 'article', 'target']),
                                       model_predictions.select(['impression_id', 'article', 'target']))
        else:
            testing.assert_frame_equal(merged_df.select(['impression_id', 'article']),
                                       model_predictions.select(['impression_id', 'article']))
        merged_df = merged_df.with_columns(
            model_predictions['prediction'].alias(f'prediction_{model_name}')
        )
        assert original_shape == merged_df.shape[0]

    return merged_df


def preprocessing(df, path_features, hybrid_weights=[], MODEL_LIST=[], drop_me=[], keep_old_features=False, test=False):
    print('Normalizing Predictions')
    df = df.with_columns(
        *[
            ((pl.col(f'prediction_{model}')-pl.col(f'prediction_{model}').min().over('impression_id')) /
             (pl.col(f'prediction_{model}').max().over('impression_id')-pl.col(f'prediction_{model}').min().over('impression_id'))).alias(f'normalized_prediction_{model}')
            for model in MODEL_LIST
        ]
    ).with_columns(
        *[((pl.col(f'prediction_{model}')-pl.col(f'prediction_{model}').min().over('article')) /
           (pl.col(f'prediction_{model}').max().over('article')-pl.col(f'prediction_{model}').min().over('article'))).alias(f'art_norm_prediction_{model}')
          for model in MODEL_LIST]
    ).with_columns(
        *[(hybrid_weights[i] * pl.col(f'normalized_prediction_{MODEL_LIST[i]}')).alias(f'prediction_hybrid_{MODEL_LIST[i]}') for i in range(len(MODEL_LIST))]
    ).with_columns(
        pl.sum_horizontal([f"prediction_hybrid_{model}" for model in MODEL_LIST]).alias(
            'prediction_hybrid')
    ).drop([f'prediction_hybrid_{model}' for model in MODEL_LIST])

    print('Building Features')
    df = df.with_columns(
        *[pl.col(f'prediction_{model}').mean().over('impression_id').alias(
            f'mean_prediction_{model}') for model in (MODEL_LIST + ['hybrid'])],
        *[pl.col(f'prediction_{model}').skew().over('impression_id').alias(
            f'skew_prediction_{model}') for model in (MODEL_LIST + ['hybrid'])],
        *[pl.col(f'prediction_{model}').std().over('impression_id').alias(
            f'std_prediction_{model}') for model in (MODEL_LIST + ['hybrid'])],
        *[pl.col(f'prediction_{model}').median().over('impression_id').alias(
            f'median_prediction_{model}') for model in (MODEL_LIST + ['hybrid'])],
        *[pl.col(f'prediction_{model}').rank(method='min', descending=True).over(
            'impression_id').alias(f'rank_prediction_{model}') for model in (MODEL_LIST + ['hybrid'])],
    ).rename({'prediction_hybrid': 'normalized_prediction_hybrid'}).with_columns(
        pl.mean_horizontal([f'art_norm_prediction_{model}' for model in MODEL_LIST]).alias(
            'art_norm_horizontal_mean'),
        pl.min_horizontal([f'art_norm_prediction_{model}' for model in MODEL_LIST]).alias(
            'art_norm_horizontal_min'),
        pl.max_horizontal([f'art_norm_prediction_{model}' for model in MODEL_LIST]).alias(
            'art_norm_horizontal_max'),
        pl.mean_horizontal([f'normalized_prediction_{model}' for model in (
            MODEL_LIST + ['hybrid'])]).alias('horizontal_mean'),
        pl.min_horizontal([f'normalized_prediction_{model}' for model in (
            MODEL_LIST + ['hybrid'])]).alias('horizontal_min'),
        pl.max_horizontal([f'normalized_prediction_{model}' for model in (
            MODEL_LIST + ['hybrid'])]).alias('horizontal_max'),
        pl.mean_horizontal([f'rank_prediction_{model}' for model in (
            MODEL_LIST + ['hybrid'])]).alias('avg_rank_pos'),
        pl.min_horizontal([f'rank_prediction_{model}' for model in (
            MODEL_LIST + ['hybrid'])]).alias('min_rank_pos'),
        pl.max_horizontal([f'rank_prediction_{model}' for model in (
            MODEL_LIST + ['hybrid'])]).alias('max_rank_pos'),
        *[(pl.col(f'prediction_{model}') - pl.col(f'median_prediction_{model}')).alias(
            f'prediction_{model}_minus_median') for model in MODEL_LIST],
    ).rename({'normalized_prediction_hybrid': 'prediction_hybrid'}).with_columns(
        pl.col('impression_id').count().over(
            'impression_id').alias('n_articles_impression')
    ).with_columns(
        *[(pl.col(f'rank_prediction_{model}')/pl.col('n_articles_impression')).alias(f'normalized_rank_prediction_{model}') for model in (MODEL_LIST + ['hybrid'])]
    ).with_columns(
        pl.mean_horizontal([f'normalized_rank_prediction_{model}' for model in (
            MODEL_LIST + ['hybrid'])]).alias('avg_norm_rank_pos'),
        pl.min_horizontal([f'normalized_rank_prediction_{model}' for model in (
            MODEL_LIST + ['hybrid'])]).alias('min_norm_rank_pos'),
        pl.max_horizontal([f'normalized_rank_prediction_{model}' for model in (
            MODEL_LIST + ['hybrid'])]).alias('max_norm_rank_pos'),
    )

    quantile_95 = df.quantile(quantile=0.95).select('horizontal_mean').item()
    
    print('Building aggregations')
    df = df.with_columns(
        pl.col('horizontal_mean').mean().over(
            'article').alias('mean_article_horizontal_mean'),
        pl.col('avg_rank_pos').le(3.5).cast(pl.UInt32).alias('is_avg_top_3'),
        pl.col('avg_rank_pos').le(1.5).cast(pl.UInt32).alias('is_avg_top_1'),
        pl.col('horizontal_mean').gt(quantile_95).cast(
            pl.UInt32).alias('over_95_qt'),
        *[pl.col(f'rank_prediction_{model}').le(3).cast(pl.UInt32).alias(
            f'is_top_3_{model}') for model in (MODEL_LIST + ['hybrid'])],
        *[pl.col(f'rank_prediction_{model}').le(1).cast(pl.UInt32).alias(
            f'is_top_1_{model}') for model in (MODEL_LIST + ['hybrid'])],
    ).with_columns(
        pl.sum_horizontal([f'is_top_3_{model}' for model in (
            MODEL_LIST + ['hybrid'])]).alias('agreement_top3'),
        pl.sum_horizontal([f'is_top_1_{model}' for model in (
            MODEL_LIST + ['hybrid'])]).alias('agreement_top1')
    ).with_columns(
        pl.col('agreement_top3') /
        pl.col('n_articles_impression').alias('normalized_agreement_top3'),
        pl.col('agreement_top1') /
        pl.col('n_articles_impression').alias('normalized_agreement_top1')
    ).drop([f'is_top_1_{model}' for model in (MODEL_LIST + ['hybrid'])] + [f'is_top_3_{model}' for model in (MODEL_LIST + ['hybrid'])])

    print('Reading old features')
    if keep_old_features:
        if not test:
            df = df.drop(columns=['target']).join(pl.scan_parquet(path_features).drop(
                drop_me).collect(), on=['impression_id', 'article'], how='left')
        else:
            df = df.join(pl.scan_parquet(path_features).drop(
                drop_me).collect, on=['impression_id', 'article'], how='left')
    else:
        if not test:
            df = df.drop(columns=['target']).join(pl.scan_parquet(path_features).drop(drop_me).select(
                ['impression_time', 'impression_id', 'article', 'user_id', 'target']).collect(), on=['impression_id', 'article'], how='left')
        else:
            df = df.join(pl.scan_parquet(path_features).drop(drop_me).select(['impression_time', 'impression_id', 'article', 'user_id']).collect(),
                         on=['impression_id', 'article'], how='left')

    print('Building rolling features')
    df = df.with_columns(
        (pl.col('horizontal_mean') - pl.col('horizontal_mean').rolling_mean_by(window_size='1h',
         by='impression_time').over('article')).alias('roll_mean_1h_horizontal_mean'),
        (pl.col('horizontal_mean') - pl.col('horizontal_mean').rolling_mean_by(window_size='1d',
         by='impression_time').over('article')).alias('roll_mean_1d_horizontal_mean'),
        (pl.col('art_norm_horizontal_mean') - pl.col('art_norm_horizontal_mean').rolling_mean_by(window_size='1h',
         by='impression_time').over('article')).alias('roll_mean_1h_art_morm_horizontal_mean'),
        (pl.col('art_norm_horizontal_mean') - pl.col('art_norm_horizontal_mean').rolling_mean_by(window_size='1d',
         by='impression_time').over('article')).alias('roll_mean_1d_art_morm_horizontal_mean'),
        (pl.col('prediction_hybrid') - pl.col('prediction_hybrid').rolling_mean_by(window_size='1h',
         by='impression_time').over('article')).alias('roll_mean_1h_prediction_hybrid'),
        (pl.col('prediction_hybrid') - pl.col('prediction_hybrid').rolling_mean_by(window_size='1d',
         by='impression_time').over('article')).alias('roll_mean_1d_prediction_hybrid')
    )

    print('Building aggregations over rolling features')
    df = df.with_columns(
        *[(pl.col(f'prediction_{model}') - pl.col(f'prediction_{model}').rolling_mean_by('impression_time',
           window_size='1h')).alias(f'prediction_{model}_minus_mean_rolling_1h') for model in (MODEL_LIST + ['hybrid'])],
        *[(pl.col(f'prediction_{model}') - pl.col(f'prediction_{model}').rolling_mean_by('impression_time',
           window_size='1d')).alias(f'prediction_{model}_minus_mean_rolling_1d') for model in (MODEL_LIST + ['hybrid'])],
        *[(pl.col(f'prediction_{model}') - pl.col(f'prediction_{model}').rolling_mean_by('impression_time', window_size='1h').over(
            'user_id')).alias(f'prediction_{model}_minus_mean_rolling_user_1h') for model in (MODEL_LIST + ['hybrid'])],
        *[(pl.col(f'prediction_{model}') - pl.col(f'prediction_{model}').rolling_mean_by('impression_time', window_size='1d').over(
            'user_id')).alias(f'prediction_{model}_minus_mean_rolling_user_1d') for model in (MODEL_LIST + ['hybrid'])],
        *[(pl.col(f'prediction_{model}') - pl.col(f'prediction_{model}').rolling_mean_by('impression_time', window_size='1h').over(
            'category')).alias(f'prediction_{model}_minus_mean_rolling_category_1h') for model in (MODEL_LIST + ['hybrid'])],
        *[(pl.col(f'prediction_{model}') - pl.col(f'prediction_{model}').rolling_mean_by('impression_time', window_size='1d').over(
            'category')).alias(f'prediction_{model}_minus_mean_rolling_category_1d') for model in (MODEL_LIST + ['hybrid'])],
    )

    return df
NORMALIZE_OVER_USER_ID = [
    'emb_bert_icm_recsys', 
    'emb_contrastive_icm_recsys', 
    'emb_emotions_icm_recsys', 
    'emb_roberta_icm_recsys', 
    'emb_w_2_vec_icm_recsys', 
    'emb_kenneth_icm_recsys', 
    'emb_distilbert_icm_recsys'
    ]

NORMALIZE_OVER_USER_ID = [
    'emb_bert_icm_recsys', 
    'emb_contrastive_icm_recsys', 
    'emb_emotions_icm_recsys', 
    'emb_roberta_icm_recsys', 
    'emb_w_2_vec_icm_recsys', 
    'emb_kenneth_icm_recsys', 
    'emb_distilbert_icm_recsys'
    ]

# NORMALIZE_OVER_USER_ID = [
#     'kenneth_emb_icm_recsys',
#     'distilbert_emb_icm',
#     'bert_emb_icm',
#     'roberta_emb_icm',
#     'w_2_vec_emb_icm',
#     'emotions_emb_icm',
#     'constrastive_emb_icm'
# ]
# C = [
#     'kenneth_emb_icm',
#     'distilbert_emb_icm',
#     'bert_emb_icm',
#     'roberta_emb_icm',
#     'w_2_vec_emb_icm',
#     'emotions_emb_icm',
#     'constrastive_emb_icm'
# ]

def build_icm_features(df, path):
    print('Reading ICM')
    recsys_features = pl.scan_parquet(path).collect()
    print('Building ICM features')
    df = df.join(recsys_features, on=['impression_id', 'article', 'user_id'], how= 'left')
    df = df.with_columns(
        *[(pl.col(c) / pl.col(c).max().over(pl.col('user_id'))).alias(f'{c}_l_inf_user_id')
        for c in NORMALIZE_OVER_USER_ID],
        *[pl.col(c).std().over(pl.col('user_id')).alias(f'std_user_id_{c}')
        for c in NORMALIZE_OVER_USER_ID],
        *[pl.col(c).skew().over(pl.col('user_id')).alias(f'skew_user_id_{c}')
        for c in NORMALIZE_OVER_USER_ID],
        *[pl.col(c).kurtosis().over(pl.col('user_id')).alias(f'kurtosis_user_id_{c}')
        for c in NORMALIZE_OVER_USER_ID],
        *[pl.col(c).entropy().over(pl.col('user_id')).alias(f'entropy_user_id_{c}')
        for c in NORMALIZE_OVER_USER_ID],
        *[(pl.col(c) - pl.col(c).median().over(pl.col('user_id'))).alias(f'{c}_minus_median_user_id')
        for c in NORMALIZE_OVER_USER_ID],
        
        *[(pl.col(c) / pl.col(c).max().over(pl.col('article'))).alias(f'{c}_l_inf_article')
        for c in NORMALIZE_OVER_ARTICLE],
        *[pl.col(c).std().over(pl.col('article')).alias(f'std_article_{c}')
        for c in NORMALIZE_OVER_ARTICLE],
        *[pl.col(c).skew().over(pl.col('article')).alias(f'skew_article_{c}')
        for c in NORMALIZE_OVER_ARTICLE],
        *[pl.col(c).kurtosis().over(pl.col('article')).alias(f'kurtosis_article_{c}')
        for c in NORMALIZE_OVER_ARTICLE],
        *[pl.col(c).entropy().over(pl.col('article')).alias(f'entropy_article_{c}')
        for c in NORMALIZE_OVER_ARTICLE],
        *[(pl.col(c) - pl.col(c).median().over(pl.col('article'))).alias(f'{c}_minus_median_article')
        for c in NORMALIZE_OVER_ARTICLE],
    )
    
    return df

if __name__ == '__main__':
    print(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if TRAIN:
        directories = [model['model_train_path'] for model in models]
        model_list = [model['model_name'] for model in models]
        df = load_predictions(directories, model_list, test=False)
        weights = [1/len(model_list)] * len(model_list)
        df = preprocessing(df, paths_features['train_df']+ '/validation_ds.parquet', hybrid_weights=weights, MODEL_LIST=model_list, drop_me=drop_me, keep_old_features=True, test=False)
        df = build_icm_features(df, paths_features['train_icm'])
        
        with open(os.path.join(paths_features['train_df'], 'data_info.json')) as data_info_file:
            data_info = json.load(data_info_file)

        categorical_columns = []
        for col in data_info['categorical_columns']:
                if col in df.columns:
                        categorical_columns.append(col)
        categorical_columns + ['is_avg_top_3', 'is_avg_top_1', 'over_95_qt']
        
        df.write_parquet(output_dir + '/train_ds.parquet')
        dataset_info = {
            'categorical_columns': categorical_columns,
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        }
        data_info_path = os.path.join(output_dir, 'data_info.json')
        with open(data_info_path, 'w') as data_info_file:
            json.dump(dataset_info, data_info_file)
    
    TRAIN = False
    
    print(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not TRAIN:
        directories = [model['model_test_path'] for model in models]
        model_list = [model['model_name'] for model in models]
        df = load_predictions(directories, model_list, test=False)
        weights = [1/len(model_list)] * len(model_list)
        df = preprocessing(df, paths_features['test_df']+ '/validation_ds.parquet', hybrid_weights=weights, MODEL_LIST=model_list, drop_me=drop_me, keep_old_features=True, test=True)
        df = build_icm_features(df, paths_features['test_icm'])
        
        with open(os.path.join(paths_features['test_df'], 'data_info.json')) as data_info_file:
            data_info = json.load(data_info_file)

        categorical_columns = []
        for col in data_info['categorical_columns']:
                if col in df.columns:
                        categorical_columns.append(col)
        categorical_columns + ['is_avg_top_3', 'is_avg_top_1', 'over_95_qt']
        
        df.write_parquet(output_dir + '/test_ds.parquet')
        dataset_info = {
            'categorical_columns': categorical_columns,
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        }
        data_info_path = os.path.join(output_dir, 'data_info.json')
        with open(data_info_path, 'w') as data_info_file:
            json.dump(dataset_info, data_info_file)