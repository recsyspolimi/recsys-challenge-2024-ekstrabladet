from polimi.utils._inference import _inference
from ebrec.utils._python import write_submission_file
from ebrec.evaluation.metrics_protocols import *
import os
import logging
from datetime import datetime
import argparse
import pandas as pd
import joblib
import json
import numpy as np
from typing_extensions import List
import polars as pl
from pathlib import Path
import tqdm
import gc
from catboost import CatBoostClassifier, CatBoostRanker

from fastauc.fastauc.fast_auc import CppAuc

import sys
sys.path.append('/home/ubuntu/RecSysChallenge2024/src')


LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def main(dataset_path, model_path, save_results, eval, behaviors_path, output_dir, batch_size=None, ranker = False):
    features_to_keep = [
 'impression_id',
 'user_id',
 'article',
 'device_type',
 'read_time',
 'scroll_percentage',
 'is_sso_user',
 'is_subscriber',
 'trendiness_score_3d',
 'weekday',
 'hour',
 'premium',
 'category',
 'sentiment_score',
 'sentiment_label',
 'num_images',
 'title_len',
 'subtitle_len',
 'body_len',
 'article_delay_days',
 'article_delay_hours',
 'Entity_MISC_Present',
 'Entity_ORG_Present',
 'Entity_PROD_Present',
 'Entity_LOC_Present',
 'is_new_article',
 'yesterday_category_daily_pct',
 'MeanCommonEntities',
 'MaxCommonEntities',
 'mean_JS',
 'max_JS',
 'std_JS',
 'topics_cosine',
 'IsFavouriteCategory',
 'Category_underholdning_Pct',
 'Category_migration_catalog_Pct',
 'Category_side9_Pct',
 'Category_tilavis_Pct',
 'Category_abonnement_Pct',
 'Category_webmaster-test-sektion_Pct',
 'Category_rssfeed_Pct',
 'Category_sport_Pct',
 'Category_sex_og_samliv_Pct',
 'Category_webtv_Pct',
 'Category_eblive_Pct',
 'mean_topics_trendiness_score',
 'mean_topics_mean_delay_days',
 'mean_topics_mean_delay_hours',
 'user_mean_delay_days',
 'user_mean_delay_hours',
 'is_inside_window_0',
 'is_inside_window_2',
 'is_inside_window_3',
 'window_category_score',
 'trendiness_score_category',
 'endorsement_10h',
 'weighted_mean_JS',
 'impression_time',
 'endorsement_10h_right',
 'normalized_endorsement_10h',
 'endorsement_10h_diff_rolling',
 'endorsement_10h_macd',
 'endorsement_10h_quantile_norm',
 'normalized_endorsement_10h_rolling_max_ratio',
 'endorsement_20h_articleuser',
 'normalized_endorsement_20h_articleuser',
 'endorsement_20h_articleuser_diff_rolling',
 'endorsement_20h_articleuser_quantile_norm',
 'normalized_endorsement_20h_articleuser_rolling_max_ratio',
 'endorsement_10h_leak',
 'normalized_endorsement_10h_leak',
 'endorsement_10h_leak_diff_rolling',
 'endorsement_10h_leak_quantile_norm',
 'normalized_endorsement_10h_leak_rolling_max_ratio',
 'trendiness_score_1d',
 'trendiness_score_5d',
 'trendiness_score_3d_leak',
 'trendiness_score_1d/3d',
 'normalized_trendiness_score_overall',
 'total_pageviews',
 'total_inviews',
 'total_read_time',
 'total_pageviews/inviews',
 'article_type',
 'clicked_count',
 'inview_count',
 'trendiness_score_3d_l_inf_impression',
 'trendiness_score_5d_l_inf_impression',
 'endorsement_10h_l_inf_impression',
 'total_pageviews/inviews_l_inf_impression',
 'mean_JS_l_inf_impression',
 'topics_cosine_l_inf_impression',
 'article_delay_hours_l_inf_impression',
 'total_pageviews_l_inf_impression',
 'total_inviews_l_inf_impression',
 'trendiness_score_category_l_inf_impression',
 'std_JS_l_inf_impression',
 'endorsement_10h_leak_l_inf_impression',
 'trendiness_score_3d_leak_l_inf_impression',
 'inview_count_l_inf_impression',
 'trendiness_score_3d_minus_median_impression',
 'trendiness_score_5d_minus_median_impression',
 'endorsement_10h_minus_median_impression',
 'total_pageviews/inviews_minus_median_impression',
 'mean_JS_minus_median_impression',
 'topics_cosine_minus_median_impression',
 'article_delay_hours_minus_median_impression',
 'total_inviews_minus_median_impression',
 'trendiness_score_category_minus_median_impression',
 'std_JS_minus_median_impression',
 'endorsement_10h_leak_minus_median_impression',
 'trendiness_score_3d_leak_minus_median_impression',
 'clicked_count_minus_median_impression',
 'inview_count_minus_median_impression',
 'article_delay_hours_rank_impression',
 'std_JS_rank_impression',
 'mean_topics_mean_delay_hours_rank_impression',
 'trendiness_score_3d_rank_impression',
 'trendiness_score_5d_rank_impression',
 'endorsement_10h_rank_impression',
 'total_pageviews/inviews_rank_impression',
 'mean_JS_rank_impression',
 'topics_cosine_rank_impression',
 'total_inviews_rank_impression',
 'trendiness_score_category_rank_impression',
 'endorsement_10h_leak_rank_impression',
 'trendiness_score_3d_leak_rank_impression',
 'clicked_count_rank_impression',
 'inview_count_rank_impression',
 'std_impression_endorsement_10h',
 'std_impression_total_pageviews/inviews',
 'std_impression_article_delay_hours',
 'std_impression_total_pageviews',
 'std_impression_total_inviews',
 'std_impression_total_read_time',
 'std_impression_endorsement_10h_leak',
 'std_impression_clicked_count',
 'std_impression_inview_count',
 'skew_impression_endorsement_10h',
 'skew_impression_mean_JS',
 'skew_impression_article_delay_hours',
 'skew_impression_total_pageviews',
 'skew_impression_total_inviews',
 'skew_impression_total_read_time',
 'skew_impression_endorsement_10h_leak',
 'skew_impression_clicked_count',
 'entropy_impression_trendiness_score_3d',
 'entropy_impression_trendiness_score_5d',
 'entropy_impression_endorsement_10h',
 'entropy_impression_total_pageviews/inviews',
 'entropy_impression_mean_topic_model_cosine',
 'entropy_impression_article_delay_hours',
 'entropy_impression_total_pageviews',
 'entropy_impression_total_inviews',
 'entropy_impression_total_read_time',
 'entropy_impression_endorsement_10h_leak',
 'entropy_impression_clicked_count',
 'entropy_impression_inview_count',
 'kurtosis_impression_total_pageviews/inviews',
 'kurtosis_impression_total_pageviews',
 'kurtosis_impression_total_inviews',
 'kurtosis_impression_clicked_count',
 'category_diversity_impression',
 'article_type_diversity_impression',
 'mean_JS_l_inf_user',
 'std_JS_l_inf_user',
 'article_delay_hours_l_inf_article',
 'mean_JS_l_inf_article',
 'std_JS_l_inf_article',
 'topics_cosine_l_inf_article']
    
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")

    dataset_name = 'validation' if eval else 'test'

    with open("/mnt/ebs_volume/experiments/dropped_features/data_info.json") as data_info_file: #<-- NB PATH HARDCODED!
        data_info = json.load(data_info_file)

    logging.info(f'Data info: {data_info}')
    logging.info(
        f'Categorical features: {np.array(data_info["categorical_columns"])}')
    logging.info(f'Reading model parameters from path: {model_path}')
    logging.info('Starting inference.')

    model = joblib.load(model_path)
    
    if isinstance(model, CatBoostClassifier):
        cat_features = model.get_param('cat_features')
        print(f'CatBoost categorical columns: {cat_features}')
    if isinstance(model, CatBoostRanker):
        cat_features = model.get_param('cat_features')
        print(f'CatBoost categorical columns: {cat_features}')
    

    evaluation_ds = _inference(os.path.join(
            dataset_path, f'{dataset_name}_ds.parquet'), data_info, model, eval, batch_size, ranker,drop_features=True, features_to_keep=features_to_keep)
    
    # if dataset_name == 'validation':
    #     evaluation_ds = _inference(os.path.join(
    #         dataset_path, f'{dataset_name}_ds.parquet'), data_info, model, eval, batch_size)
    # else:
    #     evaluation_ds = pl.concat(
    #         [_inference(os.path.join(dataset_path, f'Sliced_ds/test_slice_{i}.parquet'), data_info, model, eval, batch_size)
    #          for i in tqdm.tqdm(range(0, 101))], how='vertical_relaxed'
    #     )

    max_impression = evaluation_ds.select(
        pl.col('impression_id').max()).item(0, 0)

    gc.collect()

    logging.info('Inference completed.')

    if eval:
        evaluation_ds_grouped = evaluation_ds.group_by(
            'impression_id').agg(pl.col('target'), pl.col('prediction'))
        # met_eval = MetricEvaluator(
        #     labels=evaluation_ds_grouped['target'].to_list(),
        #     predictions=evaluation_ds_grouped['prediction'].to_list(),
        #     metric_functions=[
        #         AucScore(),
        #         MrrScore(),
        #         NdcgScore(k=5),
        #         NdcgScore(k=10),
        #     ],
        # )
        # logging.info(f'Evaluation results: {met_eval.evaluate()}')
        cpp_auc = CppAuc()
        auc= np.mean(
            [cpp_auc.roc_auc_score(np.array(y_t).astype(bool), np.array(y_s).astype(np.float32)) 
                for y_t, y_s in zip(evaluation_ds_grouped['target'].to_list(), 
                                    evaluation_ds_grouped['prediction'].to_list())]
        )
        logging.info(f'Evaluation results: {auc}')

    if save_results:
        evaluation_ds.write_parquet(os.path.join(
            output_dir, f'predictions.parquet'))
        path = Path(os.path.join(output_dir, 'predictions.txt'))

        # need to maintain the same order of the inview list
        behaviors = pl.read_parquet(behaviors_path, columns=[
                                    'impression_id', 'article_ids_inview', 'user_id'])
        ordered_predictions = behaviors.explode('article_ids_inview').with_row_index() \
            .join(evaluation_ds, left_on=['impression_id', 'article_ids_inview', 'user_id'],
                  right_on=['impression_id', 'article', 'user_id'], how='left') \
            .sort('index').group_by(['impression_id', 'user_id'], maintain_order=True).agg(pl.col('prediction'), pl.col('article_ids_inview')) \
            .with_columns(pl.col('prediction').list.eval(pl.element().rank(descending=True)).cast(pl.List(pl.Int16)))

        logging.info('Debugging predictions')
        logging.info(behaviors.filter(pl.col('impression_id') == max_impression).select(
            ['impression_id', 'article_ids_inview']).explode('article_ids_inview'))
        logging.info(evaluation_ds.filter(pl.col('impression_id') == max_impression).select(
            ['impression_id', 'article', 'prediction']))
        logging.info(ordered_predictions.filter(pl.col('impression_id') == max_impression)
                     .select(['impression_id', 'article_ids_inview', 'prediction'])
                     .explode(['article_ids_inview', 'prediction']))

        logging.info(f'Saving Results at: {path}')
        write_submission_file(ordered_predictions['impression_id'].to_list(),
                              ordered_predictions['prediction'].to_list(),
                              path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Training script for catboost")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the preprocessed dataset is placed")
    parser.add_argument("-model_path", default=None, type=str, required=True,
                        help="File path where the model is placed")
    parser.add_argument("--submit", action='store_true', default=False,
                        help='Whether to save the predictions or not')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Whether to evaluate the predictions or not')
    parser.add_argument("-behaviors_path", default=None,
                        help="The file path of the reference behaviors ordering. Mandatory to save predictions")
    parser.add_argument('-batch_size', default=None, type=int, required=False,
                        help='If passed, it will predict in batches to reduce the memory usage')
    parser.add_argument('-ranker', default=False, type=bool, required=False,
                        help='Flag to specify if the model is a ranker')

    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    MODEL_PATH = args.model_path
    SAVE_INFERENCE = args.submit
    EVAL = args.eval
    BEHAVIORS_PATH = args.behaviors_path
    BATCH_SIZE = args.batch_size
    RANKER = args.ranker

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'Inference_Test_{timestamp}' if not EVAL else f'Inference_Validation_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)

    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w",
                        format=LOGGING_FORMATTER, level=logging.INFO, force=True)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)

    main(DATASET_DIR, MODEL_PATH, SAVE_INFERENCE,
         EVAL, BEHAVIORS_PATH, output_dir, BATCH_SIZE, RANKER)
