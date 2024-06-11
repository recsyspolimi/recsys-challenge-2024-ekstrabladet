import os
import logging
from datetime import datetime
import argparse
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing_extensions import List, Tuple, Dict, Type, TypeVar
import polars as pl
import gc
import tensorflow as tf
import joblib
import torch
import tensorflow as tf
from torch.optim import AdamW
from deepctr_torch.callbacks import EarlyStopping

from polimi.utils.tf_models import *
from polimi.utils.tf_models.utils import *
import sys
sys.path.append('/home/ubuntu/RecSysChallenge2024/src')

from ebrec.evaluation.metrics_protocols import *
from polimi.utils.tf_models import *
from polimi.utils.tf_models.utils import *
from fastauc.fastauc.fast_auc import CppAuc

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
T = TypeVar('T', bound=TabularNNModel)

from sklearn.preprocessing import PowerTransformer, OrdinalEncoder
from deepctr_torch.models import AutoINT
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names

features_to_drop=[
    "Category_auto_Pct","Category_bibliotek_Pct",
    "Category_biler_Pct","Category_dagsorden_Pct",
    "Category_ferie_Pct","Category_forbrug_Pct",
    "Category_haandvaerkeren_Pct","Category_horoskoper_Pct",
    "Category_incoming_Pct","Category_krimi_Pct",
    "Category_musik_Pct","Category_nationen_Pct",
    "Category_nyheder_Pct","Category_om_ekstra_bladet_Pct",
    "Category_opinionen_Pct","Category_penge_Pct",
    "Category_plus_Pct","Category_podcast_Pct",
    "Category_services_Pct","Category_video_Pct",
    "Category_vin_Pct","EVENTPct","Entity_EVENT_Present",
    "Entity_PER_Present","LOCPct","MISCPct","MaxReadTime",
    "MaxScrollPercentage","MedianReadTime","MedianScrollPercentage",
    "MostFrequentCategory","MostFrequentHour","MostFrequentWeekday",
    "NegativePct","NeutralPct","NumArticlesHistory","NumberDifferentCategories",
    "ORGPct","PERPct","PRODPct","PctCategoryMatches","PctNotDefaultArticles",
    "PctStrongNegative","PctStrongNeutral","PctStrongPositive","PositivePct",
    "TotalReadTime","age","clicked_count_l_inf_impression",
    "endorsement_20h_articleuser_l_inf_articleuser",
    "endorsement_20h_articleuser_macd","endorsement_10h_leak_macd",
    "entropy_impression_mean_JS","entropy_impression_std_JS",
    "entropy_impression_topics_cosine",
    "entropy_impression_trendiness_score_3d_leak",
    "entropy_impression_trendiness_score_category",
    "gender","is_already_seen_article","is_inside_window_1",
    "kurtosis_impression_article_delay_hours",
    "kurtosis_impression_endorsement_10h",
    "kurtosis_impression_endorsement_10h_leak",
    "kurtosis_impression_inview_count","kurtosis_impression_mean_JS",
    "kurtosis_impression_mean_topic_model_cosine","kurtosis_impression_std_JS",
    "kurtosis_impression_topics_cosine","kurtosis_impression_total_read_time",
    "kurtosis_impression_trendiness_score_3d",
    "kurtosis_impression_trendiness_score_3d_leak",
    "kurtosis_impression_trendiness_score_5d",
    "kurtosis_impression_trendiness_score_category","last_session_duration",
    "last_session_time_hour_diff","lda_0_history_mean","lda_0_history_weighted_mean",
    "lda_1_history_mean","lda_1_history_weighted_mean",
    "lda_2_history_mean","lda_2_history_weighted_mean","lda_3_history_mean",
    "lda_3_history_weighted_mean","lda_4_history_mean","lda_4_history_weighted_mean",
    "max_ner_item_knn_scores","max_ner_svd_scores","max_topic_model_cosine",
    "mean_ner_item_knn_scores","mean_ner_svd_scores","mean_prev_sessions_duration",
    "mean_topic_model_cosine","mean_topic_model_cosine_l_inf_article",
    "mean_topic_model_cosine_l_inf_impression",
    "mean_topic_model_cosine_l_inf_user","mean_topic_model_cosine_minus_median_impression",
    "mean_topic_model_cosine_rank_impression","mean_user_trendiness_score","min_JS",
    "min_topic_model_cosine","num_topics","postcode","sentiment_label_diversity_impression",
    "skew_impression_inview_count","skew_impression_mean_topic_model_cosine","skew_impression_std_JS",
    "skew_impression_topics_cosine","skew_impression_total_pageviews/inviews",
    "skew_impression_trendiness_score_3d","skew_impression_trendiness_score_3d_leak",
    "skew_impression_trendiness_score_5d","skew_impression_trendiness_score_category",
    "std_impression_mean_JS","std_impression_mean_topic_model_cosine","std_impression_std_JS",
    "std_impression_topics_cosine","std_impression_trendiness_score_3d","std_impression_trendiness_score_3d_leak",
    "std_impression_trendiness_score_5d","std_impression_trendiness_score_category","std_topic_model_cosine",
    "topics_cosine_l_inf_user","total_ner_item_knn_scores","total_ner_svd_scores",
    "total_pageviews_minus_median_impression","total_pageviews_rank_impression",
    "total_read_time_l_inf_impression","total_read_time_minus_median_impression",
    "total_read_time_rank_impression","trendiness_score_1d/5d","weighted_mean_topic_model_cosine",
    "window_0_history_length","window_1_history_length","window_2_history_length",
    "window_3_history_length","window_topics_score",
]

def create_layer_tuple(num_layers,start):
    start_value = start
    layer_values = [start_value]
    for _ in range(num_layers - 1):
        start_value //= 2  # Update start_value by dividing by 2
        layer_values.append(start_value)
    return tuple(layer_values)

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
T = TypeVar('T', bound=TabularNNModel)


def main(dataset_path, params_path, output_dir, early_stopping_path, es_patience, transform_path):
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")
    
    train_ds = pl.scan_parquet(os.path.join(dataset_path, 'train_ds.parquet')).collect() #.drop(features_to_drop).collect()
    val_ds = pl.scan_parquet(os.path.join(early_stopping_path, 'validation_ds.parquet')).collect() if early_stopping_path else None #.drop(features_to_drop).collect() if early_stopping_path else None
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Data info: {data_info}')
        
    if 'postcode' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('postcode').fill_null(5))
        if early_stopping_path:
            val_ds = val_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('article_type').fill_null('article_default'))
        if early_stopping_path:
            val_ds = val_ds.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in train_ds.columns:
        train_ds = train_ds.drop(['impression_time'])
        
    categorical_columns = [c for c in data_info['categorical_columns']] # if c not in features_to_drop]
    
    train_ds = train_ds.drop(['impression_id', 'article', 'user_id']).to_pandas()
    train_ds[categorical_columns] = train_ds[categorical_columns].astype(str)

    X = train_ds.drop(columns=['target'])
    y = train_ds['target']
    
    numerical_columns = [c for c in X.columns if c not in categorical_columns]
    
    categories = []
    vocabulary_sizes = {}
    for cat_col in categorical_columns:
        categories_train = list(X[cat_col].unique())
        categories_train.append('Unknown')
        vocabulary_sizes[cat_col] = len(categories_train)
        categories.append(categories_train)
        
    categorical_encoder = OrdinalEncoder(categories=categories)
    X[categorical_columns] = categorical_encoder.fit_transform(X[categorical_columns], y).astype(np.int16)

    train_fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=vocabulary_sizes[feat],embedding_dim=64)
                for i,feat in enumerate(categorical_columns)] + [DenseFeat(feat, 1,)
                for feat in numerical_columns]
                
    train_feature_names = get_feature_names(train_fixlen_feature_columns)
    train_model_input = {name: X[name].values for name in train_feature_names}
    
    infos = {
        'numerical_columns': numerical_columns,
        'categorical_columns': categorical_columns,
        'categories': {
            col: categories[i] for i, col in enumerate(categorical_columns)
        }
    }
    with open(os.path.join(output_dir, 'info.json'), 'w') as info_file:
        json.dump(infos, info_file)
        
    joblib.dump(categorical_encoder, os.path.join(output_dir, 'categorical_encoder.joblib'))
    
    del train_ds
    gc.collect()
    
    if early_stopping_path:
        val_ds = val_ds.drop(['impression_id', 'article', 'user_id']).to_pandas()
        val_ds[categorical_columns] = val_ds[categorical_columns].astype(str)
        if transform_path:
            xformer = joblib.load(transform_path)
            val_ds[xformer.feature_names_in_] = xformer.transform(val_ds[xformer.feature_names_in_].replace(
                [-np.inf, np.inf], np.nan).fillna(0)).astype(np.float32)
        for i, cat_col in enumerate(categorical_columns):
            categories_val = list(val_ds[cat_col].unique())
            unknown_categories = [x for x in categories_val if x not in categories[i]]
            val_ds[cat_col] = val_ds[cat_col].replace(list(unknown_categories), 'Unknown')
        val_ds[categorical_columns] = categorical_encoder.transform(val_ds[categorical_columns]).astype(np.int16)
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=vocabulary_sizes[feat],embedding_dim=64)
                    for i,feat in enumerate(categorical_columns)] + [DenseFeat(feat, 1,)
                    for feat in numerical_columns]
        val_feature_names = get_feature_names(fixlen_feature_columns)
        test = {name:val_ds[name].values for name in val_feature_names}
        validation_data = (test, val_ds['target'].values)
        del val_ds
        gc.collect()
    else:
        validation_data = None
    
    logging.info(f'Features ({len(X.columns)}): {np.array(list(X.columns))}')
    logging.info(f'Categorical features: {np.array(data_info["categorical_columns"])}')

    logging.info(f'Starting to train the DeepFM model')
    params = {
        "dnn_dropout": 0.21551012156663485,
        "l2_reg_embedding": 0.000023594595753905804,
        "l2_reg_dnn": 0.000019077766264818277,
        "att_head_num ": 4,
        "dnn_use_bn": True,
        "att_layer_num ": 8,
        "trials": 16,
        "num_layers": 3,
        "start": 256,
        "att_res": True,
        "lr": 0.001576607039337392
  }
    dnn_hidden_units = create_layer_tuple(params['num_layers'],params['start'])

    # best epoch: 3
    model = AutoINT(train_fixlen_feature_columns,train_fixlen_feature_columns,dnn_dropout=params['dnn_dropout'],l2_reg_embedding=params['l2_reg_embedding'],att_head_num=params['att_head_num'],att_layer_num=params["att_layer_num "],att_res=True,dnn_use_bn = True,l2_reg_dnn=params['l2_reg_dnn'],dnn_hidden_units=dnn_hidden_units,dnn_activation='relu',task='binary')
    model.compile(AdamW(model.parameters(), params['lr']), "binary_crossentropy", metrics=['auc'], )
    es = EarlyStopping(monitor='val_auc', min_delta=0, verbose=2, patience=es_patience, mode='max')
    model.fit(train_model_input, y.values, batch_size=1024, epochs=10, validation_data=validation_data, callbacks=[es])
    
    logging.info(f'Model fitted. Saving the model and the feature importances at: {output_dir}')
    torch.save(model.state_dict(), os.path.join(output_dir, 'DeepFM_weights.h5'))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for DeepFM")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the preprocessed dataset is placed")
    parser.add_argument("-early_stopping_path", default=None, type=str,
                        help="Directory where the early stopping dataset is placed")
    parser.add_argument("-params_file", default=None, type=str, required=True,
                        help="File path where the catboost hyperparameters are placed")
    parser.add_argument("-model_name", default=None, type=str,
                        help="The name of the model")
    parser.add_argument("-early_stopping_patience", default=4, type=int,
                        help="The patience for early stopping")
    parser.add_argument("-numerical_transformer_es", default=None, type=str,
                        help="The path for numerical transformer to transform the early stopping data if needed")
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    HYPERPARAMS_PATH = args.params_file
    EARLY_STOPPING_PATH = args.early_stopping_path
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    TRANSFORM_PATH = args.numerical_transformer_es
    model_name = args.model_name
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if model_name is None:
        model_name = f'AutoINT_Training_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_DIR, HYPERPARAMS_PATH, output_dir, EARLY_STOPPING_PATH, EARLY_STOPPING_PATIENCE, TRANSFORM_PATH)



