import os
import logging
from datetime import datetime
import argparse
import polars as pl
import json

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"

def main(dataset_path, output_dir):
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
    logging.info(f"Loading the preprocessed dataset from {dataset_path}")
    
    train_ds = pl.read_parquet(os.path.join(dataset_path, 'train_ds.parquet'))
    with open(os.path.join(dataset_path, 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Data info: {data_info}')
    
    train_ds.drop(features_to_drop)
    train_ds.write_parquet(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Just loads the dataset, drops some features and saves the new version")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the dataframe will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the preprocessed dataset is placed")

    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_DIR, output_dir)
