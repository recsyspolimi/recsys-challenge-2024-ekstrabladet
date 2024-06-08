import polars as pl

from polars import testing

leaking_features = [    
    'endorsement_10h_leak_rank_impression',
    'trendiness_score_3d_leak_rank_impression',
    'kurtosis_impression_endorsement_10h_leak',
    'kurtosis_impression_trendiness_score_3d_leak',
    'entropy_impression_endorsement_10h_leak',
    'entropy_impression_trendiness_score_3d_leak',
    'skew_impression_endorsement_10h_leak',
    'skew_impression_trendiness_score_3d_leak',
    'std_impression_endorsement_10h_leak',
    'std_impression_trendiness_score_3d_leak',
    'endorsement_10h_leak_minus_median_impression',
    'trendiness_score_3d_leak_minus_median_impression',
    'endorsement_10h_leak_l_inf_impression',
    'trendiness_score_3d_leak_l_inf_impression',
    'trendiness_score_3d_leak',
    'normalized_endorsement_10h_leak_rolling_max_ratio',
    'endorsement_10h_leak',
    'normalized_endorsement_10h_leak',
    'endorsement_10h_leak_diff_rolling', 
    'endorsement_10h_leak_macd',
    'endorsement_10h_leak_quantile_norm'   
]
if __name__ == '__main__':
    original = pl.read_parquet('/mnt/ebs_volume/experiments/subsample_train_new/train_ds.parquet')
    print(original.drop(leaking_features).head())
    print(original.drop(leaking_features).columns)
    
    