import polars as pl


if __name__ == '__main__':
    predictions1 = pl.read_parquet('/home/ubuntu/experiments/Inference_Test_2024-05-08_23-43-54/predictions.parquet').rename({'prediction': 'prediction1'})
    predictions2 = pl.read_parquet('/home/ubuntu/experiments/Inference_Test_2024-05-09_13-48-01/predictions.parquet').rename({'prediction': 'prediction2'})

    
    predictions1 = predictions1.join(predictions2, on=['impression_id', 'user_id', 'article'], how='left').with_columns(
        ((pl.col('prediction1') + pl.col('prediction2')) / 2).alias('prediction')
    ).drop(['prediction1', 'prediction2'])
    predictions1.write_parquet('/home/ubuntu/experiments/ensemble_small_demo_lgbm/predictions_ensemble.parquet')