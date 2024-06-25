import polars as pl
                
if __name__ == '__main__':
    pred_1 = pl.read_parquet('/home/ubuntu/experiments/Inference_Test_2024-06-17_09-59-45/predictions.parquet').rename({'prediction': 'prediction_1'})
    pred_2 = pl.read_parquet('/home/ubuntu/experiments/Inference_Test_postprocessing/predictions.parquet').rename({'prediction': 'prediction_2'})
    pred_1 = pred_1.join(pred_2, on=['impression_id','article','user_id'], how='left')\
        .with_columns(
            (0.5 * pl.col('prediction_1') + 0.5 * pl.col('prediction_2')).alias('prediction')
        )
    pred_1.select(['impression_id','user_id','article','prediction']).write_parquet('/home/ubuntu/experiments/Inference_Stacking_Hybrid/predictions.parquet')