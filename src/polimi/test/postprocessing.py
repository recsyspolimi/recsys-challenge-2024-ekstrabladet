import polars as pl


if __name__ == '__main__':
    predictions = pl.scan_parquet('/home/ubuntu/experiments/Inference_Stacking_Hybrid/predictions.parquet').collect()
    predictions_columns = predictions.columns
    print(predictions)
    df = pl.scan_parquet('/mnt/ebs_volume/stacking/dataset/test_ds.parquet')\
        .select(['impression_id','user_id','article','article_delay_days',
                                   'article_delay_hours','user_mean_delay_hours',
                                   'normalized_endorsement_10h','category']).collect()
    print(df)
    predictions = predictions.join(df, on=['impression_id', 'user_id','article'], how='left')
    predictions = predictions.with_columns(
                pl.when(pl.col('article_delay_days') > 3)\
                    .then(pl.col('prediction') * 0.5)\
                    .otherwise(pl.col('prediction'))
            ).with_columns(
                pl.when(pl.col('article_delay_hours') > pl.col('user_mean_delay_hours')*3)\
                    .then(pl.col('prediction') * 1.2)\
                    .otherwise(pl.col('prediction'))
            ).with_columns(
                pl.when(pl.col('normalized_endorsement_10h') == pl.col('normalized_endorsement_10h').max().over('impression_id'))
                    .then(pl.col('prediction') * 0.95)\
                    .otherwise(pl.col('prediction'))
            ).with_columns(
                pl.when(pl.col('category') == 414)
                        .then(pl.col('prediction') * 0.80)\
                        .otherwise(pl.col('prediction'))
            ).with_columns(
                pl.when(pl.col('category') == 512)
                        .then(pl.col('prediction') * 0.90)\
                        .otherwise(pl.col('prediction'))
            ).with_columns(
                pl.when(pl.col('category') == 561)
                        .then(pl.col('prediction') * 1.3)\
                        .otherwise(pl.col('prediction'))
            ).with_columns(
                pl.when(pl.col('category') == 2077)
                        .then(pl.col('prediction') * 1.1)\
                        .otherwise(pl.col('prediction'))
            )
    print(predictions)
    predictions.select(predictions_columns).write_parquet('/home/ubuntu/experiments/Inference_Stacking_Hybrid/predictions.parquet')
