import polars as pl
from polars import testing

if __name__ == "__main__":
    
    df = pl.read_parquet('/mnt/ebs_volume/preprocessing_train_2024-04-28_14-28-17/train_ds.parquet')
    original = pl.read_parquet('/home/ubuntu/dataset/ebnerd_large/train/behaviors.parquet').select(['impression_id', 'user_id', 'article_ids_inview','impression_time'])\
        .explode('article_ids_inview').rename({'article_ids_inview' : 'article'})
        
    df_2 = df.join(original, on = ['impression_id', 'user_id', 'article'])
    
    testing.assert_frame_equal(df_2.drop('impression_time'), df, check_column_order=False, check_row_order=False)
    
    df_2.write_parquet('/mnt/ebs_volume/preprocessing_train_2024-04-28_14-28-17/train_ds.parquet')