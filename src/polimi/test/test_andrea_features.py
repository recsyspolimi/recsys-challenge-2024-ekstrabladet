from os import system, getpid, walk
from psutil import Process
import polars as pl
import time
from src.polimi.utils._polars import reduce_polars_df_memory_size
from tqdm import tqdm


def GetMemUsage():
    pid = getpid()
    py = Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return f"RAM memory GB usage = {memory_use :.4}"


def inflate_df(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    start_mem = df.estimated_size('mb')
    if verbose:
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        # Integer types
        if col_type in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
            df = df.with_columns(pl.col(col).cast(pl.Int32))
        elif col_type in [pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
            df = df.with_columns(pl.col(col).cast(pl.UInt32))
        # Float types
        elif col_type == pl.Float64:
            df = df.with_columns(pl.col(col).cast(pl.Float32))
        # List types
        elif col_type in [pl.List(pl.Int16), pl.List(pl.Int32), pl.List(pl.Int64)]:
            df = df.with_columns(pl.col(col).cast(pl.List(pl.Int32)))
        elif col_type in [pl.List(pl.UInt16), pl.List(pl.UInt32), pl.List(pl.UInt64)]:
            df.with_columns(pl.col(col).cast(pl.List(pl.UInt32)))
    return df


def add_emotions_scores(df, history):
    
    df = df.lazy()
    emotion_emb = pl.read_parquet("/home/ubuntu/dataset/emotions_embedding.parquet")\
        .with_columns(
            pl.col("emotion_scores").list.to_struct()
        .struct.rename_fields(['emotion_0', 'emotion_1', 'emotion_2', 'emotion_3', 'emotion_4', 'emotion_5']))\
        .unnest("emotion_scores")

    df = df.join(emotion_emb.lazy(), left_on='article',
                        right_on='article_id', how='left')

    print('Processed impressions')

    embedding_len = 6
    embedded_history = pl.concat(
        rows.select(['user_id', 'article_id_fixed']).explode('article_id_fixed').join(
            emotion_emb, left_on='article_id_fixed', right_on='article_id', how='left')
        .group_by('user_id').agg(
            [pl.col(f'emotion_{i}').mean().cast(pl.Float32).alias(f'user_emotion{i}') for i in range(embedding_len)])
        for rows in tqdm(history.iter_slices(20000), total=history.shape[0] // 20000)).lazy()
    
    df = df.join(embedded_history, on = 'user_id')
    return df.sink_parquet('/mnt/ebs_volume/tmp/tmp.parquet')


if __name__ == '__main__':
    print(GetMemUsage())

    # frame = inflate_df(pl.read_parquet('/mnt/ebs_volume/preprocessing_validation_2024-04-29_13-17-47/Sliced_ds/validation_slice_0.parquet'))
    # for i in range(1,99):
    #     frame = frame.vstack(
    #         inflate_df(pl.read_parquet(f'/mnt/ebs_volume/preprocessing_validation_2024-04-29_13-17-47/Sliced_ds/validation_slice_{i}.parquet'))
    #     )
    # frame = reduce_polars_df_memory_size(frame).lazy()

    val_ds = pl.scan_parquet(
        '/mnt/ebs_volume_2/preprocessing_validation_2024-04-29_13-17-47/validation_ds.parquet')
    val_ds = val_ds.collect()
    history_val = pl.read_parquet(
        '/home/ubuntu/dataset/ebnerd_large/validation/history.parquet')
    articles = pl.read_parquet(
        '/home/ubuntu/dataset/ebnerd_large/articles.parquet')
    val_ds_projection = val_ds.select(['user_id','article','impression_id'])
    
    print('Read all the dataframes')

    val_ds = val_ds.lazy()
    emotion_emb = pl.read_parquet("/home/ubuntu/dataset/emotions_embedding.parquet")\
        .with_columns(
            pl.col("emotion_scores").list.to_struct()
        .struct.rename_fields(['emotion_0', 'emotion_1', 'emotion_2', 'emotion_3', 'emotion_4', 'emotion_5']))\
        .unnest("emotion_scores")

    val_ds = val_ds.join(emotion_emb.lazy(), left_on='article',
                        right_on='article_id', how='left')

    print('Processed impressions')

    embedding_len = 6
    embedded_history = pl.concat(
        rows.select(['user_id', 'article_id_fixed']).explode('article_id_fixed').join(
            emotion_emb, left_on='article_id_fixed', right_on='article_id', how='left')
        .group_by('user_id').agg(
            [pl.col(f'emotion_{i}').mean().cast(pl.Float32).alias(f'user_emotion{i}') for i in range(embedding_len)])
        for rows in tqdm(history_val.iter_slices(20000), total=history_val.shape[0] // 20000)).lazy()
    
    val_ds = val_ds.join(embedded_history, on = 'user_id')
    
    print('Processed click_time')
    val_ds = add_emotions_scores(val_ds, history_val)

    print('Processed history')
    # del emotion_emb

    # emb = pl.scan_parquet('/mnt/ebs_volume_2/click_predictors/validation_click_predictor.parquet')
    # emb_col = emb.drop(['user_id','article']).columns
    emb = pl.scan_parquet('/mnt/ebs_volume/click_predictors/validation_click_predictor.parquet')
    emb_col = emb.drop(['user_id','article']).columns
    normalized_emb = pl.concat( rows.join(emb, on=['user_id','article'], how='left').with_columns(
        *[(pl.col(col) / pl.col(col).max().over('user_id')) for col in emb_col],
        *[(pl.col(col) / pl.col(col).max().over('impression_id')).alias(f'imp_norm_{col}') for col in emb_col])
        for rows in tqdm(val_ds_projection.partition_by('user_id'))).lazy()
   
    val_ds = val_ds.join(normalized_emb, on=['user_id','article'], how='left')
    val_ds.sink_parquet('/mnt/ebs_volume/tmp/tmp.parquet')
    # val_ds = val_ds.join(normalized_emb, on=['user_id','article'], how='left')
    # print('Collecting ...')
    # print(GetMemUsage())
