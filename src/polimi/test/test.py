import polars as pl
from polimi.utils._catboost import reduce_polars_df_memory_size
from polars import testing

RANKER = True
TRAIN_VAL = True
dataset_path = '/home/ubuntu/experiments/preprocessing_train_new'
validation_path = '/home/ubuntu/experiments/preprocessing_validation_new'
batch_split_directory = '/home/ubuntu/experiments/batches_train_val_new/batches'

model_path = '/home/ubuntu/experiments/batches_train_val_new/models'


EVAL = False
SAVE_PREDICTIONS = False
N_BATCH = 10
BATCHES = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def load_batch(dataset_path, batch_split_directory, batch_index):

    train_ds = pl.scan_parquet(dataset_path + '/train_ds.parquet')

    if TRAIN_VAL:
        val_ds = pl.scan_parquet(validation_path + '/validation_ds.parquet')
    batch = pl.scan_parquet(batch_split_directory +
                            f'/batch_{batch_index}.parquet').collect()

    subsampled_train = train_ds.filter(pl.col('impression_id').is_in(
        batch.select('impression_id'))).collect()
    columns = subsampled_train.columns

    if TRAIN_VAL:
        subsampled_val = val_ds.filter(pl.col('impression_id').is_in(
            batch.select('impression_id'))).select(columns).collect()
        subsampled_train = pl.concat(
            [subsampled_train, subsampled_val], how='vertical_relaxed')

    if 'postcode' in subsampled_train.columns:
        subsampled_train = subsampled_train.with_columns(
            pl.col('postcode').fill_null(5))
    if 'article_type' in subsampled_train.columns:
        subsampled_train = subsampled_train.with_columns(
            pl.col('article_type').fill_null('article_default'))

    subsampled_train = subsampled_train.sort(by='impression_id')
    groups = subsampled_train.select('impression_id').to_numpy().flatten()
    subsampled_train = subsampled_train.drop(
        ['impression_id', 'article', 'user_id', 'impression_time']).to_pandas()

    X = subsampled_train.drop(columns=['target'])
    y = subsampled_train['target']
    print(X.shape)

    if 'impression_time' in X:
        X = X.drop(['impression_time'])

    del train_ds, batch, subsampled_train
    gc.collect()

    return X, y, groups


if __name__=='__main__':
    df = pl.read_parquet('/home/ubuntu/experiments/batches_train_val_new/batches/batch_1.parquet').unique('impression_id')
    print(df.unique('impression_id').shape[0])
    print(df.shape[0])