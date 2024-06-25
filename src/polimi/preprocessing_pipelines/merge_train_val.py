import polars as pl
import shutil


if __name__ == '__main__':
    shutil.copyfile(
        '/mnt/ebs_volume/experiments/subsample_train_new_with_recsys/data_info.json',
        '/home/ubuntu/experiments/subsample_complete_new_with_recsys/data_info.json'
    )
    pl.concat([
        pl.read_parquet('/mnt/ebs_volume/experiments/subsample_train_new_with_recsys/train_ds.parquet'),
        pl.read_parquet('/home/ubuntu/experiments/subsample_val_new_with_recsys/train_ds.parquet'),
    ], how='diagonal_relaxed').write_parquet('/home/ubuntu/experiments/subsample_complete_new_with_recsys/train_ds.parquet')