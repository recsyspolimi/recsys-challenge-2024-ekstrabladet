from polimi.utils._catboost import subsample_dataset

if __name__ == '__main__':
    subsample_dataset('/home/ubuntu/dataset/ebnerd_small/train/behaviors.parquet', 
                      '/home/ubuntu/experiments/preprocessing_train_small_127/train_ds.parquet',
                      '/home/ubuntu/experiments/subsample_train_small_127/train_ds.parquet')