from polimi.utils._catboost import subsample_dataset

if __name__ == '__main__':
    subsample_dataset('/home/ubuntu/dataset/ebnerd_large/train/behaviors.parquet', 
                      '/home/ubuntu/experiments/preprocessing_train_2024-04-28_14-28-17/train_ds.parquet',
                      '/home/ubuntu/experiments/subsample_train_128/train_ds.parquet',)