from polimi.utils._catboost import subsample_dataset

if __name__ == '__main__':
    subsample_dataset('/home/ubuntu/dataset/ebnerd_small/train/behaviors.parquet', 
                      '/home/ubuntu/experiments/preprocessing_train_2024-05-20_14-40-50/train_ds.parquet',
                      '/home/ubuntu/experiments/subsample_train_small_click/train_ds.parquet',)