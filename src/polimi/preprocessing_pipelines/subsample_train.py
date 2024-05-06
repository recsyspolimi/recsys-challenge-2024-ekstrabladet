from polimi.utils._catboost import subsample_dataset

if __name__ == '__main__':
    subsample_dataset('/home/ubuntu/dataset/ebnerd_demo/train/behaviors.parquet', 
                      '/home/ubuntu/experiments/preprocessing_train_2024-05-04_23-08-15/train_ds.parquet',
                      '/home/ubuntu/experiments/subsamoled_train_127_demo/train_ds.parquet',)