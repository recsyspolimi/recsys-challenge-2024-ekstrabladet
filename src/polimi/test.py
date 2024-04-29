import polars as pl
from polimi.utils._catboost import _preprocessing_article_endorsement_feature
from tqdm import tqdm

if __name__ == '__main__': 
    print(pl.read_parquet('/home/ubuntu/experiments/preprocessing_train_2024-04-22_13-28-45/train_ds.parquet').head())