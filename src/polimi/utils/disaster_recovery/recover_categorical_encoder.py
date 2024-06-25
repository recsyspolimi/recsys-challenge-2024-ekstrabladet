import polars as pl
import os
import json
import joblib
from sklearn.preprocessing import OrdinalEncoder
import logging

if __name__ == '__main__':
    train_ds = pl.read_parquet(os.path.join('/mnt/ebs_volume/experiments/subsample_train_new', 'train_ds.parquet'))
    with open(os.path.join('/mnt/ebs_volume/experiments/subsample_train_new', 'data_info.json')) as data_info_file:
        data_info = json.load(data_info_file)
        
    print('Dataset Read')
    
    if 'postcode' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in train_ds.columns:
        train_ds = train_ds.drop(['impression_time'])
    
    train_ds = train_ds.drop(['impression_id', 'article', 'user_id']).to_pandas()
    
    categories = []
    for f in data_info['categorical_columns']:
        train_ds[f] = train_ds[f].astype(str).fillna('NA')
        categories_train = list(train_ds[f].unique())
        if 'NA' not in categories_train:
            categories_train.append('NA')
        categories.append(categories_train)
        
    print('Fitting categorical encoder')
    
    encoder = OrdinalEncoder(categories=categories)
    encoder.fit(train_ds[data_info['categorical_columns']], train_ds['target'])
    
    print('Saving categorical encoder')
    joblib.dump(encoder, os.path.join('/mnt/ebs_volume/models/mlp_new_features_trial_66', 'categorical_encoder.joblib'))