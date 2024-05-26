import logging
from lightgbm import LGBMClassifier, LGBMRanker
import polars as pl
from catboost import CatBoostClassifier, CatBoostRanker
from xgboost import XGBClassifier, XGBRanker
from ebrec.evaluation.metrics_protocols import *
from polimi.utils.model_wrappers import FastRGFClassifierWrapper


def get_model_class(name: str = 'catboost', ranker: bool = False):
    if name == 'catboost':
        return CatBoostClassifier if not ranker else CatBoostRanker
    if name == 'lgbm':
        return LGBMClassifier if not ranker else LGBMRanker
    elif name == 'xgb':
        return XGBClassifier if not ranker else XGBRanker
    elif name == 'fast_rgf':
        if ranker:
            logging.log('RGF do not support ranking problems, param is_rank will be ignored')
        return FastRGFClassifierWrapper

def train_predict_tree_model(train_ds, val_ds, data_info, model_class, ranker, params):
    
    if 'postcode' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in train_ds.columns:
        train_ds = train_ds.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in train_ds.columns:
        train_ds = train_ds.drop(['impression_time'])
    
    if ranker:
        train_ds = train_ds.sort(by='impression_id')
        groups = train_ds.select('impression_id').to_numpy().flatten()
    train_ds = train_ds.to_pandas()
    group_ids = train_ds['impression_id'].to_frame()
    train_ds = train_ds.drop(columns=['impression_id', 'article', 'user_id'])
    train_ds[data_info['categorical_columns']] = train_ds[data_info['categorical_columns']].astype('category')
    
    X_train = train_ds.drop(columns=['target'])
    y_train = train_ds['target']
    
    model_class = get_model_class(model_class, ranker)
    
    
    if model_class == CatBoostRanker:
        params['cat_features'] =  data_info['categorical_columns']
        model = model_class(**params)
        model.fit(X_train, y_train, group_id=groups, verbose=50)
        
    elif model_class in [XGBRanker, LGBMRanker]:
        model = model_class(**params)
        model.fit(X_train, y_train, group=group_ids.groupby('impression_id')['impression_id'].count().values)
     
    elif model_class == LGBMClassifier:
        model = model_class(**params)
        model.fit(X_train, y_train)
        
    else:
        model.fit(X_train, y_train, verbose=50)
        
        
    if 'postcode' in val_ds.columns:
        val_ds = val_ds.with_columns(pl.col('postcode').fill_null(5))
    if 'article_type' in val_ds.columns:
        val_ds = val_ds.with_columns(pl.col('article_type').fill_null('article_default'))
    if 'impression_time' in val_ds.columns:
        val_ds = val_ds.drop(['impression_time'])
    
    val_ds = val_ds.to_pandas()
    val_ds[data_info['categorical_columns']] = val_ds[data_info['categorical_columns']].astype('category')

    X_val = val_ds[X_train.columns]
    evaluation_ds = pl.from_pandas(val_ds[['impression_id', 'article', 'target']])
    
    if model_class in [CatBoostRanker, XGBRanker, LGBMRanker]:
        prediction_ds = evaluation_ds.with_columns(pl.Series(model.predict(X_val)).alias('prediction')) \
                .group_by('impression_id').agg(pl.col('target'), pl.col('prediction'))
    else:
        prediction_ds = evaluation_ds.with_columns(pl.Series(model.predict_proba(X_val)[:, 1]).alias('prediction')) \
                .group_by('impression_id').agg(pl.col('target'), pl.col('prediction'))
                
    return prediction_ds