from lightgbm import LGBMClassifier
import polars as pl
import json
import gc
from tqdm import tqdm

from polimi.utils._catboost import subsample_dataset
from ebrec.utils._behaviors import sampling_strategy_wu2019
import numpy as np
from fastauc.fastauc.fast_auc import fast_numba_auc

NUM_MODELS = 10
NPRATIO = 2

if __name__ == '__main__':

    train_ds = pl.read_parquet('/home/ubuntu/dset_complete/train_ds.parquet')
    val_ds = pl.read_parquet('/home/ubuntu/dset_complete/validation_ds.parquet')
    behaviors = pl.read_parquet('/home/ubuntu/dataset/ebnerd_small/train/behaviors.parquet').select(['impression_id', 'user_id', 'article_ids_inview', 'article_ids_clicked'])

    with open('/home/ubuntu/dset_complete/data_info.json') as info_file:
        data_info = json.load(info_file)
        
    with open('/home/ubuntu/RecSysChallenge2024/configuration_files/lightgbm_new_noK_trial_289.json') as params_file:
        params = json.load(params_file)
        
    evaluation_ds = val_ds.select(['impression_id', 'user_id', 'article', 'target'])
    val_ds = val_ds.drop(['impression_id', 'article', 'user_id']).to_pandas()
    val_ds[data_info['categorical_columns']] = val_ds[data_info['categorical_columns']].astype('category')

    X_val = val_ds.drop(columns=['target'])

    del val_ds
    gc.collect()
        
    bagging_predictions = []
    for i in range(NUM_MODELS):
        
        behaviors_subsample = behaviors.pipe(
                sampling_strategy_wu2019, npratio=NPRATIO, shuffle=False, with_replacement=True, seed=42+i
            ).drop('article_ids_clicked').explode('article_ids_inview').rename({'article_ids_inview' : 'article'}) \
            .with_columns(pl.col('user_id').cast(pl.UInt32), pl.col('article').cast(pl.Int32))
            
        train_ds_subsample = behaviors_subsample.join(train_ds, on=['impression_id', 'user_id', 'article'], how='left')
        
        train_ds_subsample = train_ds_subsample.drop(['impression_id', 'article', 'user_id']).to_pandas()
        train_ds_subsample[data_info['categorical_columns']] = train_ds_subsample[data_info['categorical_columns']].astype('category')
        
        X = train_ds_subsample.drop(columns=['target'])
        y = train_ds_subsample['target']
        
        del train_ds_subsample
        gc.collect()
        
        model = LGBMClassifier(**params, verbosity=-1)
        model.fit(X, y)
        
        predictions = model.predict_proba(X_val[X.columns])[:, 1]
        evaluation_ds = evaluation_ds.with_columns(pl.Series(predictions).alias('prediction'))
        evaluation_ds_grouped = evaluation_ds.group_by('impression_id').agg(pl.col('target'), pl.col('prediction'))
        
        auc = np.mean(
            [fast_numba_auc(np.array(y_t).astype(bool), np.array(y_s).astype(np.float32)) 
                for y_t, y_s in zip(evaluation_ds_grouped['target'].to_list(), 
                                    evaluation_ds_grouped['prediction'].to_list())]
        )
        print(f'Iteration {i} auc: {auc}')
        bagging_predictions.append(predictions)
        
    predictions_mean = np.mean(bagging_predictions)

    evaluation_ds = evaluation_ds.with_columns(pl.Series(predictions).alias('prediction'))
    evaluation_ds_grouped = evaluation_ds.group_by('impression_id').agg(pl.col('target'), pl.col('prediction'))

    auc = np.mean(
        [fast_numba_auc(np.array(y_t).astype(bool), np.array(y_s).astype(np.float32)) 
            for y_t, y_s in zip(evaluation_ds_grouped['target'].to_list(), 
                                evaluation_ds_grouped['prediction'].to_list())]
    )
    print('Bagging AUC: {auc}')