import polars as pl
import numpy as np
from ebrec.utils._python import write_submission_file
from pathlib import Path
import os

ORIGINAL_PATH = '/home/ubuntu/dataset/ebnerd_testset/test/behaviors.parquet'
SUBMISSION = '/home/ubuntu/experiments/Inference_Test_2024-05-05_12-12-59/predictions.parquet'
OUT_PATH = '/home/ubuntu/tmp'

if __name__ == '__main__':
    sub = pl.read_parquet(SUBMISSION)

    behaviors = pl.read_parquet(ORIGINAL_PATH, columns=[
        'impression_id', 'article_ids_inview', 'user_id'])

    ordered_predictions = behaviors.explode('article_ids_inview').with_row_index() \
        .join(sub, left_on=['impression_id', 'article_ids_inview', 'user_id'],
              right_on=['impression_id', 'article', 'user_id'], how='left') \
        .sort('index').group_by(['impression_id', 'user_id'], maintain_order=True).agg(pl.col('prediction'), pl.col('article_ids_inview')) \
        .with_columns(pl.col('prediction').list.eval(pl.element().rank(descending=True)).cast(pl.List(pl.Int16)))

    bey_len = len(ordered_predictions.filter(pl.col('impression_id')
                  == 0).select(pl.col('prediction')).limit(1).item())
    bey_prediction = np.arange(1, bey_len + 1, dtype=np.int16)

    bey_num = ordered_predictions.filter(pl.col('impression_id') == 0).shape[0]
    
    new_col = pl.Series('prediction', [bey_prediction for i in range(bey_num)])
    
    modified = pl.concat([ordered_predictions.filter(pl.col('impression_id') != 0),ordered_predictions.filter(pl.col('impression_id') == 0).with_columns(new_col)], how='vertical_relaxed')
    
    assert ordered_predictions.filter(pl.col('impression_id')!= 0).equals(modified.filter(pl.col('impression_id')!= 0))
    assert ordered_predictions.select(['impression_id', 'user_id']).equals(modified.select(['impression_id', 'user_id']))
    assert ordered_predictions.shape == modified.shape
    print(behaviors.shape)
    print(modified.shape)
    assert modified.shape == behaviors.shape 

    print(modified)
    
    write_submission_file(modified['impression_id'].to_list(),
                          modified['prediction'].to_list(),
                          path = Path(os.path.join(OUT_PATH, 'predictions.txt')))
