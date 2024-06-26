import argparse
import polars as pl
import numpy as np
from ebrec.utils._python import write_submission_file
from pathlib import Path
from polars import testing
import os
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Training script for catboost")
    parser.add_argument("-prediction_1", default=None, type=str,
                        help="The directory for the predictions of the first model")
    parser.add_argument("-prediction_2", default=None, type=str, required=True,
                        help="The directory for the predictions of the second model")
    parser.add_argument("-original_path", default=None, type=str, required=True,
                        help="The path of the original testset behavior dataframe")
    parser.add_argument("-output_dir", default=None, type=str, required=True,
                        help="File path where the submission will be placed")

    args = parser.parse_args()
    predictions_1_path = args.prediction_1
    predictions_2_path = args.prediction_2
    ORIGINAL_PATH = args.original_path
    OUTPUT_DIR = args.output_dir
    
    os.makedirs(OUTPUT_DIR)
    
    pred_1 = pl.read_parquet(predictions_1_path).rename({'prediction': 'prediction_1'})
    pred_2 = pl.read_parquet(predictions_2_path).rename({'prediction': 'prediction_2'})
    pred_1 = pred_1.join(pred_2, on=['impression_id','article','user_id'], how='left')\
        .with_columns(
            (0.5 * pl.col('prediction_1') + 0.5 * pl.col('prediction_2')).alias('prediction')
        )
        
    sub = pred_1

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
    
    testing.assert_frame_equal(ordered_predictions.filter(pl.col('impression_id')!= 0), modified.filter(pl.col('impression_id')!= 0))
    testing.assert_frame_equal(behaviors.select(['impression_id', 'user_id']), modified.select(['impression_id', 'user_id']))
    
    assert ordered_predictions.filter(pl.col('impression_id')!= 0).equals(modified.filter(pl.col('impression_id')!= 0))
    assert ordered_predictions.select(['impression_id', 'user_id']).equals(modified.select(['impression_id', 'user_id']))
    assert ordered_predictions.shape == modified.shape
    print(behaviors.shape)
    print(modified.shape)
    print(modified)
    
    write_submission_file(modified['impression_id'].to_list(),
                          modified['prediction'].to_list(),
                          path = Path(os.path.join(OUTPUT_DIR, 'predictions.txt')))
