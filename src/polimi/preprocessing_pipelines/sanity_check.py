import polars as pl
from tqdm import tqdm

TYPE = 'testt'
ORIGINAL_PATH = '/home/ubuntu/dataset/ebnerd_testset/test/behaviors.parquet'
PREPROCESSING_PATH = '/home/ubuntu/experiments/inference_test_catboost_new_noK_0.8348/predictions.parquet'
if __name__ == '__main__':
    
    original_ds = pl.read_parquet(ORIGINAL_PATH).select(['user_id', 'impression_id', 'article_ids_inview'])\
        .explode('article_ids_inview').rename({'article_ids_inview': 'article'})
    df = None
    if TYPE != 'test':
        df = pl.read_parquet(PREPROCESSING_PATH).select(['user_id', 'article', 'impression_id'])
           
    else :
        df_slices = []
        for i in tqdm(range(0,101)):
            df_slices.append(pl.read_parquet(PREPROCESSING_PATH + f'/test_slice_{i}.parquet').select(['user_id', 'article', 'impression_id']))
        df = pl.concat(df_slices, how='vertical_relaxed')
    
    print(original_ds.join(df, on= ['user_id', 'article', 'impression_id'], how='anti'))
    assert original_ds.join(df, on= ['user_id', 'article', 'impression_id'], how='anti').shape[0] == 0
