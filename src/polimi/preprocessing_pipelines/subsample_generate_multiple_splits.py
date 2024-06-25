import polars as pl
import os
from ebrec.utils._behaviors import sampling_strategy_wu2019
from tqdm import tqdm

original_datset_path = '/home/ubuntu/dataset/ebnerd_large/validation/behaviors.parquet'
dataset_path = '/mnt/ebs_volume_new/stacking/dataset/train_ds.parquet'
npratio = 2



if __name__ == '__main__':
    starting_dataset =  pl.read_parquet(original_datset_path).select(['impression_id','user_id','article_ids_inview','article_ids_clicked'])
    dataset = pl.read_parquet(dataset_path)
        
    for i in range(10):
        new_path = f'/mnt/ebs_volume_new/stacking/dataset/splits/subsampled_train_{i}'
        os.mkdir(new_path)
        
        behaviors = pl.concat(
            rows.pipe(
                sampling_strategy_wu2019, npratio=npratio, shuffle=False, with_replacement=True, seed=i
            ).explode('article_ids_inview').drop(columns = 'article_ids_clicked').rename({'article_ids_inview' : 'article'})\
            .with_columns(pl.col('user_id').cast(pl.UInt32),
                        pl.col('article').cast(pl.Int32))\
            
            for rows in tqdm(starting_dataset.iter_slices(1000), total=starting_dataset.shape[0] // 1000)
        )
            
        behaviors.join(dataset, on = ['impression_id','user_id','article'], how = 'left').write_parquet(new_path + '/train_ds.parquet')