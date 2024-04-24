import polars as pl

from polimi.utils._polars import reduce_polars_df_memory_size
from polimi.utils._embeddings import _add_normalized_features_emb

def explode_ds(df):
    return df.select('impression_id','user_id','article_ids_inview').explode('article_ids_inview')\
        .rename({'article_ids_inview':'article'})

if __name__ == "__main__":
       
    embeddings = ['emotions','w2v','contrastive','roberta','distil']
    datasets = ['train.parquet','validation.parquet','test.parquet']
    types = ['train','validation','test']
    ds = []
    ds.append(pl.read_parquet('/home/ubuntu/dataset/ebnerd_large/train/behaviors.parquet'))
    ds.append(pl.read_parquet('/home/ubuntu/dataset/ebnerd_large/validation/behaviors.parquet'))
    ds.append(pl.read_parquet('/home/ubuntu/dataset/ebnerd_testset/test/behaviors.parquet'))
    
    for i in range(3):
        original = explode_ds(ds[i])
    
        print(f'Processing {datasets[i]}')
        
        path = f'~/tmp/{embeddings[0]}_{datasets[i]}'
        print(path)
        merged = pl.read_parquet(path)
        
        for emb in embeddings[1:]:
            path = f'~/tmp/{emb}_{datasets[i]}'
            print(path)
            merged = merged.join(pl.read_parquet(path), 
                                on = ['impression_id','user_id','article'])
    
        merged = reduce_polars_df_memory_size(merged)

        assert merged.shape[0] == original.shape[0]        
        merged = _add_normalized_features_emb(df = merged)
        print(merged.head())
        merged.write_parquet(f'~/tmp/{types[i]}_user_item_emb_dist.parquet')
