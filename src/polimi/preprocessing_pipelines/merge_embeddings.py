import polars as pl

from polimi.utils._polars import reduce_polars_df_memory_size

def explode_ds(df):
    return df.select('impression_id','user_id','article_ids_inview').explode('article_ids_inview')\
        .rename({'article_ids_inview':'article'})

if __name__ == "__main__":
       
    embeddings = ['emotions','w2v','contrastive','roberta',]
    datasets = ['train.parquet','validation.parquet','test.parquet']
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
            
        print(merged.head())
        merged = reduce_polars_df_memory_size(merged)
        
        assert merged.shape[0] == original.shape[0]
        
