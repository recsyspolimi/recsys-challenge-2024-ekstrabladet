import polars as pl
import tqdm
import simsimd
import numpy as np

def fast_distance(u,v):
    return simsimd.cosine(np.asarray(u),np.asarray(v))

def build_embeddings_similarity(df, history, embeddings, new_column_name) -> pl.DataFrame:
    '''
    This function takes in input 
     - a dataframe that must contain 'user_id' and 'article' columns
     - history dataframe
     - embedding dataframe
     - name of the new column in the df dataframe
    It computes an average of the embeddings of the items in the user history and then computes a cosine similarity between the embedding of the item and the new embedding of the user 
    (items and users used for the computation came from df rows).
    It concatenates the cosine similarity to df dataframe.
    '''
    embeddings = embeddings.rename({embeddings.columns[0] : 'article_id', embeddings.columns[1]: 'item_embedding'})
    embedding_len = len(embeddings['item_embedding'].limit(1).item())
    users_embeddings = pl.concat(
                    rows.select('user_id','article_id_fixed').explode('article_id_fixed').rename({'article_id_fixed':'article_id'}).join(embeddings,on='article_id')\
                        .with_columns(pl.col("item_embedding").list.to_struct()).unnest("item_embedding")\
                            .group_by('user_id').agg(
                                            [pl.col(f'field_{i}').mean() for i in range(embedding_len)])\
                            .with_columns(
                                            pl.concat_list([f"field_{i}" for i in range(embedding_len)]).alias('user_embedding')
                                            )\
                            .select('user_id','user_embedding')
                     for rows in  history.iter_slices(1000))
    user_item_emb_similarity = pl.concat(
                rows.select(['user_id','article']).join(embeddings, left_on = 'article', right_on = 'article_id')\
                .join(users_embeddings, on = 'user_id')\
                .with_columns(
                     pl.struct(['user_embedding', 'item_embedding']).map_elements(
                                        lambda x: fast_distance(x['user_embedding'], x['item_embedding']), return_dtype=pl.Float64).cast(pl.Float64).alias(new_column_name)
                )\
            for rows in tqdm.tqdm(df.iter_slices(1000), total=df.shape[0] // 1000)).unique()
    
    return  pl.concat(
                rows.join(user_item_emb_similarity, on = ['user_id','article'],how = 'left').fill_null(value = 0)
            for rows in df.iter_slices(1000))