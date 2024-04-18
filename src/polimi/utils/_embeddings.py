import polars as pl
import tqdm
import simsimd
import numpy as np


def fast_distance(u, v):
    '''
    Computes the cosine distance between two vectors u and v.
    '''
    return simsimd.cosine(np.asarray(u), np.asarray(v))


def _build_embeddings_similarity(df, embeddings, users_embeddings, new_column_name) -> pl.DataFrame:
    '''
    Computes the cosine distance between the user and the item embeddings.

    Args:
        df: a dataframe that must contain 'user_id' and 'article' columns
        embeddings: a dataframe that must contain 'article_id' and 'item_embedding' columns
        users_embeddings: a dataframe that must contain 'user_id' and 'user_embedding' columns
        new_column_name: the name of the new column in the df dataframe

    Returns:
        a dataframe that contains 'user_id', 'article' and new_column_name columns with the cosine distance between the user and the item embeddings
    '''
    print('Computing embeddings similarity...')
    return df.select(['user_id', 'article']).join(embeddings, left_on='article', right_on='article_id')\
        .join(users_embeddings, on='user_id')\
        .with_columns(
        pl.struct(['user_embedding', 'item_embedding']).map_elements(
            lambda x: fast_distance(x['user_embedding'], x['item_embedding']), return_dtype=pl.Float32).cast(pl.Float32).alias(new_column_name)
    )\
        .select(['user_id', 'article', new_column_name])


def _build_user_embeddings(df, embeddings) -> pl.DataFrame:
    '''
    Builds user embeddings averaging the embeddings of the items in the user history.

    Args:
        df: a dataframe that must contain 'user_id' and 'article' columns
        embeddings: a dataframe that must contain 'article_id' and 'item_embedding' columns

    Returns:
        a dataframe that contains 'user_id' and 'user_embedding' columns with the user embeddings
    '''
    embedding_len = len(embeddings['item_embedding'].limit(1).item())

    print('Building user embeddings...')
    return pl.concat(
        rows.select('user_id', 'article_id_fixed').explode('article_id_fixed').rename(
            {'article_id_fixed': 'article_id'}).join(embeddings, on='article_id')
        .with_columns(pl.col("item_embedding").list.to_struct()).unnest("item_embedding")
        .group_by('user_id').agg(
            [pl.col(f'field_{i}').mean().cast(pl.Float32) for i in range(embedding_len)])
        .with_columns(
            pl.concat_list([f"field_{i}" for i in range(
                            embedding_len)]).alias('user_embedding')
        )
        .select('user_id', 'user_embedding')
        for rows in tqdm.tqdm(df.iter_slices(1000), total=df.shape[0] // 1000))


def build_embeddings_similarity(df, history, embeddings, new_column_name) -> pl.DataFrame:
    '''
    It computes an average of the embeddings of the items in the user history and then computes a cosine similarity between the embedding of the item and the new embedding of the user 
    (items and users used for the computation came from df rows).
    It concatenates the cosine similarity to df dataframe.

    Args:
        df: a dataframe that must contain 'user_id' and 'article' columns
        history: a dataframe that must contain 'user_id' and 'article' columns
        embeddings: a dataframe that must contain 'article_id' and 'item_embedding' columns
        new_column_name: the name of the new column in the df dataframe

    Returns:
        a dataframe that contains 'user_id', 'article' and new_column_name columns with the cosine distance between the user and the item embeddings
    '''

    user_embeddings = _build_user_embeddings(history, embeddings)
    return _build_embeddings_similarity(df, embeddings, user_embeddings, new_column_name)


def iterator_build_embeddings_similarity(df, users_embeddings, embeddings, new_column_name, n_batches: int = 10):
    '''
    This function is an iterator that yields the result of the build_embeddings_similarity function.
    It computes an average of the embeddings of the items in the user history and then computes a cosine similarity between the embedding of the item and the new embedding of the user
    (items and users used for the computation came from df rows).
    It concatenates the cosine similarity to df dataframe.

    Args:
        df: a dataframe that must contain 'user_id' and 'article' columns
        users_embeddings: a dataframe that must contain 'user_id' and 'user_embedding' columns
        embeddings: a dataframe that must contain 'article_id' and 'item_embedding' columns
        new_column_name: the name of the new column in the df dataframe
        n_batches: the number of batches to split the df dataframe
    '''

    for sliced_df in df.iter_slices(df.shape[0] // n_batches):
        slice_features = sliced_df.pipe(_build_embeddings_similarity, embeddings=embeddings,
                                        users_embeddings=users_embeddings, new_column_name=new_column_name)

        yield slice_features
