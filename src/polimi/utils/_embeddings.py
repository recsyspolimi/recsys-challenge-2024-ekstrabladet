import polars as pl
from tqdm import tqdm
import simsimd
import numpy as np
from sklearn import preprocessing
from pathlib import Path

from polimi.utils._polars import reduce_polars_df_memory_size

def fast_distance(u, v):
    '''
    Computes the cosine distance between two vectors u and v.
    '''
    return simsimd.cosine(np.asarray(u), np.asarray(v))

def distance_function_768(x):
    return simsimd.cosine(np.asarray(x[:768]), np.asarray(x[768:]))

def distance_function_384(x):
    return simsimd.cosine(np.asarray(x[:384]), np.asarray(x[384:]))

def distance_function_300(x):
    return simsimd.cosine(np.asarray(x[:300]), np.asarray(x[300:]))
    
def distance_function_6(x):
    return simsimd.cosine(np.asarray(x[:6]), np.asarray(x[6:]))

def get_distance_function(len):
    if len == 768:
        return distance_function_768
    elif len == 300:
        return distance_function_300
    elif len == 384:
        return distance_function_384
    elif len == 6:
        return distance_function_6

def _add_normalized_features_emb(df = None ,path = None):
    
    if df is None and path is None:
        raise ValueError('You must provide a dataframe or a path')
    elif path is not None:
        df = pl.read_parquet(path)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # get the minmun value of each column
    columns= df.head(1).drop(['impression_id','user_id','article']).columns
    for col in range(3, len(df.columns)):
        df = df.replace_column(col, pl.Series(columns[col - 3], scaler.fit_transform(df.select(columns[col - 3]).to_numpy()).squeeze()))
    df = df.with_columns(
                            pl.mean_horizontal(columns).alias('mean_norm_user_item_emb_dist'),
                            pl.min_horizontal(columns).alias('min_norm_user_item_emb_dist'),
                            pl.max_horizontal(columns).alias('max_norm_user_item_emb_dist')
                         )
    return reduce_polars_df_memory_size(df)

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
        for rows in tqdm(df.iter_slices(1000), total=df.shape[0] // 1000))


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


def add_features_cosine_history_fb_embeddings(train_ds, articles, history, fb_embeddings):
    history_ds = history.select(["user_id", "article_id_fixed"])
    article_embeddings = fb_embeddings.select(["article_id","FacebookAI/xlm-roberta-base"]).rename({"article_id": "article_id_fixed","FacebookAI/xlm-roberta-base":"embeddings"})
    
    dfs = []
    for rows in tqdm.tqdm(train_ds.iter_slices(1000), total=train_ds.shape[0] // 1000):
        df = (
            rows.select(["impression_id", "user_id", "article"])  # Select only useful columns
            .join(article_embeddings, left_on="article",right_on="article_id_fixed", how="left")  # Add articles details
            .join(other=history_ds, on="user_id", how="left")  # Add history of the user
            .explode("article_id_fixed")  # Explode the user's history
            .join(other=article_embeddings, on="article_id_fixed", how="left")  # Add embeddings of the articles in the history
            .with_columns(
                # Calculate cosine similarity only with relevant embeddings
                pl.struct(['embeddings','embeddings_right']).map_elements(
                    lambda x: fast_distance(x['embeddings'], x['embeddings_right']),
                    return_dtype=pl.Float64
                ).alias('cos_sim')
            )
            .group_by(["impression_id", "article"])
            .agg([  # Grouping on all the "n" articles in the user's history, compute aggregations of the "n" cosine similarity values
                pl.col("cos_sim").mean().alias("fb_mean_cos_sim"),
                pl.col("cos_sim").min().alias("fb_min_cos_sim"),
                pl.col("cos_sim").max().alias("fb_max_cos_sim"),
                pl.col("cos_sim").std().alias("fb_std_cos_sim")]
            )
        )
        dfs.append(df)
    
    df = pl.concat(dfs)
    
    return train_ds.join(other=df, on=["impression_id", "article"], how="left")

def calculate_weights_decay_ImprTime(impression_times, decay_rate):
    # Convert impression times to numpy array for easier calculations
    impression_times_np = np.array(impression_times)
    
    # Calculate time differences relative to the latest impression time
    time_diffs = impression_times_np.max() - impression_times_np
    time_diffs = time_diffs.astype(float)
    # Apply exponential decay function to the time differences to compute weights
    weights = np.exp(-decay_rate * time_diffs)
    
    # Normalize the weights
    sum_weights = np.sum(weights)
    normalized_weights = weights / sum_weights if sum_weights != 0 else weights
    
    # Convert the normalized weights array to a Python list of floats
    normalized_weights_list = normalized_weights.tolist()
    
    return normalized_weights_list

def build_weighted_timestamps_embeddings(df, history, embeddings, emb_type):
    '''
    This function computes an average of the embeddings of the items in the user history weighting them on impression_time_fixed
    and then computes a cosine similarity between the embedding of the item and the new embedding of the user 
    (items and users used for the computation come from df rows).
    It concatenates the cosine similarity to df dataframe.
    Requires that history contains 'impression_time_fixed'
    Requires embedding to have two columns: article_id and embedding.
    '''
    embeddings = embeddings.rename({embeddings.columns[0]: 'article_id', embeddings.columns[1]: 'item_embedding'})
    embedding_len = len(embeddings['item_embedding'].limit(1).item())
    history = history.with_columns(pl.col('impression_time_fixed').map_elements(lambda x : calculate_weights_decay_ImprTime(x,1e-10)).alias('Weights_time_decay'))
    

    # Iterate over history dataframe in chunks
    users_embeddings = pl.concat(
        [
            (
                rows.select('user_id', 'article_id_fixed', 'impression_time_fixed','Weights_time_decay')
                .explode('article_id_fixed','impression_time_fixed','Weights_time_decay')
                .rename({'article_id_fixed': 'article_id'})
                .join(embeddings, on='article_id')
                .with_columns(
                    pl.col("item_embedding").list.to_struct()
                ).unnest("item_embedding")
                .with_columns(
                    [pl.col(f'field_{i}').mul(pl.col('Weights_time_decay'))for i in range(embedding_len)]
                
                )
                .group_by('user_id')
                .agg(
                    [pl.col(f'field_{i}').mean().alias(f'embeddings_mean_{i}') for i in range(embedding_len)]
                )
                .with_columns(pl.concat_list([f'embeddings_mean_{i}' for i in range(embedding_len)]).alias('user_embedding_weighted_TS'))
                .select('user_id', 'user_embedding_weighted_TS')
            )
            for rows in tqdm(history.iter_slices(1000), total=history.shape[0] // 1000)
        ]
    )

    return users_embeddings, 'user_embedding_weighted_TS', f'TW_click_predictor_{emb_type}'

def calculate_weights_scroll(scroll_percentages):
    # Convert scroll percentages to numpy array for easier calculations
    scroll_percentages_np = np.array(scroll_percentages)
    arr_filled = np.array([np.nan_to_num(sub_arr, nan=0) for sub_arr in scroll_percentages_np])
    weights = np.sqrt(arr_filled)
    
    # Normalize the weights
    sum_weights = np.sum(weights)
    normalized_weights = weights / sum_weights if sum_weights != 0 else weights
    
    # Convert the normalized weights array to a Python list of floats
    normalized_weights_list = normalized_weights.tolist()
    
    return normalized_weights_list

def calculate_weights_readtimes(read_times):
    # Convert read times to numpy array for easier calculations
    read_times_np = np.array(read_times)
    
    # Calculate the difference between each read time and the maximum read time
    max_read_time = max(read_times_np)  # Assuming the maximum read time is known
    diff_with_max = max_read_time - read_times_np
    
    # Apply square root function to the difference to compute weights
    weights = np.sqrt(diff_with_max)
    
    # Normalize the weights
    sum_weights = np.sum(weights)
    normalized_weights = weights / sum_weights if sum_weights != 0 else weights
    
    # Convert the normalized weights array to a Python list of floats
    normalized_weights_list = normalized_weights.tolist()
    
    return normalized_weights_list

def build_weighted_SP_embeddings(df, history, embeddings, emb_type):
    '''
    This function computes an average of the embeddings of the items in the user history 
    and then computes a cosine similarity between the embedding of the item and the new embedding of the user 
    (items and users used for the computation come from df rows).
    It concatenates the cosine similarity to df dataframe.
    '''
    embeddings = embeddings.rename({embeddings.columns[0]: 'article_id', embeddings.columns[1]: 'item_embedding'})
    embedding_len = len(embeddings['item_embedding'].limit(1).item())
    history = history.with_columns(pl.col('scroll_percentage_fixed').map_elements(lambda x : calculate_weights_scroll(x)).alias('Weights_scroll'))
    # Iterate over history dataframe in chunks
    users_embeddings = pl.concat(
        [
            (
                rows.select('user_id', 'article_id_fixed', 'scroll_percentage_fixed','Weights_scroll')
                .explode('article_id_fixed','scroll_percentage_fixed','Weights_scroll')
                .rename({'article_id_fixed': 'article_id'})
                .join(embeddings, on='article_id')
                .with_columns(
                    pl.col("item_embedding").list.to_struct()
                ).unnest("item_embedding")
                .with_columns(
                    [pl.col(f'field_{i}').mul(pl.col('Weights_scroll'))for i in range(embedding_len)]
                
                )
                .group_by('user_id')
                .agg(
                    [pl.col(f'field_{i}').mean().alias(f'embeddings_mean_{i}') for i in range(embedding_len)]
                )
                .with_columns(pl.concat_list([f'embeddings_mean_{i}' for i in range(embedding_len)]).alias('user_embedding_weight_SP'))
                .select('user_id', 'user_embedding_weight_SP')
            )
            for rows in tqdm(history.iter_slices(1000), total=history.shape[0] // 1000)
        ]
    )

    return users_embeddings, 'user_embedding_weight_SP', f'SP%W_click_predictor_{emb_type}'
    # Iterate over df dataframe in chunks
 

def build_weighted_readtime_embeddings(df, history, embeddings, emb_type):
    '''
    This function computes an average of the embeddings of the items in the user history 
    and then computes a cosine similarity between the embedding of the item and the new embedding of the user 
    (items and users used for the computation come from df rows).
    It concatenates the cosine similarity to df dataframe.
    '''
    embeddings = embeddings.rename({embeddings.columns[0]: 'article_id', embeddings.columns[1]: 'item_embedding'})
    embedding_len = len(embeddings['item_embedding'].limit(1).item())
    history = history.with_columns( pl.col('read_time_fixed').map_elements(lambda x : calculate_weights_readtimes(x)).alias('Weights_readtime'))

    # Iterate over history dataframe in chunks
    users_embeddings = pl.concat(
        [
            (
                rows.select('user_id', 'article_id_fixed', 'scroll_percentage_fixed','Weights_readtime')
                .explode('article_id_fixed','scroll_percentage_fixed','Weights_readtime')
                .rename({'article_id_fixed': 'article_id'})
                .join(embeddings, on='article_id')
                .with_columns(
                    pl.col("item_embedding").list.to_struct()
                ).unnest("item_embedding")
                .with_columns(
                    [pl.col(f'field_{i}').mul(pl.col('Weights_readtime'))for i in range(embedding_len)]
                
                )
                .group_by('user_id')
                .agg(
                    [pl.col(f'field_{i}').mean().alias(f'embeddings_mean_{i}') for i in range(embedding_len)]
                )
                .with_columns(pl.concat_list([f'embeddings_mean_{i}' for i in range(embedding_len)]).alias('user_embedding_weight_readtime'))
                .select('user_id', 'user_embedding_weight_readtime')
            )
            for rows in tqdm(history.iter_slices(1000), total=history.shape[0] // 1000)
        ]
    )
    
    return users_embeddings, 'user_embedding_weight_readtime', f'readtime_click_predictor_{emb_type}'

def _build_mean_user_embeddings(df, history, embeddings, emb_type) -> pl.DataFrame:
    embeddings = embeddings.rename({embeddings.columns[0]: 'article_id', embeddings.columns[1]: 'item_embedding'})
    embedding_len = len(embeddings['item_embedding'].limit(1).item())

    print('Building user embeddings...')
    users_embeddings =  pl.concat(
        rows.select('user_id', 'article_id_fixed').explode('article_id_fixed')\
        .rename({'article_id_fixed': 'article_id'}).join(embeddings, on='article_id')\
        .with_columns(pl.col("item_embedding").list.to_struct()).unnest("item_embedding")
        .group_by('user_id').agg(
            [pl.col(f'field_{i}').mean().cast(pl.Float32) for i in range(embedding_len)])
        .with_columns(
            pl.concat_list([f"field_{i}" for i in range(
                            embedding_len)]).alias('mean_user_embedding')
        )
        .select('user_id', 'mean_user_embedding')
        for rows in tqdm(history.iter_slices(1000), total=history.shape[0] // 1000))
    
    return users_embeddings, 'mean_user_embedding', f'{emb_type}_user_item_distance'


def compute_similarity(df, users_embeddings, embeddings, column_name, new_col_name):
    
    df = df.with_columns(pl.col('user_id').cast(pl.UInt32),
                         pl.col('article').cast(pl.Int32))
    user_item_emb_similarity = pl.concat(
                rows.select(['user_id','article']).join(embeddings, left_on = 'article', right_on = 'article_id')\
                .join(users_embeddings, on = 'user_id')\
                .with_columns(
                     pl.struct([column_name, 'item_embedding']).map_elements(
                                        lambda x: fast_distance(x[column_name], x['item_embedding']), return_dtype=pl.Float32).cast(pl.Float32).alias(new_col_name)
                ).drop(['item_embedding_right','user_embedding_right'])
            for rows in tqdm(df.iter_slices(1000), total=df.shape[0] // 1000)).unique(subset=['user_id', 'article'])
    
    return user_item_emb_similarity.select(['user_id','article',new_col_name])
  

def iterator_weighted_embedding(df, users_embeddings, embeddings, column_name, new_col_name, n_batches: int = 10):

    for sliced_df in df.iter_slices(df.shape[0] // n_batches):
        slice_features = sliced_df.pipe(compute_similarity, users_embeddings=users_embeddings,
                                        embeddings=embeddings, column_name= column_name, new_col_name=new_col_name)

        yield slice_features
        
        
        

# Embeddings Lorenzo
EMB_NAME_DICT = {'Ekstra_Bladet_contrastive_vector': 'contrastive_vector',
                 'FacebookAI_xlm_roberta_base': 'xlm_roberta_base',
                 'Ekstra_Bladet_image_embeddings': 'image_embeddings',
                 'google_bert_base_multilingual_cased': 'bert_base_multilingual_cased'}


def build_norm_m_dict(articles: pl.DataFrame, dataset_path: Path, emb_name_dict:dict=EMB_NAME_DICT, logging=print):
    norm_m_dict = {}
    article_emb_mapping = articles.select('article_id').unique().with_row_index()
    for dir, file_name in emb_name_dict.items():
        logging(f'Processing {file_name} embedding matrix...')
        emb_df = pl.read_parquet(dataset_path / dir / f'{file_name}.parquet')
        emb_df.columns = ['article_id', 'embedding']
        logging(f'Building normalized embeddings matrix for {file_name}...')
        m = build_normalized_embeddings_matrix(emb_df, article_emb_mapping)
        norm_m_dict[file_name] = m
    return norm_m_dict

def build_normalized_embeddings_matrix(emb_df: pl.DataFrame, article_emb_mapping: pl.DataFrame, shrinkage: float = 1e-6, logging=print):
    missing_articles_in_embedding = list(set(article_emb_mapping['article_id'].to_numpy()) - set(emb_df['article_id'].to_numpy()))
    if len(missing_articles_in_embedding) > 0:
        logging(f'[Warning... {len(missing_articles_in_embedding)} missing articles in embedding matrix]')
        emb_size = len(emb_df['embedding'][0])
        null_vector = np.zeros(emb_size, dtype=np.float32)
        emb_df = emb_df.vstack(pl.DataFrame({'article_id': missing_articles_in_embedding, 'embedding': [null_vector] * len(missing_articles_in_embedding)}))
        
    emb_df = article_emb_mapping.join(emb_df, on='article_id', how='left')
    m = np.array([np.array(row) for row in emb_df['embedding'].to_numpy()])
    row_norms = np.linalg.norm(m, axis=1, keepdims=True)
    m = m / (row_norms + shrinkage)
    return m


def build_embeddings_scores(behaviors:pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame, norm_m_dict:dict):

    article_emb_mapping = articles.select('article_id').unique().with_row_index()

    history_m = history\
        .select('user_id', pl.col('article_id_fixed').list.eval(
                    pl.element().replace(article_emb_mapping['article_id'], article_emb_mapping['index'], default=None)))\
        .with_row_index('user_index')

    user_history_map = history_m.select('user_id', 'user_index')
    history_m = history_m['article_id_fixed'].to_numpy()
    df = behaviors.select('impression_id', 'user_id', pl.col('article_ids_inview').alias('article'))\
        .join(user_history_map, on='user_id')\
        .with_columns(
            pl.col('article').list.eval(
                pl.element().replace(article_emb_mapping['article_id'], article_emb_mapping['index'], default=None))\
                    .name.suffix('_index'),
        ).drop('impression_time_fixed', 'scroll_percentage_fixed', 'read_time_fixed')\
        .sort(['user_id', 'impression_id'])
        
    df = pl.concat([
        slice.explode(['article_index', 'article']).with_columns(
            *[pl.lit(
                np.dot(
                    m[slice['article_index'].explode().to_numpy()], 
                    m[history_m[key[0]]].T
                    )
                ).alias(f'{emb_name}_scores') for emb_name, m in norm_m_dict.items()]
        )\
        .drop(['article_index', 'user_index'])\
        .group_by(['impression_id', 'user_id'])\
        .agg(pl.all())
        for key, slice in tqdm(df.partition_by(by=['user_index'], as_dict=True).items(), total=df['user_index'].n_unique())
    ])
    return df

def build_history_w(history: pl.DataFrame, articles: pl.DataFrame):
    history_w_articles = history.explode(pl.all().exclude('user_id')).join(
        articles.select('article_id', 
            (pl.col('body') + pl.col('title') + pl.col('subtitle')).str.len_chars().alias('article_id_fixed_article_len'),
            'last_modified_time', 'published_time'), left_on='article_id_fixed', right_on='article_id'
        )\
        .with_columns(
            (pl.col('impression_time_fixed') - pl.col('published_time')).alias('time_to_impression'),
        ).group_by('user_id').agg(pl.all())
        
    
    history_w = history_w_articles.select('user_id', 'time_to_impression', 
                                          'impression_time_fixed', 'scroll_percentage_fixed', 
                                          'read_time_fixed', 'article_id_fixed_article_len')\
        .explode(pl.all().exclude('user_id'))\
        .with_columns(pl.col('scroll_percentage_fixed').fill_null(0.0))\
        .with_columns(
            pl.col('read_time_fixed').truediv(pl.col('article_id_fixed_article_len') + 1).alias('read_time_fixed_article_len_ratio'),
            # scroll_percentage
            (pl.col('scroll_percentage_fixed') - pl.col('scroll_percentage_fixed').min()).truediv(pl.col('scroll_percentage_fixed').max() - pl.col('scroll_percentage_fixed').min()).over('user_id').alias('scroll_percentage_fixed_mmnorm'),
            # time_to_impression
            pl.col('time_to_impression').dt.total_hours().sqrt().alias('time_to_impression_hours_sqrt'),
            pl.lit(1).truediv(pl.col('time_to_impression').dt.total_hours().sqrt() + 1).alias('time_to_impression_hours_inverse_sqrt'),
        ).group_by('user_id').agg(pl.all())\
        .with_columns(
            pl.col('read_time_fixed_article_len_ratio').list.eval(pl.element().truediv(pl.element().sum()).cast(pl.Float32)).alias('read_time_fixed_article_len_ratio_l1_w'),
            pl.col('scroll_percentage_fixed_mmnorm').list.eval(pl.element().truediv(pl.element().sum()).cast(pl.Float32)).alias('scroll_percentage_fixed_mmnorm_l1_w'),
            pl.col('time_to_impression_hours_sqrt').list.eval(pl.element().truediv(pl.element().sum()).cast(pl.Float32)).alias('time_to_impression_hours_sqrt_l1_w'),
            pl.col('time_to_impression_hours_inverse_sqrt').list.eval(pl.element().truediv(pl.element().sum()).cast(pl.Float32)).alias('time_to_impression_hours_inverse_sqrt_l1_w'),
        )
    l1_w_cols = [col for col in history_w.columns if col.endswith('_l1_w')]
    history_w = history_w.select('user_id', *l1_w_cols)
    return history_w

def weight_scores(df: pl.DataFrame, scores_cols: list[str], weights_cols: list[str]):
    df = reduce_polars_df_memory_size(df)
    return pl.concat([
        slice.explode(['article'] + scores_cols).with_columns(
            *[pl.lit(
                np.array([np.array(i) for i in slice[col_score].explode().to_numpy()]) * slice[col_w][0].to_numpy(),
                dtype=pl.List(pl.Float32)
            ).alias(f'{col_score}_weighted_{col_w}')
            for col_w in weights_cols for col_score in scores_cols]
        ).drop(weights_cols).group_by('impression_id', 'user_id').agg(pl.all())
        for slice in tqdm(df.partition_by('user_id'), total=df['user_id'].n_unique())    
    ])

def build_embeddings_agg_scores(df: pl.DataFrame, agg_cols: list[str] = [], last_k: list[int] = [], batch_size:int=50000, drop:bool=True):
    df = pl.concat([
        slice.with_columns(
            *[pl.col(col).list.eval(pl.element().list.mean()).name.suffix('_mean') for col in agg_cols],
            *[pl.col(col).list.eval(pl.element().list.max()).name.suffix('_max') for col in agg_cols],
            *[pl.col(col).list.eval(pl.element().list.min()).name.suffix('_min') for col in agg_cols],
            *[pl.col(col).list.eval(pl.element().list.std()).name.suffix('_std') for col in agg_cols],
            *[pl.col(col).list.eval(pl.element().list.median()).name.suffix('_median') for col in agg_cols],
            # Last k
            *[pl.col(col).list.eval(pl.element().list.tail(k).list.mean()).name.suffix(f'_mean_tail_{k}') for k in last_k for col in agg_cols],
            *[pl.col(col).list.eval(pl.element().list.tail(k).list.max()).name.suffix(f'_max_tail_{k}') for k in last_k for col in agg_cols],
            *[pl.col(col).list.eval(pl.element().list.tail(k).list.min()).name.suffix(f'_min_tail_{k}') for k in last_k for col in agg_cols],
            *[pl.col(col).list.eval(pl.element().list.tail(k).list.std()).name.suffix(f'_std_tail_{k}') for k in last_k for col in agg_cols],
            *[pl.col(col).list.eval(pl.element().list.tail(k).list.median()).name.suffix(f'_median_tail_{k}') for k in last_k for col in agg_cols],
        ).drop(agg_cols if drop else [])
        for slice in tqdm(df.iter_slices(batch_size), total=df.shape[0] // batch_size)
    ])
    return df