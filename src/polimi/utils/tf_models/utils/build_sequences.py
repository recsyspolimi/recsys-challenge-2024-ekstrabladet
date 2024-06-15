import polars as pl
from polimi.utils._polars import reduce_polars_df_memory_size
from tqdm import tqdm
import numpy as np



def build_history_seq(history: pl.DataFrame, articles: pl.DataFrame, batch_size=10000):
    topics = articles['topics'].explode().unique().drop_nans().drop_nulls().sort().to_frame().with_row_index()
    category = articles['category'].unique().drop_nans().drop_nulls().sort().to_frame().with_row_index(offset=1)
    subcategory = articles['subcategory'].explode().unique().drop_nans().drop_nulls().sort().to_frame().with_row_index()
    sentiment_label = articles['sentiment_label'].explode().unique().drop_nans().drop_nulls().sort().to_frame().with_row_index(offset=1)
    mask = 0

    articles_p = articles.select(['article_id', 'category', 'subcategory', 'premium', 'topics', 'sentiment_label'])\
        .with_columns(
            pl.col('topics').fill_null(pl.lit([])),
            pl.col('subcategory').fill_null(pl.lit([]))
        )\
        .with_columns(
            pl.col('topics').list.eval(pl.element().replace(topics['topics'], topics['index'], default=None)).list.drop_nulls().cast(pl.List(pl.Int32)),
            pl.col('category').replace(category['category'], category['index'], default=None).fill_null(mask).cast(pl.Int32),
            pl.col('sentiment_label').replace(sentiment_label['sentiment_label'], sentiment_label['index'], default=None).fill_null(mask).cast(pl.Int32),
            pl.col('subcategory').list.eval(pl.element().replace(subcategory['subcategory'], subcategory['index'], default=None)).list.drop_nulls().cast(pl.List(pl.Int32)),
            pl.col('premium').cast(pl.Int8)
    )
        
    dummies_topics = articles_p.select('article_id', 'topics').explode('topics').drop_nulls().to_dummies(columns=['topics'])\
        .group_by('article_id').agg(pl.all().sum())
    dummies_subcategories = articles_p.select('article_id', 'subcategory').explode('subcategory').drop_nulls().to_dummies(columns=['subcategory'])\
        .group_by('article_id').agg(pl.all().sum())
        
        
    articles_p = articles_p.join(dummies_topics, on='article_id', how='left')\
        .join(dummies_subcategories, on='article_id', how='left')\
        .drop('topics', 'subcategory')
    one_hot_cols = [col for col in articles_p.columns if col.startswith('topics_') or col.startswith('subcategory_')]
    articles_p = articles_p.with_columns(
        pl.col(one_hot_cols).fill_null(0)
    )
    articles_p = reduce_polars_df_memory_size(articles_p)
    
    
    history_seq = pl.concat([
        slice.explode(pl.all().exclude('user_id'))\
            .with_columns(
                pl.col('scroll_percentage_fixed').fill_null(0.),
                pl.col('read_time_fixed').fill_null(0.),
            )\
            .with_columns(
                (pl.col('impression_time_fixed').dt.hour() // 4).alias('hour_group'),
                pl.col('impression_time_fixed').dt.weekday().alias('weekday'),
            ).drop('impression_time_fixed')\
            .rename({'scroll_percentage_fixed': 'scroll_percentage', 'read_time_fixed': 'read_time'})
            .join(articles_p, left_on='article_id_fixed', right_on='article_id', how='left').drop('article_id_fixed')\
            .group_by('user_id').agg(pl.all())
        for slice in history.iter_slices(batch_size)
    ])
    
    # Order columns (especially one_hot ones)
    topics_cols = sorted([col for col in history_seq.columns if col.startswith('topics_')], key=lambda x: int(x.split('_')[-1]))
    subcategory_cols = sorted([col for col in history_seq.columns if col.startswith('subcategory_')], key=lambda x: int(x.split('_')[-1]))
    all_others = set(history_seq.columns) - set(topics_cols) - set(subcategory_cols) - {'user_id'}
    history_seq = history_seq.select(['user_id'] + list(all_others) + topics_cols + subcategory_cols)
    
    return history_seq
    
    

# Return dict with key = name_feature and value the tuple (X, y) containing the input/output sequence
def build_sequences_seq(history_seq: pl.DataFrame, window: int, stride: int):
    all_features = history_seq.drop('user_id').columns
    
    multi_one_hot_cols = ['topics', 'subcategory']
    categorical_cols = ['category', 'weekday', 'hour_group', 'sentiment_label']
    name_idx_dict = {key: [i for i, col in enumerate(all_features) if col.startswith(key)] for key in multi_one_hot_cols + categorical_cols}
    numerical_cols = ['scroll_percentage', 'read_time', 'premium']
    name_idx_dict['numerical'] = [i for i, col in enumerate(all_features) if col in numerical_cols]
        
    res = {key: ([], []) for key in name_idx_dict.keys()}

    for user_df in tqdm(history_seq.partition_by('user_id', maintain_order=False)): #output order is not consistent with input order, but faster
        x = user_df.drop('user_id').to_numpy()[0]
        x = np.array([np.array(x_i) for x_i in x])
                
        i = 0
        if i + window >= x.shape[1]:
            # in case history is shorter than the window then we pad it and select the last element as target
            pad_width = window - (x.shape[1] - 1)
            pad_m = np.zeros((x.shape[0], pad_width))
            padded_x = np.concatenate((pad_m, x[:, :-1]), axis=1)
            y_i = x[:, -1]
            
            for key, idx in name_idx_dict.items():
                res[key][0].append(padded_x[idx, :].T)
                res[key][1].append(y_i[idx].T)
            
        else:
            while i + window < x.shape[1]:
                # in case history is larger than the window then we select the window and the target randomly between the next elements
                x_i = x[:, i:i+window]
                target_random_id = np.random.randint(i+window, x.shape[1])
                y_i = x[:, target_random_id]
                
                for key, idx in name_idx_dict.items():
                    res[key][0].append(x_i[idx, :].T)
                    res[key][1].append(y_i[idx].T)
                
                i+=stride
                         
            #TODO: add padding for the last sequence, if we want to keep it
                

    for key in res.keys():
        res[key] = (np.array(res[key][0]), np.array(res[key][1]))
    
    return res




def build_sequences_cls_iterator(history_seq: pl.DataFrame, behaviors: pl.DataFrame, window:int):
    mask = 0
    history_seq_trucated = history_seq.with_columns(
        pl.all().exclude('user_id').list.reverse().list.eval(pl.element().extend_constant(mask, window)).list.reverse().list.tail(window).name.keep()
    )
    
    for user_id, user_history in tqdm(history_seq_trucated.partition_by(['user_id'], as_dict=True, maintain_order=False).items()): #order not maintained
        for b in behaviors.filter(pl.col('user_id') == user_id[0]).iter_slices(1):
            yield (b.drop('target'), user_history, b['target'].item())
        
    
    