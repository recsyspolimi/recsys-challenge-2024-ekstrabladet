import polars as pl
from polimi.utils._polars import reduce_polars_df_memory_size
from tqdm import tqdm
import numpy as np
import tensorflow as tf


N_CATEGORY = 32
N_SUBCATEGORY = 262
N_TOPICS = 78
N_SENTIMENT_LABEL = 3
N_HOUR_GROUP = 6 #23(max_hour) // 4 = 5
N_WEEKDAY = 7 # [0, 6]


def build_history_seq(history: pl.DataFrame, articles: pl.DataFrame, batch_size=10000):
    topics = articles['topics'].explode().unique().drop_nans().drop_nulls().sort().to_frame().with_row_index(offset=1)
    category = articles['category'].unique().drop_nans().drop_nulls().sort().to_frame().with_row_index(offset=1)
    subcategory = articles['subcategory'].explode().unique().drop_nans().drop_nulls().sort().to_frame().with_row_index(offset=1)
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
        
    dummies_topics = articles_p.select('article_id', 'topics').explode('topics').fill_null(0)
    one_hot_topics = tf.keras.utils.to_categorical(dummies_topics['topics'].to_numpy(), num_classes=N_TOPICS + 1)
    dummies_topics = dummies_topics.with_columns(
        pl.Series(one_hot_topics, dtype=pl.List(pl.Int8)).alias('topics')
    ).with_columns(
        *[pl.col('topics').list.get(i).alias(f'topics_{i}') for i in range(N_TOPICS + 1)]
    ).drop('topics')
    
    
    dummies_subcategories = articles_p.select('article_id', 'subcategory').explode('subcategory').fill_null(0)
    one_hot_subcategories = tf.keras.utils.to_categorical(dummies_subcategories['subcategory'].to_numpy(), num_classes=N_SUBCATEGORY + 1)
    dummies_subcategories = dummies_subcategories.with_columns(
        pl.Series(one_hot_subcategories, dtype=pl.List(pl.Int8)).alias('subcategory')
    ).with_columns(
        *[pl.col('subcategory').list.get(i).alias(f'subcategory_{i}') for i in range(N_SUBCATEGORY + 1)]
    ).drop('subcategory')
               
    articles_p = articles_p.join(dummies_topics, on='article_id', how='left')\
        .join(dummies_subcategories, on='article_id', how='left')\
        .drop('topics', 'subcategory')
    articles_p = reduce_polars_df_memory_size(articles_p)
    
    
    history_seq = pl.concat([
        slice.explode(pl.all().exclude('user_id'))\
            .with_columns(
                pl.col('scroll_percentage_fixed').fill_null(0.),
                pl.col('read_time_fixed').fill_null(0.),
            )\
            .with_columns(
                (pl.col('impression_time_fixed').dt.hour() // 4).alias('hour_group'),
                pl.col('impression_time_fixed').dt.weekday().sub(1).alias('weekday'), #sub 1 to have from [0, 6]
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
# def build_sequences_seq(history_seq: pl.DataFrame, window: int, stride: int):
#     all_features = history_seq.drop('user_id').columns
    
#     multi_one_hot_cols = ['topics', 'subcategory']
#     categorical_cols = ['category', 'weekday', 'hour_group', 'sentiment_label']
#     name_idx_dict = {key: [i for i, col in enumerate(all_features) if col.startswith(key)] for key in multi_one_hot_cols + categorical_cols}
#     numerical_cols = ['scroll_percentage', 'read_time', 'premium']
#     name_idx_dict['numerical'] = [i for i, col in enumerate(all_features) if col in numerical_cols]
        
#     res = {key: ([], []) for key in name_idx_dict.keys()}
    
#     for user_df in tqdm(history_seq.partition_by('user_id', maintain_order=False)): #output order is not consistent with input order, but faster
#         x = user_df.drop('user_id').to_numpy()[0]
#         x = np.array([np.array(x_i) for x_i in x])
                
#         i = 0
#         if i + window >= x.shape[1]:
#             # in case history is shorter than the window then we pad it and select the last element as target
#             pad_width = window - (x.shape[1] - 1)
#             pad_m = np.zeros((x.shape[0], pad_width))
#             padded_x = np.concatenate((pad_m, x[:, :-1]), axis=1)
#             y_i = x[:, -1]
            
#             for key, idx in name_idx_dict.items():
#                 res[key][0].append(padded_x[idx, :].T)
#                 res[key][1].append(y_i[idx].T)
                
                
            
#         else:
#             while i + window < x.shape[1]:
#                 # in case history is larger than the window then we select the window and the target randomly between the next elements
#                 x_i = x[:, i:i+window]
#                 target_random_id = np.random.randint(i+window, x.shape[1])
#                 y_i = x[:, target_random_id]
                
#                 for key, idx in name_idx_dict.items():
#                     res[key][0].append(x_i[idx, :].T)
#                     res[key][1].append(y_i[idx].T)
                
#                 i+=stride
                         
#             #TODO: add padding for the last sequence, if we want to keep it
                

#     for key in res.keys():
#         res[key] = (np.array(res[key][0]), np.array(res[key][1]))
    
#     return res



# Return dict with key = name_feature and value the tuple (X, y) containing the input/output sequence
def build_sequences_seq_iterator(history_seq: pl.DataFrame, window: int, stride: int):
    all_features = history_seq.drop('user_id').columns
    
    multi_one_hot_cols = ['topics', 'subcategory']
    categorical_cols = ['category', 'weekday', 'hour_group', 'sentiment_label']
    # caterical_cols_num_classes = {key: history_seq[key].explode().max() + 1 for key in categorical_cols}  #uncomment if you don't want to hardcode
    caterical_cols_num_classes = {
        'category': N_CATEGORY + 1,#+1 to handle null values
        'weekday': N_WEEKDAY,
        'hour_group': N_HOUR_GROUP,
        'sentiment_label': N_SENTIMENT_LABEL + 1 #+1 to handle null
    }
    #it can be hardcoded if needed
    name_idx_dict = {key: [i for i, col in enumerate(all_features) if col.startswith(key)] for key in multi_one_hot_cols + categorical_cols}
    numerical_cols = ['scroll_percentage', 'read_time', 'premium']
    name_idx_dict['numerical'] = [i for i, col in enumerate(all_features) if col in numerical_cols]
    remove_from_output = ['weekday', 'hour_group', 'numerical']
    
    res_x = {}
    res_y = {}
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
                res_x[f'input_{key}'] = padded_x[idx, :].T
                if key not in remove_from_output:
                    y_i_key = y_i[idx].T
                    if key in categorical_cols:
                        y_i_key = tf.keras.utils.to_categorical(y_i_key, num_classes=caterical_cols_num_classes[key]).reshape(-1)
                    res_y[f'output_{key}'] = y_i_key
            yield res_x, res_y  
            
        else:
            while i + window < x.shape[1]:
                # in case history is larger than the window then we select the window and the target randomly between the next elements
                x_i = x[:, i:i+window]
                target_random_id = np.random.randint(i+window, x.shape[1])
                y_i = x[:, target_random_id]
                
                for key, idx in name_idx_dict.items():
                    res_x[f'input_{key}'] = x_i[idx, :].T   
                    if key not in remove_from_output:                 
                        y_i_key = y_i[idx].T
                        if key in categorical_cols:
                            y_i_key = tf.keras.utils.to_categorical(y_i_key, num_classes=caterical_cols_num_classes[key]).reshape(-1)
                        res_y[f'output_{key}'] = y_i_key
                yield res_x, res_y                
                i+=stride
                         
            #TODO: add padding for the last sequence, if we want to keep it



def build_sequences_cls_iterator(history_seq: pl.DataFrame, behaviors: pl.DataFrame, window:int):
    mask = 0
    history_seq_trucated = history_seq.with_columns(
        pl.all().exclude('user_id').list.reverse().list.eval(pl.element().extend_constant(mask, window)).list.reverse().list.tail(window).name.keep()
    )
    
    for user_id, user_history in tqdm(history_seq_trucated.partition_by(['user_id'], as_dict=True, maintain_order=False).items()): #order not maintained
        for b in behaviors.filter(pl.col('user_id') == user_id[0]).iter_slices(1):
            yield (b.drop('target'), user_history, b['target'].item())
        
    
    