try:
    import polars as pl
except ImportError:
    print("polars not available")

from pathlib import Path
import time
import scipy.sparse as sps
from tqdm import tqdm
import numpy as np
from RecSys_Course_AT_PoliMi.Recommenders.BaseRecommender import BaseRecommender
from RecSys_Course_AT_PoliMi.Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from polimi.utils._polars import reduce_polars_df_memory_size
import logging


"""
Utils for building URM tables.
""" 
BATCH_SIZE=70000
def build_articles_with_processed_ner(articles: pl.DataFrame):        
        return articles.select('article_id', 'ner_clusters')\
            .with_columns(pl.col('ner_clusters').list.eval(pl.element().str.strip_chars_start('\"')))\
            .with_columns(pl.col('ner_clusters').list.eval(pl.element().str.strip_chars(' ')))\
            .with_columns(pl.col('ner_clusters').list.eval(pl.element().str.to_lowercase()))\
            .with_columns(pl.col('ner_clusters').list.eval(pl.element().filter(pl.element().str.len_chars() > 0)))\
            .with_columns(pl.col('ner_clusters').list.drop_nulls())\
            .with_columns(pl.col('ner_clusters').list.unique())\
            .with_columns(pl.col('ner_clusters').list.sort())\
            .sort('article_id')\
            .set_sorted('article_id')

def compute_sparsity_ratio(URM: sps.csr_matrix):
    total_elements = URM.shape[0] * URM.shape[1]
    nonzero_elements = URM.count_nonzero()
    sparsity_ratio = 1 - (nonzero_elements / total_elements)
    return sparsity_ratio


def _build_implicit_urm(df: pl.DataFrame, x_col: str, y_col: str, x_mapping: pl.DataFrame, y_mapping: pl.DataFrame):
    return sps.csr_matrix((np.ones(df.shape[0]), (df[x_col].to_numpy(), df[y_col].to_numpy())),
                         shape=(x_mapping.shape[0], y_mapping.shape[0]))
    
def build_user_id_mapping(history: pl.DataFrame):
    return history.select('user_id')\
        .unique('user_id') \
        .cast(pl.UInt32)\
        .drop_nulls() \
        .sort('user_id') \
        .with_row_index() \
        .select(['index', 'user_id'])\
        .rename({'index': 'user_index'})        
        
def build_ner_mapping(articles: pl.DataFrame):
    return articles\
        .select('ner_clusters')\
        .explode('ner_clusters') \
        .rename({'ner_clusters': 'ner'})\
        .unique('ner')\
        .drop_nulls()\
        .sort('ner')\
        .with_row_index()\
        .rename({'index': 'ner_index'})\
        .cast({'ner_index': pl.UInt32})


def build_item_mapping(articles: pl.DataFrame):
    return articles\
        .select('article_id')\
        .unique()\
        .drop_nulls()\
        .sort('article_id')\
        .with_row_index()\
        .rename({'index': 'item_index'})

def _build_batch_ner_interactions(df: pl.DataFrame, 
                  articles: pl.DataFrame, 
                  user_id_mapping: pl.DataFrame,
                  ner_mapping: pl.DataFrame,
                  articles_id_col = 'article_id_fixed',
                  batch_size=BATCH_SIZE):
    
    articles_ner_index = articles\
        .with_columns(pl.col('ner_clusters').list.eval(pl.element().replace(ner_mapping['ner'], ner_mapping['ner_index'], default=None, return_dtype=pl.UInt32)).list.drop_nulls())
    
    df = df.select('user_id', articles_id_col)\
        .group_by('user_id')\
        .agg(pl.col(articles_id_col).flatten().unique())
        
    df = pl.concat([
        slice.with_columns(
            pl.col(articles_id_col).list.eval(pl.element().replace(articles_ner_index['article_id'], articles_ner_index['ner_clusters'], default=None, return_dtype=pl.List(pl.UInt32)))\
                .list.eval(pl.element().flatten())\
                .list.drop_nulls()\
                .list.unique()\
                .list.sort()\
        ).rename({articles_id_col: 'ner_index'})\
        .filter(pl.col('ner_index').list.len() > 0)\
        .with_columns(pl.col('user_id').replace(user_id_mapping['user_id'], user_id_mapping['user_index']).cast(pl.UInt32))
        for slice in tqdm(df.iter_slices(batch_size), total=df.shape[0]//batch_size)
    ]).rename({'user_id': 'user_index'})\
    .sort('user_index')\
    .explode('ner_index')
        
    return reduce_polars_df_memory_size(df)
    


def build_ner_urm(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame, articles_id_col='article_id_fixed', batch_size=BATCH_SIZE):
    ap = build_articles_with_processed_ner(articles)
    ner_mapping = build_ner_mapping(ap)
    user_id_mapping = build_user_id_mapping(history)
    if articles_id_col == 'article_id_fixed':
        ner_interactions = _build_batch_ner_interactions(history, articles, user_id_mapping, ner_mapping, articles_id_col, batch_size=batch_size)
    else:
        ner_interactions = _build_batch_ner_interactions(behaviors, articles, user_id_mapping, ner_mapping, articles_id_col, batch_size=batch_size)

    return _build_implicit_urm(ner_interactions, 'user_index', 'ner_index', user_id_mapping, ner_mapping)
    
def _build_recsys_interactions(
                history: pl.DataFrame,
                user_id_mapping: pl.DataFrame,
                item_mapping: pl.DataFrame,
                articles_id_col = 'article_id_fixed'):
    
    df = history\
            .select('user_id',articles_id_col)\
            .explode(articles_id_col)\
            .unique()\
            .join(user_id_mapping,on='user_id')\
            .join(item_mapping,left_on=articles_id_col,right_on='article_id')\
            .unique(['user_index','item_index'])\
            .rename({articles_id_col:'article_id'})
    
    return reduce_polars_df_memory_size(df)
       

def build_recsys_urm(history: pl.DataFrame,
                     user_id_mapping: pl.DataFrame,
                     item_mapping: pl.DataFrame,
                     articles_id_col = 'article_id_fixed'
                    ):
    recsys_interactions = _build_recsys_interactions(history, user_id_mapping, item_mapping, articles_id_col)
    return _build_implicit_urm(recsys_interactions,'user_index', 'item_index', user_id_mapping, item_mapping)
        

def train_recommender(URM: sps.csr_matrix, recommender: BaseRecommender, params: dict, file_name:str = None, output_dir: Path = None):
    rec_instance= recommender(URM)
    rec_instance.fit(**params)

    if output_dir:
        rec_instance.save_model(folder_path=str(output_dir), file_name=file_name)

    return rec_instance


def load_recommender(URM: sps.csr_matrix, recommender: BaseRecommender, file_path: Path, file_name: str=None):
    rec_instance = recommender(URM)
    rec_instance.load_model(folder_path=str(file_path), file_name=file_name)
    return rec_instance

def build_recsys_features_icms(URM_train, ICMs,history,behaviors,articles ):

    recs = []


    logging.info('Training content-base recommenders ... ')
    contrastive = ItemKNNCBFRecommender(URM_train=URM_train,ICM_train=ICMs[0],verbose=False)
    contrastive.fit(similarity= 'asymmetric', topK= 192, shrink= 569, asymmetric_alpha= 0.9094884938503743) 
    recs.append(contrastive)

    w_2_vec = ItemKNNCBFRecommender(URM_train=URM_train,ICM_train=ICMs[1],verbose=False)
    w_2_vec.fit(similarity= 'cosine', topK= 359, shrink= 562) 
    recs.append(w_2_vec)

    bert = ItemKNNCBFRecommender(URM_train=URM_train,ICM_train=ICMs[2],verbose=False)
    bert.fit(similarity= 'euclidean', topK= 1457, shrink= 329, normalize_avg_row= True, similarity_from_distance_mode= 'exp', normalize= False) 
    recs.append(bert)

    roberta = ItemKNNCBFRecommender(URM_train=URM_train,ICM_train=ICMs[3],verbose=False)
    roberta.fit(similarity= 'cosine', topK= 363, shrink= 29) 
    recs.append(roberta)

    distilbert = ItemKNNCBFRecommender(URM_train=URM_train,ICM_train=ICMs[4],verbose=False)
    distilbert.fit(similarity= 'asymmetric', topK= 921, shrink= 1, asymmetric_alpha= 0.774522157812755) 
    recs.append(distilbert)

    kenneth = ItemKNNCBFRecommender(URM_train=URM_train,ICM_train=ICMs[5],verbose=False)
    kenneth.fit(similarity= 'asymmetric', topK= 303, shrink= 574, asymmetric_alpha= 1.7852169782747023) 
    recs.append(kenneth)

    emotion = ItemKNNCBFRecommender(URM_train=URM_train,ICM_train=ICMs[6],verbose=False)
    emotion.fit(similarity= 'euclidean', topK= 1099, shrink= 752, normalize_avg_row= True, similarity_from_distance_mode= 'lin', normalize= False) 
    recs.append(emotion)

    logging.info('Building recsys features ... ')
    recsys_features = build_recsys_features(history=history,behaviors=behaviors,articles=articles,recs=recs)
    
    return recsys_features

def build_urm_ner_scores(behaviors: pl.DataFrame, history: pl.DataFrame, articles: pl.DataFrame, recs: list[BaseRecommender], batch_size=BATCH_SIZE):
    user_id_mapping = build_user_id_mapping(history)
    ap = build_articles_with_processed_ner(articles)
    ner_mapping = build_ner_mapping(ap)
    ap = ap.with_columns(
        pl.col('ner_clusters').list.eval(pl.element().replace(ner_mapping['ner'], ner_mapping['ner_index'], default=None)).list.drop_nulls().alias('ner_clusters_index'),
    )
    
    df = behaviors.rename({'article_ids_inview': 'candidate_ids'})\
        .with_columns(
            pl.col('candidate_ids').list.eval(pl.element().replace(ap['article_id'], ap['ner_clusters_index'], default=[])).alias('candidate_ner_index'),
            pl.col('user_id').replace(user_id_mapping['user_id'], user_id_mapping['user_index'], default=None).alias('user_index'),
        ).select('impression_id', 'user_id', 'user_index', 'candidate_ids', 'candidate_ner_index').sort('user_id') 
          
    df = reduce_polars_df_memory_size(df)
    
    df = pl.concat([ #remove caidates with empty ner_index list
        slice.explode(['candidate_ids', 'candidate_ner_index'])\
            .filter(pl.col('candidate_ner_index').list.len() > 0)\
            .group_by(['impression_id', 'user_id', 'user_index']).agg(pl.all())
        for slice in tqdm(df.iter_slices(batch_size), total=df.shape[0]//batch_size)
    ])
    all_items = ner_mapping['ner_index'].unique().sort().to_list()
    df = pl.concat([
        slice.explode(['candidate_ids', 'candidate_ner_index']).with_columns(
                *[pl.col('candidate_ner_index').list.eval(
                    pl.element().replace(all_items, rec._compute_item_score(user_index, all_items)[0, all_items], default=None)
                ).alias(f"{rec.RECOMMENDER_NAME}_ner_scores") for rec in recs]
            ).drop('user_index', 'candidate_ner_index')\
            .group_by(['impression_id', 'user_id']).agg(pl.all())
        for user_index, slice in tqdm(df.partition_by(['user_index'], as_dict=True).items(), total=df['user_index'].n_unique())
    ])
    
    scores_cols = [col for col in df.columns if col.endswith('_ner_scores')]
    df = df.with_columns(
            *[pl.col(col).list.eval(pl.element().list.sum()).alias(f'sum_{col}') for col in scores_cols],
            *[pl.col(col).list.eval(pl.element().list.max()).alias(f'max_{col}') for col in scores_cols],
            *[pl.col(col).list.eval(pl.element().list.mean()).alias(f'mean_{col}') for col in scores_cols],
        ).drop(scores_cols)\
        .rename({'candidate_ids': 'article'})
    return df

def build_recsys_features(history: pl.DataFrame, behaviors: pl.DataFrame, articles: pl.DataFrame, recs: list[BaseRecommender], save_path:Path=None ):
    start_time = time.time()
    user_id_mapping = build_user_id_mapping(history)
    item_mapping = build_item_mapping(articles)
    
    recsys_scores = behaviors\
            .select('impression_id', 'article_ids_inview', 'user_id')\
            .explode('article_ids_inview')\
            .unique()\
            .rename({'article_ids_inview': 'article_id'})\
            .join(item_mapping, on='article_id')\
            .join(user_id_mapping, on='user_id')\
            .sort(['user_index', 'item_index'])\
            .rename({'article_id': 'article'})\
            .group_by('user_index').map_groups(lambda df: df.pipe(_compute_recommendations, recommenders=recs))\
            .drop('user_index')\
            .drop('item_index')
            
    recsys_scores = reduce_polars_df_memory_size(recsys_scores)

    if save_path:
        print(f'Saving scores features ... [{save_path}]')
        recsys_scores.write_parquet(save_path / 'recsys_scores_features.parquet')

    print(f'Built recsys scores features in {((time.time() - start_time)/60):.1f} minutes')
    return recsys_scores

    
def create_embeddings_icms(input_path, articles):
    parent_path = input_path.parent
        

    contrastive_vector_2 = pl.read_parquet(parent_path.joinpath('Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet'))
    w_2_vec = pl.read_parquet(parent_path.joinpath('Ekstra_Bladet_word2vec/document_vector.parquet'))
    roberta = pl.read_parquet(parent_path.joinpath('FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet'))
    google_bert = pl.read_parquet(parent_path.joinpath('google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet'))
    distilbert = pl.read_parquet(parent_path.joinpath('distilbert_title_embedding.parquet'))
    kenneth = pl.read_parquet(parent_path.joinpath('kenneth_embedding.parquet'))
    emotions = pl.read_parquet(parent_path.joinpath('emotions_embedding.parquet'))

    articles_mapping = articles.select('article_id').with_row_index().rename({'index': 'article_index'})
    
    associations = {
            'contrastive_vector' : contrastive_vector_2,
            'document_vector': w_2_vec,
            'google-bert/bert-base-multilingual-cased': google_bert,
            'FacebookAI/xlm-roberta-base': roberta,
            'title_embedding': distilbert,
            'kenneth_title+subtitle': kenneth,
            'emotion_scores': emotions
    }

    ICMs = []
    for k,value in associations.items():
        ICM_dataframe = value.join(articles, on='article_id').select(['article_id',k]).with_columns(
        pl.col(k).apply(lambda lst : list(range(len(lst))), return_dtype=pl.List(pl.Int64)).alias("indici")      
        )\
        .explode([k,'indici'])\
        .rename({'indici': 'feature_id'})\
        .join(articles_mapping, on='article_id')\
        .drop('article_id')

        n_articles = ICM_dataframe.select('article_index').n_unique()
        n_features = ICM_dataframe.select('feature_id').n_unique()

        ICM = sps.csr_matrix((ICM_dataframe[k].to_numpy(),
                (ICM_dataframe["article_index"].to_numpy(), ICM_dataframe["feature_id"].to_numpy())),
                shape = (n_articles, n_features))

        ICMs.append(ICM)

    return ICMs
            

def _compute_recommendations(user_items_df, recommenders):
    user_index = user_items_df['user_index'].to_list()[0]
    items = user_items_df['item_index'].to_numpy()

    scores = []
    for rec in recommenders:
        scores.append(rec._compute_item_score([user_index], items)[0, items])

    return user_items_df.with_columns(
        [
            pl.Series(model).alias(f'recs{index}') for index, model in enumerate(scores)
        ]
    )
    

def rename_icms(recsys_features):

    couple = {
        'recs0': 'constrastive_emb_icm',
        'recs1': 'w_2_vec_emb_icm',
        'recs2': 'bert_emb_icm',
        'recs3': 'roberta_emb_icm',
        'recs4': 'distilbert_emb_icm',
        'recs5': 'kenneth_emb_icm',
        'recs6': 'emotions_emb_icm'
    }

    for col in recsys_features.columns:
        if couple.get(col,None) != None:
            recsys_features = recsys_features.rename({col: couple[col]})

    return reduce_polars_df_memory_size(recsys_features)