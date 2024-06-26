import logging
from pathlib import Path
from datetime import datetime
import argparse
import polars as pl
from polimi.utils._urm import build_user_id_mapping, build_ner_mapping, build_ner_urm,build_recsys_urm,build_item_mapping, build_articles_with_processed_ner, compute_sparsity_ratio
from polimi.utils._custom import save_sparse_csr, read_json
import os
import scipy.sparse as sps
from RecSys_Course_AT_PoliMi.Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from polimi.utils._urm import build_recsys_features

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"

def main(preprocessing_path: Path, behaviors_train: Path, behaviors_val : Path, history_train: Path, history_val: Path,articles: Path, embeddings_path: Path, output_dir: Path):
    logging.info(f"Loading the preprocessed dataset from {preprocessing_path}")
    
    articles = pl.read_parquet(articles)

    behaviors_train = pl.read_parquet(behaviors_train)
    history_train = pl.read_parquet(history_train)

    behaviors_val = pl.read_parquet(behaviors_val)
    history_val = pl.read_parquet(history_val)

    
    embeddings = []
    for file_name in os.listdir(embeddings_path):
        if os.path.isfile(embeddings_path.joinpath(file_name)):
            emb = pl.read_parquet(embeddings_path.joinpath(file_name))
            embeddings.append(emb)
        else:
            continue

    articles_mapping = articles.select('article_id').with_row_index().rename({'index': 'article_index'})

    ICMs = []

    print(ICMs)

    associations = {
        'contrastive_vector' : contrastive_vector_2,
        'document_vector': w_2_vec,
        'google-bert/bert-base-multilingual-cased': google_bert,
        'FacebookAI/xlm-roberta-base': roberta,
        'title_embedding': distilbert,
        'kenneth_title+subtitle': kenneth,
        'emotion_scores': emotions  
    }

    for k,value in associations.items():
        ICM_dataframe = value.join(articles, on='article_id').select(['article_id',k]).with_columns(
        pl.col(k).apply(lambda lst : list(range(len(lst)))).alias("indici")      
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


    item_mapping = build_item_mapping(articles)
    user_id_mapping = build_user_id_mapping(history_train.vstack(history_val))

    URM_train = build_recsys_urm(history_train, user_id_mapping, item_mapping, 'article_id_fixed')

    recs = []

    bert = ItemKNNCBFRecommender(URM_train=URM_train,ICM_train=ICMs[2])
    bert.fit(similarity= 'euclidean', topK= 1457, shrink= 329, normalize_avg_row= True, similarity_from_distance_mode= 'exp', normalize= False) 
    recs.append(bert)

    contrastive = ItemKNNCBFRecommender(URM_train=URM_train,ICM_train=ICMs[6])
    contrastive.fit(similarity= 'asymmetric', topK= 192, shrink= 569, asymmetric_alpha= 0.9094884938503743) 
    recs.append(contrastive)

    emotion = ItemKNNCBFRecommender(URM_train=URM_train,ICM_train=ICMs[5])
    emotion.fit(similarity= 'euclidean', topK= 1099, shrink= 752, normalize_avg_row= True, similarity_from_distance_mode= 'lin', normalize= False) 
    recs.append(emotion)

    roberta = ItemKNNCBFRecommender(URM_train=URM_train,ICM_train=ICMs[3])
    roberta.fit(similarity= 'cosine', topK= 363, shrink= 29) 
    recs.append(roberta)

    w_2_vec = ItemKNNCBFRecommender(URM_train=URM_train,ICM_train=ICMs[4])
    w_2_vec.fit(similarity= 'cosine', topK= 359, shrink= 562) 
    recs.append(w_2_vec)

    kenneth = ItemKNNCBFRecommender(URM_train=URM_train,ICM_train=ICMs[0])
    kenneth.fit(similarity= 'asymmetric', topK= 303, shrink= 574, asymmetric_alpha= 1.7852169782747023) 
    recs.append(kenneth)

    distilbert = ItemKNNCBFRecommender(URM_train=URM_train,ICM_train=ICMs[1])
    distilbert.fit(similarity= 'asymmetric', topK= 921, shrink= 1, asymmetric_alpha= 0.774522157812755) 
    recs.append(distilbert)



    recsys_features = build_recsys_features(history=history_train.vstack(history_val),behaviors=behaviors_train,articles=articles,recs=recs)

    
    

    couple = {
        'recs0': 'emb_kenneth_icm_recsys',
        'recs1': 'emb_distilbert_icm_recsys',
        'recs2': 'emb_bert_icm_recsys',
        'recs3': 'emb_roberta_icm_recsys',
        'recs4': 'emb_w_2_vec_icm_recsys',
        'recs5': 'emb_emotions_icm_recsys',
        'recs6': 'emb_contrastive_icm_recsys'
    }


    for col in recsys_features.columns:
        if couple[col] != None:
            recsys_features = recsys_features.rename({col: couple[col]})

    
    prepro = pl.read_parquet(preprocessing_path)
    
    new_prepro = prepro.join(
        recsys_features, on=['impression_id', 'user_id', 'article'], how='left')
    

    NORMALIZE_OVER_USER_ID = [
        'emb_bert_icm_recsys',
        'emb_contrastive_icm_recsys',
        'emb_emotions_icm_recsys',
        'emb_roberta_icm_recsys',
        'emb_w_2_vec_icm_recsys',
        'emb_kenneth_icm_recsys',
        'emb_distilbert_icm_recsys'
    ]
    NORMALIZE_OVER_ARTICLE = [
        'emb_bert_icm_recsys',
        'emb_contrastive_icm_recsys',
        'emb_emotions_icm_recsys',
        'emb_roberta_icm_recsys',
        'emb_w_2_vec_icm_recsys',
        'emb_kenneth_icm_recsys',
        'emb_distilbert_icm_recsys'
    ]


    new_prepro = new_prepro.with_columns(
    
        *[(pl.col(c) / pl.col(c).max().over(pl.col('user_id'))).alias(f'{c}_l_inf_user_id')
        for c in NORMALIZE_OVER_USER_ID],
        *[pl.col(c).std().over(pl.col('user_id')).alias(f'std_user_id_{c}')
        for c in NORMALIZE_OVER_USER_ID],
        *[pl.col(c).skew().over(pl.col('user_id')).alias(f'skew_user_id_{c}')
        for c in NORMALIZE_OVER_USER_ID],
        *[pl.col(c).kurtosis().over(pl.col('user_id')).alias(f'kurtosis_user_id_{c}')
        for c in NORMALIZE_OVER_USER_ID],
        *[pl.col(c).entropy().over(pl.col('user_id')).alias(f'entropy_user_id_{c}')
        for c in NORMALIZE_OVER_USER_ID],
        *[(pl.col(c) - pl.col(c).median().over(pl.col('user_id'))).alias(f'{c}_minus_median_user_id')
        for c in NORMALIZE_OVER_USER_ID],

        *[(pl.col(c) / pl.col(c).max().over(pl.col('article'))).alias(f'{c}_l_inf_article')
        for c in NORMALIZE_OVER_ARTICLE],
        *[pl.col(c).std().over(pl.col('article')).alias(f'std_article_{c}')
        for c in NORMALIZE_OVER_ARTICLE],
        *[pl.col(c).skew().over(pl.col('article')).alias(f'skew_article_{c}')
        for c in NORMALIZE_OVER_ARTICLE],
        *[pl.col(c).kurtosis().over(pl.col('article')).alias(f'kurtosis_article_{c}')
        for c in NORMALIZE_OVER_ARTICLE],
        *[pl.col(c).entropy().over(pl.col('article')).alias(f'entropy_article_{c}')
        for c in NORMALIZE_OVER_ARTICLE],
        *[(pl.col(c) - pl.col(c).median().over(pl.col('article'))).alias(f'{c}_minus_median_article')
        for c in NORMALIZE_OVER_ARTICLE],


    )

    new_prepro.write_parquet(output_dir)

    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adding recsys features ...")
    parser.add_argument("-preprocessing_path", default=None, type=str, required=True,
                        help="Directory where the already created preprocessing is placed")
    parser.add_argument("-behaviors_train", default=None, type=str,required=True,
                        help="Specify the behaviors train.")
    parser.add_argument("-behaviors_val", default=None, type=str,required=True,
                        help="Specify the behaviors validation.")
    parser.add_argument("-history_train", default=None, type=str,required=True,
                        help="Specify the history train.")
    parser.add_argument("-history_val", default=None, type=str,required=True,
                        help="Specify the history validation.")
    parser.add_argument("-articles", default=None, type=str,required=True,
                        help="Specify the articles.")
    parser.add_argument("-embeddings_directory", default=None, type=str,required=True,
                        help="Specify the directory where the embeddings can be found")
    parser.add_argument("-output_path", default=None, type=str,required=True,
                        help="Specify the directory where the new preprocessing is placed")
    
    args = parser.parse_args()
    PREPROCESSING_DIR = Path(args.preprocessing_path)
    BEHAVIORS_TRAIN = Path(args.behaviors_train)
    BEHAVIORS_VAL = Path(args.behaviors_val)
    HISTORY_TRAIN = Path(args.history_train)
    HISTORY_VAL = Path(args.history_val)
    EMBEDDINGS_DIR = Path(args.embeddings_directory)
    ARTICLES = Path(args.articles)
    OUTPUT_DIR = Path(args.output_path)
    
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR.joinpath("log.txt")    
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(PREPROCESSING_DIR,BEHAVIORS_TRAIN,BEHAVIORS_VAL,HISTORY_TRAIN,HISTORY_VAL,ARTICLES,EMBEDDINGS_DIR,OUTPUT_DIR)
