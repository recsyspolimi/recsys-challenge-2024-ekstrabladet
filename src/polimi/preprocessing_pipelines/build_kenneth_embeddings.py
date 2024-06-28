from pathlib import Path
import polars as pl
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from scipy.special import softmax
import tqdm
import argparse
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
def row_to_list(row):
    return row.tolist()

def main(PATH, OUTPATH):
    articles = pl.read_parquet(PATH)
    full_title_articles = articles.select(['article_id','title','subtitle'])\
                        .with_columns(
                            pl.concat_str(
                                    [
                                        pl.col('title'),
                                        pl.col('subtitle')
                                    ],
                                    separator=" ",
                                ).alias('full_title')).select(['article_id','full_title'])
    
    # Sentences we want sentence embeddings for
    sentences = full_title_articles['full_title'].to_list()
    article_ids = full_title_articles['article_id'].to_list()

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('KennethTM/MiniLM-L6-danish-encoder')
    model = AutoModel.from_pretrained('KennethTM/MiniLM-L6-danish-encoder')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    kenneth = pd.DataFrame({
    'embeddings': embeddings,
    'articles_id': article_ids
    })

    kenneth['embeddings'] = kenneth.apply(lambda row: row_to_list(row), axis=1)

    Path(OUTPATH).mkdir(parents=True, exist_ok=True)
    kenneth.to_csv(OUTPATH  + '/kenneth.csv')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script for building emotions embeddingd")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the embeddings will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory of the article dataframe")

    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    
    main(DATASET_DIR, OUTPUT_DIR)

