from pathlib import Path
import polars as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import tqdm
import argparse


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
  
    tokenizer = AutoTokenizer.from_pretrained("NikolajMunch/danish-emotion-classification")

    model = AutoModelForSequenceClassification.from_pretrained("NikolajMunch/danish-emotion-classification")
    def compute_emotion_score(text):
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        # [disgust, fear, joy, surprise, sadness, anger]
        scores = output[0][0].detach().numpy()
        return softmax(scores).tolist()

    articles_emotions =  pl.concat(
            rows.with_columns(
            pl.struct(['full_title'])\
                .map_elements(lambda x: compute_emotion_score(x['full_title']),return_dtype=pl.List(pl.Float64)).cast(pl.List(pl.Float64)).alias('emotion_scores')
            )
        for rows in tqdm.tqdm(full_title_articles.iter_slices(100), total=full_title_articles.shape[0] // 100))
    
    Path(OUTPATH).mkdir(parents=True, exist_ok=True)
    articles_emotions.drop('full_title').write_parquet(OUTPATH  + '/articles_emotion.parquet')
    

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

