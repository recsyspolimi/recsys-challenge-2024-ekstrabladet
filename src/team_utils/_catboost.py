from tqdm import tqdm
try:
    import polars as pl
except ImportError:
    print("polars not available")
"""
Utils for catboost.
""" 
def add_features_JS_history_topics(train_ds, articles, history):
    """
    Returns train_ds enriched with features computed using the user's history.
    For each impression (user_id, article_id) considers the user's history (composed of "n" articles) and computes "n" Jaccard Similarity values, between the set of
    topics of the article of the impression and the "n" sets of topics of the articles in the user's history.
    Then, these "n" values get aggregated using mean, min, max, std.dev.

     Args:
        train_ds: The training dataset (Can contain any feature, but it MUST contain user_id and article)
        articles: The articles dataset (MUST contain article_id and topics)
        history: The history dataset (MUST contain user_id and article_id_fixed)

    Returns:
        pl.DataFrame: The training dataset with added features
    """
    article_ds= articles.select(["article_id","topics"]).rename({"article_id":"article"})
    history_ds = history.select(["user_id","article_id_fixed"])
    
    df = pl.concat(
    (
    rows.select(["impression_id","user_id","article"]) #Select only useful columns
        .join(article_ds,on="article",how="left") #Add topics of the inview_article
        .join(other = history_ds, on = "user_id",how="left") #Add history of the user
        .explode("article_id_fixed") #explode the user's history
        #For each article of the user's history, add its topics
        .join(other = article_ds.rename({"article":"article_id_fixed","topics":"topics_history"}), on="article_id_fixed",how="left")
        #add the JS between the topics of the article_inview and the topics of the article in the history
        .with_columns(
        (pl.col("topics").list.set_intersection(pl.col("topics_history")).list.len().truediv(
            pl.col("topics").list.set_union(pl.col("topics_history")).list.len()
        )).alias("JS")
        ).group_by(["impression_id","article"]).agg([ #grouping on all the "n" articles in the user's history, compute aggregations of the "n" JS values
            pl.col("JS").mean().alias("mean_JS"),
            pl.col("JS").min().alias("min_JS"),
            pl.col("JS").max().alias("max_JS"),
            pl.col("JS").std().alias("std_JS")]
        )
    for rows in tqdm(train_ds.iter_slices(100), total = train_ds.shape[0] // 100) #Process train_ds in chunks of rows
    )

    )
    return train_ds.join(other = df, on=["impression_id","article"], how="left")

