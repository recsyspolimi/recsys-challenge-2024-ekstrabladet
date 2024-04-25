python3 ~/RecSysChallenge2024/src/polimi/scripts/build_embedding_features.py \
    -output_dir ~/experiments \
    -dataset_path ~/dataset/ebnerd_large \
    -feature_name distil_user_item_distance \
    -emdeddings_file ~/dataset/distilbert_title_embedding.parquet

wait 10