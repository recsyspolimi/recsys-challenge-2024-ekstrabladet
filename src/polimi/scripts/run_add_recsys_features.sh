python ~/RecSysChallenge2024/src/polimi/scripts/add_recsys_features.py \
    -behaviors_train /home/ubuntu/dataset/ebnerd_small/train/behaviors.parquet \
    -behaviors_val /home/ubuntu/dataset/ebnerd_small/validation/behaviors.parquet \
    -history_train /home/ubuntu/dataset/ebnerd_small/train/history.parquet\
    -history_val /home/ubuntu/dataset/ebnerd_small/validation/history.parquet\
    -articles /home/ubuntu/dataset/ebnerd_small/articles.parquet\
    -preprocessing_path /home/ubuntu/experiments/preprocessing_train_small_new/train_ds.parquet \
    -embeddings_directory /home/ubuntu/dataset/embeddings \
    -output_path /home/ubuntu/
    