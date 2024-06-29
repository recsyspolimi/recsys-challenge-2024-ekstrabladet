python ~/RecSysChallenge2024/src/polimi/scripts/add_recsys_features.py \
    -behaviors_train ~/RecSysChallenge2024/dataset/ebnerd_demo/train/behaviors.parquet \
    -behaviors_val ~/RecSysChallenge2024/dataset/ebnerd_demo/validation/behaviors.parquet \
    -history_train ~/RecSysChallenge2024/dataset/ebnerd_demo/train/history.parquet\
    -history_val ~/RecSysChallenge2024/dataset/ebnerd_demo/validation/history.parquet\
    -articles ~/RecSysChallenge2024/dataset/ebnerd_demo/articles.parquet\
    -preprocessing_path ~/RecSysChallenge2024/preprocessing/train_ds.parquet \
    -embeddings_directory ~/RecSysChallenge2024/dataset/embeddings \
    