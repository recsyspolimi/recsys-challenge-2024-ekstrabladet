# WARNING
# prima di runnare 147f, assicurarci di aver già:
#   - run_create_urm.sh
#   - run_train_recommenders.sh
# da passare come parametri

python ~/RecSysChallenge2024/src/polimi/scripts/preprocessing.py \
    -output_dir /home/ubuntu/experiments \
    -dataset_path /home/ubuntu/dataset/ebnerd_small \
    -dataset_type train \
    -preprocessing_version new_click \
    -previous_version /home/ubuntu/experiments/preprocessing_train_small_new/train_ds.parquet \
    -emb_scores_path /home/ubuntu/features/embedding_scores/ebnerd_small/train \
    -urm_ner_path /home/ubuntu/features/urm_ner_scores/ebnerd_small/train\
    -emotion_emb_path /home/ubuntu/dataset/emotions_embedding.parquet \
    -click_predictors_path /home/ubuntu/dataset/click_predictors/train_click_predictor.parquet \
    -rec_sys_path /home/ubuntu/franci_features/features/small/train/recsys_scores_features.parquet


wait 10

# python ~/RecSysChallenge2024/src/polimi/scripts/batch_preprocessing.py \
#     -output_dir ~/experiments \
#     -dataset_path /mnt/ebs_volume/recsys2024/dataset/ebnerd_demo \
#     -dataset_type validation \
#     -urm_ner_path /mnt/ebs_volume/recsys2024/urm/ner/small \
#     -ners_models_path /mnt/ebs_volume/recsys2024/algo/ner/small \
#     -recsys_urm_path /mnt/ebs_volume/recsys2024/urm/recsys/small/\
#     -recsys_models_path /mnt/ebs_volume/recsys2024/algo/recsys/small/\
#     -preprocessing_version 127f

# wait 10

# python ~/RecSysChallenge2024/src/polimi/scripts/batch_preprocessing.py \
#     -output_dir ~/experiments \
#     -dataset_path /mnt/ebs_volume/recsys2024/dataset/ebnerd_small \
#     -dataset_type test \
#     -urm_ner_path /mnt/ebs_volume/recsys2024/urm/ner/small \
#     -ners_models_path /mnt/ebs_volume/recsys2024/algo/ner/small \
#     -recsys_urm_path /mnt/ebs_volume/recsys2024/urm/recsys/small/\
#     -recsys_models_path /mnt/ebs_volume/recsys2024/algo/recsys/small/\
#     -preprocessing_version 147f




