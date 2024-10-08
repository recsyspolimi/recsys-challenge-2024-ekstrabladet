# WARNING
# prima di runnare 147f, assicurarci di aver già:
#   - run_create_urm.sh
#   - run_train_recommenders.sh
# da passare come parametri

python ~/RecSysChallenge2024/src/polimi/scripts/preprocessing_group_kfold.py \
    -output_dir /home/ubuntu/experiments \
    -dataset_path /home/ubuntu/dataset/ebnerd_small \
    -preprocessing_version new \
    -n_folds 5

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




