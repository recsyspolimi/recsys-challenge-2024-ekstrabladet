python ~/RecSysChallenge2024/src/polimi/scripts/create_lgbm_dataset.py \
    -output_dir /mnt/ebs_volume/experiments/subsample_train_with_recsys_lgbm \
    -train_dataset_path /mnt/ebs_volume/experiments/subsample_train_new_with_recsys \
    -num_bins 419

python ~/RecSysChallenge2024/src/polimi/scripts/create_lgbm_dataset.py \
    -output_dir /mnt/ebs_volume/experiments/subsample_complete_new_with_recsys_lgbm \
    -train_dataset_path /mnt/ebs_volume/experiments/subsample_complete_new_with_recsys \
    -num_bins 419

python ~/RecSysChallenge2024/src/polimi/scripts/create_lgbm_dataset.py \
    -output_dir /mnt/ebs_volume/experiments/preprocessing_train_with_recsys_lgbm \
    -train_dataset_path /mnt/ebs_volume/experiments/preprocessing_train_new_with_recsys \
    -num_bins 36 \
    --ranker_dataset

python ~/RecSysChallenge2024/src/polimi/scripts/create_lgbm_dataset.py \
    -output_dir /mnt/ebs_volume/experiments/preprocessing_complete_with_recsys_lgbm \
    -train_dataset_path /mnt/ebs_volume/experiments/preprocessing_train_new_with_recsys \
    -val_dataset_path /mnt/ebs_volume/experiments/preprocessing_val_new_with_recsys \
    -num_bins 36 \
    --ranker_dataset

python ~/RecSysChallenge2024/src/polimi/scripts/create_lgbm_dataset.py \
    -output_dir /mnt/ebs_volume/experiments/preprocessing_train_new_lgbm \
    -train_dataset_path /mnt/ebs_volume/experiments/preprocessing_train_new \
    -num_bins 36 \
    --ranker_dataset

python ~/RecSysChallenge2024/src/polimi/scripts/create_lgbm_dataset.py \
    -output_dir /mnt/ebs_volume/experiments/preprocessing_complete_new_lgbm \
    -train_dataset_path /mnt/ebs_volume/experiments/preprocessing_train_new \
    -val_dataset_path /mnt/ebs_volume/experiments/preprocessing_validation_new \
    -num_bins 36 \
    --ranker_dataset