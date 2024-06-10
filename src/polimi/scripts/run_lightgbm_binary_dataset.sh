python ~/RecSysChallenge2024/src/polimi/scripts/lgbm_training_binary_dataset.py \
    -output_dir /mnt/ebs_volume/models \
    -dataset_path /mnt/ebs_volume/experiments/preprocessing_train_new_lgbm/lgbm_dataset.bin \
    -lgbm_params_file /home/ubuntu/RecSysChallenge2024/configuration_files/lightgbm_ranker_new_noK_457.json \
    -model_name lightgbm_ranker_new_trial_457 \
    --ranker