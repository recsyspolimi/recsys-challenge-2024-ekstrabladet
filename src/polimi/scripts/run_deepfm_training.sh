python ~/RecSysChallenge2024/src/polimi/scripts/deepfm_training.py \
    -output_dir /mnt/ebs_volume/models \
    -dataset_path /mnt/ebs_volume/experiments/subsample_train_new_yeo_johnson_fit \
    -params_file /home/ubuntu/RecSysChallenge2024/configuration_files/best_params_deepfm.json \
    -early_stopping_path /home/ubuntu/experiments/early_stopping_ds_yeo_johnson \
    -model_name deepFM_new_features