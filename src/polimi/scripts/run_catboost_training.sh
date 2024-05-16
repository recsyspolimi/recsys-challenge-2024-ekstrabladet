python ~/RecSysChallenge2024/src/polimi/scripts/catboost_training.py \
    -output_dir /mnt/ebs_volume/models \
    -dataset_path /mnt/ebs_volume/experiments/subsample_train_new \
    -catboost_params_file /home/ubuntu/RecSysChallenge2024/configuration_files/catboost_baseline_new_features.json \
    -catboost_verbosity 20 \
    -model_name catboost_baseline_new_features