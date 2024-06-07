python ~/RecSysChallenge2024/src/polimi/scripts/catboost_training.py \
    -output_dir /mnt/ebs_volume/models \
    -dataset_path /mnt/ebs_volume/experiments/subsample_complete_new_with_recsys/ \
    -catboost_params_file /home/ubuntu/RecSysChallenge2024/configuration_files/catboost_classifier_recsys_best.json \
    -catboost_verbosity 20 \
    -model_name catboost_classifier_train_val_recsys