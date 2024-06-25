python ~/RecSysChallenge2024/src/polimi/scripts/catboost_training.py \
    -output_dir  ~/experiments/models \
    -dataset_path ~/experiments/subsample_complete_new_with_recsys/ \
    -catboost_params_file ~/RecSysChallenge2024/configuration_files/catboost_classifier_recsys_best.json \
    -catboost_verbosity 20 \
    -model_name catboost_classifier_train_val_recsys