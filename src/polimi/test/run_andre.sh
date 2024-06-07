python ~/RecSysChallenge2024/src/polimi/scripts/catboost_training.py \
    -output_dir /home/ubuntu/experiments \
    -dataset_path /home/ubuntu/experiments/preprocessing_train_small_new \
    -catboost_params_file /home/ubuntu/RecSysChallenge2024/configuration_files/catboost_ranker_new_noK_95.json \
    -catboost_verbosity 20 \
    -model_name catboost_ranker_andre \
    --ranker