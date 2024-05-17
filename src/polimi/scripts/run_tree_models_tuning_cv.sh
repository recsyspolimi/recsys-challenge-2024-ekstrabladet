python ~/RecSysChallenge2024/src/polimi/scripts/tree_models_tuning_cv.py \
    -output_dir /home/ubuntu/experiments \
    -folds_dataset_path /home/ubuntu/experiments/preprocessing_user_group_5folds \
    -n_trials 500 \
    -n_folds 5 \
    -study_name catboost_ranker_new_group_5fold \
    -storage mysql+pymysql://admin:MLZwrgaib8iha7DU9jgP@recsys2024-db.crio26omekmi.eu-west-1.rds.amazonaws.com/recsys2024 \
    -model_name catboost \
    --is_rank
