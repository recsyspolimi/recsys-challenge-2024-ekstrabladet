python ~/RecSysChallenge2024/src/polimi/scripts/lightgbm_tuning.py \
    -training_dataset_path "/home/ubuntu/experiments_1/subsample_train_small_new" \
    -validation_dataset_path "/home/ubuntu/experiments_1/preprocessing_validation_small_new" \
    -n_trials 500 \
    -study_name "lightgbm_tuning_new_noK" \
    -storage mysql+pymysql://admin:MLZwrgaib8iha7DU9jgP@recsys2024-db.crio26omekmi.eu-west-1.rds.amazonaws.com/recsys2024 \