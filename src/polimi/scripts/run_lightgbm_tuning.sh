python ~/RecSysChallenge2024/src/polimi/scripts/lightgbm_tuning.py \
    -training_dataset_path "/home/ubuntu/experiments/subsample_train_small_click" \
    -validation_dataset_path "/home/ubuntu/experiments/preprocessing_validation_2024-05-20_14-30-21" \
    -n_trials 500 \
    -study_name "lightgbm_tuning_click" \
    -storage mysql+pymysql://admin:MLZwrgaib8iha7DU9jgP@recsys2024-db.crio26omekmi.eu-west-1.rds.amazonaws.com/recsys2024 \