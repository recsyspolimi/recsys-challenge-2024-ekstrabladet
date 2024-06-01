python ~/RecSysChallenge2024/src/polimi/scripts/nn_tuning.py \
    -output_dir /home/ubuntu/experiments \
    -training_dataset_path /home/ubuntu/experiments/subsample_train_small_new \
    -validation_dataset_path /home/ubuntu/experiments/preprocessing_validation_small_new \
    -nn_type wd \
    -n_trials 500 \
    -study_name wide_deep_tuning_new_noK \
    -storage mysql+pymysql://admin:MLZwrgaib8iha7DU9jgP@recsys2024-db.crio26omekmi.eu-west-1.rds.amazonaws.com/recsys2024
