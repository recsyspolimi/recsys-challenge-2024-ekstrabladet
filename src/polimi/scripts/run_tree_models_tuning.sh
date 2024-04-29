python ~/RecSysChallenge2024/src/polimi/scripts/lightgbm_tuning.py \
    -output_dir /home/ubuntu/experiments \
    -training_dataset_path /home/ubuntu/experiments/preprocessing_train_2024-04-22_21-04-07 \
    -validation_dataset_path /home/ubuntu/experiments/preprocessing_validation_2024-04-22_18-09-12 \
    -n_trials 500 \
    -study_name mlp_tuning_127 \
    -storage mysql+pymysql://admin:MLZwrgaib8iha7DU9jgP@recsys2024-db.crio26omekmi.eu-west-1.rds.amazonaws.com/recsys2024 \
    -model_name catboost \
    --is_rank
