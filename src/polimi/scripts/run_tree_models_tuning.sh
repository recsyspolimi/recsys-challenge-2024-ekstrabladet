python ~/RecSysChallenge2024/src/polimi/scripts/tree_models_tuning.py \
    -output_dir /home/ubuntu/experiments \
    -training_dataset_path /home/ubuntu/experiments/preprocessing_train_small_127 \
    -validation_dataset_path /home/ubuntu/experiments/preprocessing_validation_small_127 \
    -n_trials 500 \
    -study_name catboost_ranker_127 \
    -storage mysql+pymysql://admin:MLZwrgaib8iha7DU9jgP@recsys2024-db.crio26omekmi.eu-west-1.rds.amazonaws.com/recsys2024 \
    -model_name catboost \
    --is_rank

python ~/RecSysChallenge2024/src/polimi/scripts/tree_models_tuning.py \
    -output_dir /home/ubuntu/experiments \
    -training_dataset_path /home/ubuntu/experiments/subsample_train_small_142 \
    -validation_dataset_path /home/ubuntu/experiments/preprocessing_validation_small_142 \
    -n_trials 500 \
    -study_name catboost_classifier_142 \
    -storage mysql+pymysql://admin:MLZwrgaib8iha7DU9jgP@recsys2024-db.crio26omekmi.eu-west-1.rds.amazonaws.com/recsys2024 \
    -model_name catboost

python ~/RecSysChallenge2024/src/polimi/scripts/tree_models_tuning.py \
    -output_dir /home/ubuntu/experiments \
    -training_dataset_path /home/ubuntu/experiments/preprocessing_train_small_142 \
    -validation_dataset_path /home/ubuntu/experiments/preprocessing_validation_small_142 \
    -n_trials 500 \
    -study_name xgboost_ranker_142 \
    -storage mysql+pymysql://admin:MLZwrgaib8iha7DU9jgP@recsys2024-db.crio26omekmi.eu-west-1.rds.amazonaws.com/recsys2024 \
    -model_name xgb \
    --is_rank

python ~/RecSysChallenge2024/src/polimi/scripts/tree_models_tuning.py \
    -output_dir /home/ubuntu/experiments \
    -training_dataset_path /home/ubuntu/experiments/subsample_train_small_142 \
    -validation_dataset_path /home/ubuntu/experiments/preprocessing_validation_small_142 \
    -n_trials 500 \
    -study_name xgboost_classifier_142 \
    -storage mysql+pymysql://admin:MLZwrgaib8iha7DU9jgP@recsys2024-db.crio26omekmi.eu-west-1.rds.amazonaws.com/recsys2024 \
    -model_name xgb

python ~/RecSysChallenge2024/src/polimi/scripts/tree_models_tuning.py \
    -output_dir /home/ubuntu/experiments \
    -training_dataset_path /home/ubuntu/experiments/subsample_train_small_142 \
    -validation_dataset_path /home/ubuntu/experiments/preprocessing_validation_small_142 \
    -n_trials 500 \
    -study_name fastrgf_classifier_142 \
    -storage mysql+pymysql://admin:MLZwrgaib8iha7DU9jgP@recsys2024-db.crio26omekmi.eu-west-1.rds.amazonaws.com/recsys2024 \
    -model_name fast_rgf
