python ~/RecSysChallenge2024/src/polimi/scripts/tree_models_tuning.py \
    -output_dir /home/ubuntu/experiments \
    -training_dataset_path /home/ubuntu/experiments/feature_selection_datasets/small/train \
    -validation_dataset_path /home/ubuntu/experiments/feature_selection_datasets/small/validation \
    -n_trials 500 \
    -study_name lgbm_classifier_featureselection_t60_noK \
    -storage mysql+pymysql://admin:MLZwrgaib8iha7DU9jgP@recsys2024-db.crio26omekmi.eu-west-1.rds.amazonaws.com/recsys2024 \
    -model_name lgbm \
   # --is_rank


python /Users/lorecampa/Desktop/Projects/RecSysChallenge2024/src/polimi/scripts/tree_models_tuning.py \
    -output_dir /Users/lorecampa/Desktop/Projects/RecSysChallenge2024/experiments/trash \
    -training_dataset_path /Users/lorecampa/Desktop/Projects/RecSysChallenge2024/dataset/preprocessing/subsample_train_small_new \
    -validation_dataset_path /Users/lorecampa/Desktop/Projects/RecSysChallenge2024/dataset/preprocessing/preprocessing_validation_small_new \
    -n_trials 1 \
    -study_name test \
    -model_name xgb