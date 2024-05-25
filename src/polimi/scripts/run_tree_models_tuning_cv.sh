# python ~/RecSysChallenge2024/src/polimi/scripts/tree_models_tuning_cv.py \
#     -output_dir /home/ubuntu/experiments \
#     -folds_dataset_path /home/ubuntu/experiments/preprocessing_user_group_5folds \
#     -n_trials 500 \
#     -dataset_path /home/ubuntu/dataset/preprocessed_grouped_dataset.pkl \
#     -study_name catboost_cls_new_group_5fold_v2 \
#     -storage mysql+pymysql://admin:MLZwrgaib8iha7DU9jgP@recsys2024-db.crio26omekmi.eu-west-1.rds.amazonaws.com/recsys2024 \
#     -model_name catboost
#     # --is_rank

python ~/RecSysChallenge2024/src/polimi/scripts/tree_models_tuning_cv.py \
        -output_dir /home/ubuntu/experiments \
        -folds_dataset_path /home/ubuntu/preprocesssing/preprocessing_moving_window_small_new \
        -n_trials 500 \
        -dataset_path /home/ubuntu/dataset/ebnerd_small \
        -study_name catboost_cls_mw_w4_wval2_st2_new_small \
        -storage mysql+pymysql://admin:MLZwrgaib8iha7DU9jgP@recsys2024-db.crio26omekmi.eu-west-1.rds.amazonaws.com/recsys2024 \
        -model_name catboost