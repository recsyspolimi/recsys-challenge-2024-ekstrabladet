# # evaluating a model
# python ~/RecSysChallenge2024/src/polimi/scripts/inference.py \
#    -output_dir ~/experiments \
#    -dataset_path /mnt/ebs_volume/experiments/preprocessing_validation_new \
#    -model_path /mnt/ebs_volume/models/catboost_new_noK_trial_172_train/model.joblib \
#    -behaviors_path /home/ubuntu/dataset/ebnerd_large/validation/behaviors.parquet \
#    -batch_size 1000000 \
#    --eval
#    #--submit

python ~/RecSysChallenge2024/src/polimi/scripts/inference.py \
   -output_dir ~/experiments \
   -dataset_path /mnt/ebs_volume/experiments/preprocessing_test_new \
   -model_path /mnt/ebs_volume/models/xgb_train_new_noK/model.joblib \
   -behaviors_path /home/ubuntu/dataset/ebnerd_testset/test/behaviors.parquet \
   -batch_size 1000000 \
   -XGBoost True \
   --submit

# python ~/RecSysChallenge2024/src/polimi/scripts/inference.py \
#    -output_dir ~/experiments \
#    -dataset_path /mnt/ebs_volume/experiments/preprocessing_recsys_small/validation_ds.parquet \
#    -model_path /mnt/ebs_volume/models/catboost_classifier_small_recsys_/model.joblib \
#    -behaviors_path /home/ubuntu/dataset/ebnerd_small/validation/behaviors.parquet \
#    -batch_size 1000000 \
#    --eval \
#    --submit
   
    

#wait 10

# # to verify the submission format quickly
# python ~/RecSysChallenge2024/src/polimi/scripts/inference.py \
#     -output_dir ~/experiments \
#     -dataset_path ~/experiments/preprocessing_validation_demo_2024-04-12_19-46-34 \
#     -model_path ~/experiments/Lightgbm_Training_2024-04-11_11-48-30/model.joblib \
#     -batch_size 1000000 \
#     --eval \
#     --submit \
#     -behaviors_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_demo/validation/behaviors.parquet

# wait 10

# # producing a submission
# python ~/RecSysChallenge2024/src/polimi/scripts/inference.py \
#      -output_dir ~/experiments \
#      -dataset_path /home/ubuntu/experiments/preprocessing_validation_small_new \
#      -model_path /mnt/ebs_volume/models/lightgbm_cls_new_trial_289_train/model.joblib \
#      -batch_size 1000000 \
#      --eval \
#      -behaviors_path /home/ubuntu/dataset/ebnerd_small/validation/behaviors.parquet



######################################################################################################################
# # evaluating a model
# python ~/RecSysChallenge2024/src/polimi/scripts/inference_dropping_columns.py \
#     -output_dir ~/experiments \
#     -dataset_path /mnt/ebs_volume/experiments/preprocessing_validation_new \
#     -model_path /mnt/ebs_volume/models/catboost_dropped_features/model.joblib \
#     -batch_size 1000000 \
#     --eval \
#     -behaviors_path /home/ubuntu/dataset/ebnerd_large/validation/behaviors.parquet \
#     # --submit 


# # producing a submission dropping some columns (ignora)
# python ~/RecSysChallenge2024/src/polimi/scripts/inference_dropping_columns.py \
#     -output_dir ~/experiments \
#     -dataset_path /mnt/ebs_volume/experiments/preprocessing_test_new \
#     -model_path /mnt/ebs_volume/models/catboost_dropped_features/model.joblib \
#     -batch_size 1000000 \
#     --submit \
#     -behaviors_path /home/ubuntu/dataset/ebnerd_testset/test/behaviors.parquet