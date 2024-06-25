python ~/RecSysChallenge2024/src/polimi/scripts/inference.py \
   -output_dir ~/experiments \
   -dataset_path /mnt/ebs_volume/experiments/preprocessing_test_new \
   -model_path /mnt/ebs_volume/models/xgb_train_new_noK/model.joblib \
   -behaviors_path /home/ubuntu/dataset/ebnerd_testset/test/behaviors.parquet \
   -batch_size 1000000 \
   -XGBoost True \
   --submit