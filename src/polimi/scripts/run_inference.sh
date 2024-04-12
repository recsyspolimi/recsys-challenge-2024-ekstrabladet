# evaluating a model
python ~/RecSysChallenge2024/src/polimi/scripts/inference.py \
    -output_dir ~/experiments \
    -dataset_path ~/experiments/preprocessing_validation_small_2024-04-12_20-16-31 \
    -model_path ~/experiments/Lightgbm_Training_2024-04-11_11-48-30/model.joblib \
    --eval

wait 10

# producing a submission
python ~/RecSysChallenge2024/src/polimi/scripts/inference.py \
    -output_dir ~/experiments \
    -dataset_path ~/experiments/preprocessing_test_2024-04-12_09-29-47 \
    -model_path ~/experiments/Lightgbm_Training_2024-04-11_11-48-30/model.joblib \
    --submit \
    -behaviors_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_testset/test/behaviors.parquet
