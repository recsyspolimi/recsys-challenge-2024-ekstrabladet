# evaluating a model
python ~/RecSysChallenge2024/src/polimi/scripts/inference.py \
    -output_dir ~/experiments \
    -dataset_path ~/experiments/preprocessing_validation_2024-04-12_10-41-47 \
    -model_path ~/experiments/Lightgbm_Training_2024-04-11_11-48-30/model.joblib \
    -batch_size 1000000 \
    --eval

wait 10

# to verify the submission format quickly
python ~/RecSysChallenge2024/src/polimi/scripts/inference.py \
    -output_dir ~/experiments \
    -dataset_path ~/experiments/preprocessing_validation_demo_2024-04-12_19-46-34 \
    -model_path ~/experiments/Lightgbm_Training_2024-04-11_11-48-30/model.joblib \
    -batch_size 1000000 \
    --eval \
    --submit \
    -behaviors_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_demo/validation/behaviors.parquet

wait 10

# producing a submission
python ~/RecSysChallenge2024/src/polimi/scripts/inference.py \
    -output_dir ~/experiments \
    -dataset_path ~/experiments/preprocessing_test_2024-04-13_11-24-19 \
    -model_path ~/experiments/Lightgbm_Training_2024-04-11_11-48-30/model.joblib \
    -batch_size 1000000 \
    --submit \
    -behaviors_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_testset/test/behaviors.parquet
