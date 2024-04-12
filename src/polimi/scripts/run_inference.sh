# evaluating a model
python ./polimi/scripts/inference.py \
    -output_dir /home/ubuntu/experiments \
    -dataset_path /home/ubuntu/experiments/preprocessing_validation_2024-04-12_08-51-34 \
    -model_path /home/ubuntu/experiments/Lightgbm_Training_2024-04-11_11-48-30 \
    -eval

wait 10

# producing a submission
python ./polimi/scripts/inference.py \
    -output_dir ~/experiments \
    -dataset_path ~/experiments/preprocessing_test_2024-04-12_09-29-47 \
    -model_path ~/experiments/Lightgbm_Training_2024-04-11_11-48-30 \
    -submit \
    -behaviors_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_testset/test/behaviors.parquet
